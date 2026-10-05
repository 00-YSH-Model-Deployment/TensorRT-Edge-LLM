/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "runtime/state/contextCache/encoderEmbeddingCache.h"

#include "common/checkMacros.h"
#include "common/logger.h"

#include <algorithm>
#include <exception>
#include <limits>
#include <system_error>
#include <vector>

namespace trt_edgellm
{
namespace rt
{

EncoderEmbeddingCache::EncoderEmbeddingCache(int64_t maxBudgetBytes, bool streamOrdered)
    : mBudgetBytes(maxBudgetBytes)
{
    if (mBudgetBytes <= 0 || !streamOrdered)
    {
        return;
    }

    int device{-1};
    CUDA_CHECK(cudaGetDevice(&device));
    int poolsSupported{0};
    CUDA_CHECK(cudaDeviceGetAttribute(&poolsSupported, cudaDevAttrMemoryPoolsSupported, device));
    if (poolsSupported == 0)
    {
        LOG_WARNING(
            "EncoderEmbeddingCache: device %d has no stream-ordered memory pools; entries use cudaMalloc.", device);
        return;
    }

    cudaMemPoolProps props{};
    props.allocType = cudaMemAllocationTypePinned;
    props.location.type = cudaMemLocationTypeDevice;
    props.location.id = device;
    CUDA_CHECK(cudaMemPoolCreate(&mPool, &props));
    // The default threshold of 0 returns freed pool memory to the driver at every synchronization, so steady-state
    // eviction would reserve memory again on each request. The budget already bounds what the entries hold.
    uint64_t releaseThreshold = std::numeric_limits<uint64_t>::max();
    CUDA_CHECK(cudaMemPoolSetAttribute(mPool, cudaMemPoolAttrReleaseThreshold, &releaseThreshold));
}

EncoderEmbeddingCache::~EncoderEmbeddingCache() noexcept
{
    try
    {
        clear();
    }
    catch (std::exception const& e)
    {
        LOG_ERROR("EncoderEmbeddingCache: failed to release entries: %s", e.what());
    }
    catch (...)
    {
    }
    if (mPool != nullptr)
    {
        cudaError_t const status = cudaMemPoolDestroy(mPool);
        if (status != cudaSuccess)
        {
            LOG_ERROR("EncoderEmbeddingCache: cudaMemPoolDestroy failed: %s", cudaGetErrorString(status));
        }
    }
}

std::optional<std::reference_wrapper<Tensor const>> EncoderEmbeddingCache::lookup(Hash128 key)
{
    if (mBudgetBytes <= 0)
    {
        return std::nullopt;
    }

    auto it = mEntries.find(key);
    if (it == mEntries.end())
    {
        return std::nullopt;
    }

    it->second.lastAccess = std::chrono::steady_clock::now();
    return std::cref(it->second.embedding);
}

void EncoderEmbeddingCache::store(
    Hash128 key, Tensor const& embedding, int64_t numTokens, int64_t hiddenSize, cudaStream_t stream)
{
    if (mBudgetBytes <= 0)
    {
        return;
    }

    // The entry size comes from numTokens and hiddenSize, not the source tensor's capacity, which may be
    // over-allocated when the output embedding is a reshaped view over a max-capacity buffer.
    storeSlice(key, embedding.rawPointer(), numTokens, hiddenSize, embedding.getDataType(), stream);
}

void EncoderEmbeddingCache::storeSlice(Hash128 key, void const* devicePtr, int64_t numTokens, int64_t hiddenSize,
    nvinfer1::DataType dtype, cudaStream_t stream)
{
    if (mBudgetBytes <= 0)
    {
        return;
    }

    auto it = mEntries.find(key);
    if (it != mEntries.end())
    {
        it->second.lastAccess = std::chrono::steady_clock::now();
        return;
    }

    int64_t const entryBytes = numTokens * hiddenSize * static_cast<int64_t>(utils::getTypeSize(dtype));
    if (entryBytes <= 0 || entryBytes > mBudgetBytes)
    {
        return;
    }

    evictUntilFits(entryBytes, stream);

    Tensor cached;
    if (mPool != nullptr)
    {
        void* data{nullptr};
        CUDA_CHECK(cudaMallocFromPoolAsync(&data, static_cast<size_t>(entryBytes), mPool, stream));
        cached = Tensor(data, {numTokens, hiddenSize}, DeviceType::kGPU, dtype, "encoderEmbeddingCacheSlice");
    }
    else
    {
        cached = Tensor({numTokens, hiddenSize}, DeviceType::kGPU, dtype, "encoderEmbeddingCacheSlice");
    }
    CUDA_CHECK(cudaMemcpyAsync(cached.rawPointer(), devicePtr, entryBytes, cudaMemcpyDeviceToDevice, stream));

    EncoderEmbeddingCacheEntry entry{};
    entry.embedding = std::move(cached);
    entry.numTokens = numTokens;
    entry.hiddenSize = hiddenSize;
    entry.lastAccess = std::chrono::steady_clock::now();

    mEntries.emplace(key, std::move(entry));
    mUsedBytes += entryBytes;

    LOG_DEBUG("EncoderEmbeddingCache: stored slice (%lld tokens, %lld bytes). Used: %lld / %lld bytes.",
        static_cast<long long>(numTokens), static_cast<long long>(entryBytes), static_cast<long long>(mUsedBytes),
        static_cast<long long>(mBudgetBytes));
}

void EncoderEmbeddingCache::clear()
{
    if (mPool != nullptr && !mEntries.empty())
    {
        // cudaFree does not wait for pool memory, and no stream orders every past reader of these entries.
        CUDA_CHECK(cudaDeviceSynchronize());
        for (auto& [key, entry] : mEntries)
        {
            CUDA_CHECK(cudaFree(entry.embedding.rawPointer()));
        }
    }
    mEntries.clear();
    mUsedBytes = 0;
}

void EncoderEmbeddingCache::evictUntilFits(int64_t requiredBytes, cudaStream_t stream)
{
    while (mUsedBytes + requiredBytes > mBudgetBytes && !mEntries.empty())
    {
        // Find LRU entry (oldest lastAccess).
        auto oldest = mEntries.begin();
        for (auto it = mEntries.begin(); it != mEntries.end(); ++it)
        {
            if (it->second.lastAccess < oldest->second.lastAccess)
            {
                oldest = it;
            }
        }

        int64_t const freedBytes = oldest->second.embedding.getMemoryCapacity();
        if (mPool != nullptr)
        {
            CUDA_CHECK(cudaFreeAsync(oldest->second.embedding.rawPointer(), stream));
        }
        mUsedBytes -= freedBytes;
        mEntries.erase(oldest);

        LOG_DEBUG("EncoderEmbeddingCache: evicted entry (%lld bytes freed). Used: %lld / %lld bytes.",
            static_cast<long long>(freedBytes), static_cast<long long>(mUsedBytes),
            static_cast<long long>(mBudgetBytes));
    }
}

OpaqueIdentityHasher::OpaqueIdentityHasher(int32_t numThreads)
{
    for (int32_t thread = 1; thread < numThreads; ++thread)
    {
        try
        {
            mWorkers.emplace_back([this]() { workerLoop(); });
        }
        catch (std::system_error const& e)
        {
            LOG_WARNING("OpaqueIdentityHasher: started %zu of %d hashing threads: %s", mWorkers.size() + 1, numThreads,
                e.what());
            break;
        }
    }
}

OpaqueIdentityHasher::~OpaqueIdentityHasher()
{
    {
        std::lock_guard<std::mutex> const lock(mMutex);
        mStop = true;
    }
    mWork.notify_all();
    for (auto& worker : mWorkers)
    {
        worker.join();
    }
}

std::vector<Hash128> OpaqueIdentityHasher::hash(std::vector<std::string_view> const& items)
{
    std::vector<Hash128> results(items.size());
    std::unique_lock<std::mutex> lock(mMutex);
    mItems = &items;
    mResults = results.data();
    mCount = items.size();
    mNextItem = 0;
    mDoneItems = 0;
    lock.unlock();
    mWork.notify_all();

    lock.lock();
    while (mNextItem < mCount)
    {
        size_t const index = mNextItem++;
        lock.unlock();
        results[index] = hashOpaqueIdentity(items[index]);
        lock.lock();
        ++mDoneItems;
    }
    // Workers claim items only while mNextItem < mCount, so once every claimed item is done none touches the
    // caller's vectors again.
    mFinished.wait(lock, [this]() { return mDoneItems == mCount; });
    mItems = nullptr;
    mResults = nullptr;
    mCount = 0;
    mNextItem = 0;
    mDoneItems = 0;
    return results;
}

void OpaqueIdentityHasher::workerLoop()
{
    std::unique_lock<std::mutex> lock(mMutex);
    while (true)
    {
        mWork.wait(lock, [this]() { return mStop || mNextItem < mCount; });
        if (mStop)
        {
            return;
        }
        size_t const index = mNextItem++;
        std::string_view const item = (*mItems)[index];
        Hash128* const result = mResults + index;
        lock.unlock();
        *result = hashOpaqueIdentity(item);
        lock.lock();
        if (++mDoneItems == mCount)
        {
            mFinished.notify_one();
        }
    }
}

} // namespace rt
} // namespace trt_edgellm
