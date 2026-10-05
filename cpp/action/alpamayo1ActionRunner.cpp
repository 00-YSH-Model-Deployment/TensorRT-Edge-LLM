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

#include "alpamayo1ActionRunner.h"
#include "action/actionUtils.h"
#include "common/bindingNames.h"
#include "common/checkMacros.h"
#include "common/cudaUtils.h"
#include "common/logger.h"
#include "common/pagedKvTypes.h"
#include "common/trtUtils.h"
#include "kernels/kvCacheUtilKernels/kvCacheUtilsKernels.h"
#include "kernels/posEncoding/initializeCosSinCache.h"
#include "profiling/metrics.h"
#include "profiling/nvtx_wrapper.h"
#include "profiling/timer.h"
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <nlohmann/json.hpp>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

using namespace trt_edgellm;
using namespace nvinfer1;
using Json = nlohmann::json;

namespace trt_edgellm
{
namespace rt
{

namespace
{
constexpr int32_t kDEFAULT_DENOISE_STEPS{10};
// TensorRT requires 256-byte aligned tensor addresses, so every time-grid point gets its own slot and a flow step binds
// time_steps_t0 and time_steps_t1 straight to two slots.
constexpr int64_t kTIME_SLOT_FLOATS{256 / static_cast<int64_t>(sizeof(float))};
} // namespace

bool Alpamayo1ActionRunner::preprocess(LLMGenerationRequest const& request,
    std::vector<std::vector<int32_t>>& batchedInputIds, tokenizer::Tokenizer const* tokenizer)
{
    tokenizer::TokenToRanks const& specialTokens = tokenizer->getSpecialTokensEncoder();
    int32_t const requestBatchSize = static_cast<int32_t>(request.requests.size());
    int32_t const actionBatchSize = static_cast<int32_t>(std::count_if(request.requests.begin(), request.requests.end(),
        [](auto const& req) { return req.pastTrajectory.has_value(); }));
    for (int32_t i = 0; i < requestBatchSize; ++i)
    {
        std::vector<int32_t>& tokenIds = batchedInputIds[i];
        LLMGenerationRequest::Request const& req = request.requests[i];

        if (!req.pastTrajectory)
        {
            LOG_WARNING("Request %d has no past trajectory; skipping.", i);
            continue;
        }

        auto const itStart = specialTokens.find(kTrajHistoryStartStr);
        auto const itPad = specialTokens.find(kTrajHistoryPadStr);
        auto const itEnd = specialTokens.find(kTrajHistoryEndStr);
        if (itStart == specialTokens.end() || itPad == specialTokens.end() || itEnd == specialTokens.end())
        {
            LOG_WARNING(
                "Special tokens <|traj_history_start|>, <|traj_history_pad|>, or <|traj_history_end|> not found in "
                "tokenizer; skipping trajectory tokenization.");
            continue;
        }

        tokenizer::Rank const startId = itStart->second;
        tokenizer::Rank const padId = itPad->second;
        tokenizer::Rank const endId = itEnd->second;

        std::vector<tokenizer::Rank> const actualTokens
            = action_utils::trajectoryToTokenIds(*req.pastTrajectory, mConfig.numTrajTokens, mConfig.trajTokenStart);

        size_t scanIdx = 0;
        while (scanIdx < tokenIds.size())
        {
            if (tokenIds[scanIdx] != startId)
            {
                ++scanIdx;
                continue;
            }
            size_t endMarkerIdx = scanIdx + 1;
            while (endMarkerIdx < tokenIds.size() && tokenIds[endMarkerIdx] == padId)
            {
                ++endMarkerIdx;
            }
            if (endMarkerIdx >= tokenIds.size() || tokenIds[endMarkerIdx] != endId)
            {
                ++scanIdx;
                continue;
            }

            size_t const numPads = endMarkerIdx - scanIdx - 1;
            if (numPads != actualTokens.size())
            {
                LOG_ERROR("Trajectory placeholder token count (%zu) does not match trajectory encoding length (%zu).",
                    numPads, actualTokens.size());
                return false;
            }

            for (size_t k = 0; k < actualTokens.size(); ++k)
            {
                tokenIds[scanIdx + 1 + k] = actualTokens[k];
            }
            // <|traj_history_start|> and <|traj_history_end|>, plus the trajectory tokens
            scanIdx += 2 + actualTokens.size();
        }
    }

    if (actionBatchSize > mMaxActionBatchSize)
    {
        LOG_ERROR("action batch size %d exceeds engine max batch size %d", actionBatchSize, mMaxActionBatchSize);
        return false;
    }

    try
    {
        initializeNoiseTrajectory(mNoiseSeed, actionBatchSize);
    }
    catch (std::exception const& e)
    {
        LOG_ERROR("Initial noise trajectory initialization failed: %s", e.what());
        return false;
    }
    return true;
}

Alpamayo1ActionRunner::Alpamayo1ActionRunner(std::string const& engineDir, std::string const& checkpointDir,
    cudaStream_t stream, KVCacheManager::Config const& kvCacheConfig)
    : mStream(stream)
{
    LOG_DEBUG("Loading action runner from %s", engineDir.c_str());

    std::string actionEnginePath = engineDir + "/action.engine";

    // Load runtime and deserialize engine.
    mRuntime = std::unique_ptr<IRuntime>(createInferRuntime(gLogger));
    ELLM_CHECK(mRuntime, "Failed to create TensorRT runtime");

    mEngine = deserializeCudaEngineFromFile(*mRuntime, actionEnginePath);

    mContext = std::unique_ptr<IExecutionContext>(
        mEngine->createExecutionContext(ExecutionContextAllocationStrategy::kUSER_MANAGED));
    ELLM_CHECK(mContext, "Failed to create execution context");

    bool const profileSet = mContext->setOptimizationProfileAsync(0, stream);
    ELLM_CHECK(profileSet, "Failed to set optimization profile");

    setNonBlockingAuxStreams(mContext.get(), mEngine.get(), mAuxStreams);

    if (char const* graph = std::getenv("EDGELLM_ALPAMAYO_ACTION_CUDA_GRAPH"))
    {
        mUseCudaGraph = std::atoi(graph) != 0;
        LOG_INFO("Alpamayo action CUDA graph %s", mUseCudaGraph ? "on" : "off");
    }

    bool const configParsed = parseModelConfig(engineDir + "/config.json");
    ELLM_CHECK(configParsed, "Failed to parse model config");

    mExternalWeights = std::make_unique<ExternalWeightManager>();
    mExternalWeights->load(engineDir, engineDir + "/config.json", stream, checkpointDir);
    if (mExternalWeights->size() > 0)
    {
        mExternalWeights->bindToContext(*mEngine, *mContext, "action");
    }

    try
    {
        allocateTensors(kvCacheConfig);
    }
    catch (std::exception const& e)
    {
        LOG_ERROR("Alpamayo1ActionRunner tensor allocation failed: %s", e.what());
        throw;
    }

    CUDA_CHECK(cudaStreamSynchronize(stream));
}

Alpamayo1ActionRunner::~Alpamayo1ActionRunner() noexcept
{
    resetDenoiseGraphs();
}

int64_t Alpamayo1ActionRunner::getRequiredContextMemorySize() const
{
    return mEngine ? mEngine->getDeviceMemorySizeV2() : 0;
}

bool Alpamayo1ActionRunner::setContextMemory(rt::Tensor& sharedContextMemory)
{
    int64_t const requiredSize = getRequiredContextMemorySize();
    if (sharedContextMemory.getMemoryCapacity() < requiredSize)
    {
        LOG_ERROR("Shared context memory (%zu bytes) is smaller than action engine requires (%zu bytes)",
            static_cast<size_t>(sharedContextMemory.getMemoryCapacity()), static_cast<size_t>(requiredSize));
        return false;
    }
    mContext->setDeviceMemoryV2(sharedContextMemory.rawPointer(), sharedContextMemory.getMemoryCapacity());
    resetDenoiseGraphs();
    return true;
}

void Alpamayo1ActionRunner::setDynamicInputShapes(int32_t activeBatchSize)
{
    Dims const kvCacheStartIndexShape = {1, {activeBatchSize}};
    Dims const noiseShape = {3, {activeBatchSize, mNumWaypoints, 2}};
    Dims const ropeCosShape = {3, {activeBatchSize, mNumWaypoints, mRopeHeadDim}};
    Dims const ropeIdxShape = {2, {activeBatchSize, mNumWaypoints}};
    Dims const kvShape = {4,
        {activeBatchSize, static_cast<int64_t>(mConfig.numKVHeads), static_cast<int64_t>(mKVCacheCapacity),
            mConfig.headDim}};

    bool status = true;
    status &= mContext->setInputShape(binding_names::kKVCacheStartIndex, kvCacheStartIndexShape);
    status &= mContext->setInputShape(binding_names::kNoiseTrajectory, noiseShape);
    status &= mContext->setInputShape(binding_names::kRopeCosSin, ropeCosShape);
    status &= mContext->setInputShape(binding_names::kAttentionPosId, ropeIdxShape);
    for (int32_t i = 0; i < mConfig.numDecoderLayers; ++i)
    {
        status &= mContext->setInputShape(binding_names::formatKCacheName(i, true).c_str(), kvShape);
        status &= mContext->setInputShape(binding_names::formatVCacheName(i, true).c_str(), kvShape);
    }
    if (!status)
    {
        LOG_ERROR("setDynamicInputShapes failed for activeBatchSize=%d", activeBatchSize);
        throw std::runtime_error("setDynamicInputShapes failed");
    }
}

std::vector<std::vector<FutureTrajectoryPoint>> Alpamayo1ActionRunner::sampleTrajectory(
    cudaStream_t stream, HybridCacheManager const& kvcache, ActionKvBatchView const& batch)
{
    int32_t const activeBatchSize = batch.batchSize;
    TIME_STAGE(metrics::StageNames::kACTION_INFERENCE, stream);
    NVTX_SCOPED_RANGE(nvtx_action_sample,
        ("ACTION_SAMPLE_TRAJECTORY[BS=" + std::to_string(activeBatchSize) + "]").c_str(), nvtx_colors::PURPLE);

    std::vector<std::vector<FutureTrajectoryPoint>> result;
    if (activeBatchSize <= 0 || activeBatchSize > mMaxActionBatchSize)
    {
        LOG_ERROR("Invalid action batch size %d (maximum %d)", activeBatchSize, mMaxActionBatchSize);
        return result;
    }

    if (!reshapeActionTensorsForActiveBatch(activeBatchSize))
    {
        LOG_ERROR("reshapeActionTensorsForActiveBatch failed for activeBatchSize=%d", activeBatchSize);
        return result;
    }

    if (batch.kvLengthsHost.getDeviceType() != DeviceType::kCPU
        || batch.kvLengthsDevice.getDeviceType() != DeviceType::kGPU
        || batch.kvLengthsHost.getDataType() != DataType::kINT32
        || batch.kvLengthsDevice.getDataType() != DataType::kINT32
        || batch.kvLengthsHost.getShape().volume() < activeBatchSize
        || batch.kvLengthsDevice.getShape().volume() < activeBatchSize
        || static_cast<int32_t>(batch.mropeDeltas.size()) != activeBatchSize)
    {
        LOG_ERROR("Invalid action KV batch view for batch size %d", activeBatchSize);
        return result;
    }
    // The engine appends its mNumWaypoints action tokens at kvcache_start_index, inside the same capacity.
    int32_t const* const lengthsHost = batch.kvLengthsHost.dataPointer<int32_t>();
    for (int32_t b = 0; b < activeBatchSize; ++b)
    {
        if (lengthsHost[b] <= 0 || lengthsHost[b] > mKVCacheCapacity - mNumWaypoints)
        {
            LOG_ERROR(
                "Action KV cache length %d for row %d plus %d action tokens exceeds the action engine KV "
                "capacity %d; re-export the action expert with a larger --max-kv-cache-capacity.",
                lengthsHost[b], b, mNumWaypoints, mKVCacheCapacity);
            return result;
        }
    }

    bindStaticTensors(activeBatchSize, const_cast<void*>(batch.kvLengthsDevice.rawPointer()));

    KVCacheManager::Config const& kvConfig = kvcache.getKVCacheManager().getConfig();
    size_t const elemSize = rt::utils::getTypeSize(kvConfig.kvCacheType);
    for (int32_t i = 0; i < mConfig.numDecoderLayers; ++i)
    {
        rt::Tensor& kCacheBlock = mKCacheLayers[static_cast<size_t>(i)];
        rt::Tensor& vCacheBlock = mVCacheLayers[static_cast<size_t>(i)];
        kernel::gatherPagedKVToHeadMajor(kvcache.getCombinedKVCache(i).rawPointer(), kCacheBlock.rawPointer(),
            vCacheBlock.rawPointer(), batch.pageTable.kernelView().dataPointer<int32_t>(),
            batch.kvLengthsDevice.dataPointer<int32_t>(), batch.pageTable.maxPagesPerSeq(), activeBatchSize,
            mKVCacheCapacity, mConfig.numKVHeads, mConfig.headDim, elemSize, stream);
    }

    // Compute position IDs with rope delta offset
    int64_t* mropePosPtr = static_cast<int64_t*>(mRopePositionIdsHost.rawPointer());
    for (int32_t b = 0; b < activeBatchSize; ++b)
    {
        int64_t const basePos = batch.mropeDeltas[static_cast<size_t>(b)] + static_cast<int64_t>(lengthsHost[b]);
        for (int32_t dim = 0; dim < 3; ++dim)
        {
            for (int32_t w = 0; w < mNumWaypoints; ++w)
            {
                mropePosPtr[b * mNumWaypoints * 3 + dim * mNumWaypoints + w] = static_cast<int64_t>(w) + basePos;
            }
        }
    }

    CUDA_CHECK(cudaMemcpyAsync(mRopePositionIdsDevice.rawPointer(), mRopePositionIdsHost.rawPointer(),
        static_cast<size_t>(activeBatchSize) * 3ULL * static_cast<size_t>(mNumWaypoints) * sizeof(int64_t),
        cudaMemcpyHostToDevice, stream));

    kernel::initializeMRopeCosSin(static_cast<float*>(mRopeCosSinDevice.rawPointer()),
        static_cast<int64_t*>(mRopePositionIdsDevice.rawPointer()), mConfig.ropeTheta,
        static_cast<int64_t>(mRopeHeadDim), mNumWaypoints, activeBatchSize, true, mConfig.mropeSectionH,
        mConfig.mropeSectionW, stream);

    // Copy inputs to device
    size_t const trajBytes
        = static_cast<size_t>(activeBatchSize) * static_cast<size_t>(mNumWaypoints) * 2ULL * sizeof(float);
    CUDA_CHECK(cudaMemcpyAsync(mNoiseTrajectoryDevice.rawPointer(), mNoiseTrajectoryHost.rawPointer(), trajBytes,
        cudaMemcpyHostToDevice, stream));

    cudaGraphExec_t graphExec{nullptr};
    if (mUseCudaGraph)
    {
        auto const it = std::find_if(mDenoiseGraphs.begin(), mDenoiseGraphs.end(),
            [activeBatchSize](DenoiseGraph const& g) { return g.batchSize == activeBatchSize; });
        graphExec = it == mDenoiseGraphs.end() ? nullptr : it->exec;
    }
    if (graphExec != nullptr)
    {
        cudaError_t const launched = cudaGraphLaunch(graphExec, stream);
        if (launched != cudaSuccess)
        {
            LOG_ERROR("Action CUDA graph launch failed: %s", cudaGetErrorString(launched));
            return result;
        }
    }
    else if (!enqueueDenoiseSteps(stream))
    {
        return result;
    }

    int32_t const numCalls = static_cast<int32_t>(mTimeGrid.size()) - 1;
    CUDA_CHECK(cudaMemcpyAsync(
        mDenoisedTrajectoryHost.rawPointer(), trajectoryBuffer(numCalls), trajBytes, cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // Extract trajectory points
    result.resize(activeBatchSize);
    float* accelKappa = static_cast<float*>(mDenoisedTrajectoryHost.rawPointer());
    for (int32_t b = 0; b < activeBatchSize; ++b)
    {
        result[b].reserve(mNumWaypoints);
        float const* row = accelKappa + b * (mNumWaypoints * 2);
        for (int32_t w = 0; w < mNumWaypoints; ++w)
        {
            result[b].emplace_back(row[w * 2], row[w * 2 + 1]);
        }
    }

    if (mUseCudaGraph && graphExec == nullptr)
    {
        captureDenoiseGraph(activeBatchSize, stream);
    }
    return result;
}

void Alpamayo1ActionRunner::bindStaticTensors(int32_t activeBatchSize, void* kvLengthsDevice)
{
    if (activeBatchSize == mBoundBatchSize && kvLengthsDevice == mBoundKvLengths)
    {
        return;
    }
    if (kvLengthsDevice != mBoundKvLengths)
    {
        resetDenoiseGraphs();
    }
    mBoundBatchSize = 0;
    setDynamicInputShapes(activeBatchSize);

    bool setEngineIOStatus{true};
    setEngineIOStatus &= mContext->setTensorAddress(binding_names::kKVCacheStartIndex, kvLengthsDevice);
    setEngineIOStatus &= mContext->setTensorAddress(binding_names::kRopeCosSin, mRopeCosSinDevice.rawPointer());
    setEngineIOStatus &= mContext->setTensorAddress(binding_names::kAttentionPosId, mPositionIdsDevice.rawPointer());
    for (int32_t i = 0; i < mConfig.numDecoderLayers; ++i)
    {
        void* const kCacheBlock = mKCacheLayers[static_cast<size_t>(i)].rawPointer();
        void* const vCacheBlock = mVCacheLayers[static_cast<size_t>(i)].rawPointer();
        setEngineIOStatus &= mContext->setTensorAddress(binding_names::formatKCacheName(i, true).c_str(), kCacheBlock);
        setEngineIOStatus &= mContext->setTensorAddress(binding_names::formatVCacheName(i, true).c_str(), vCacheBlock);
        setEngineIOStatus &= mContext->setTensorAddress(binding_names::formatKCacheName(i, false).c_str(), kCacheBlock);
        setEngineIOStatus &= mContext->setTensorAddress(binding_names::formatVCacheName(i, false).c_str(), vCacheBlock);
    }
    if (!setEngineIOStatus)
    {
        LOG_ERROR("Failed to bind action engine input tensors.");
        throw std::runtime_error("Failed to bind input tensors.");
    }
    mBoundBatchSize = activeBatchSize;
    mBoundKvLengths = kvLengthsDevice;
}

void* Alpamayo1ActionRunner::trajectoryBuffer(int32_t k) noexcept
{
    return (k % 2 == 0 ? mNoiseTrajectoryDevice : mDenoisedTrajectoryDevice).rawPointer();
}

bool Alpamayo1ActionRunner::enqueueDenoiseSteps(cudaStream_t stream)
{
    float* const timeSlots = mTimeGridDevice.dataPointer<float>();
    int32_t const numCalls = static_cast<int32_t>(mTimeGrid.size()) - 1;
    for (int32_t k = 0; k < numCalls; ++k)
    {
        bool bound = mContext->setTensorAddress(binding_names::kTimeStepsT0, timeSlots + k * kTIME_SLOT_FLOATS);
        bound &= mContext->setTensorAddress(binding_names::kTimeStepsT1, timeSlots + (k + 1) * kTIME_SLOT_FLOATS);
        bound &= mContext->setTensorAddress(binding_names::kNoiseTrajectory, trajectoryBuffer(k));
        bound &= mContext->setTensorAddress(binding_names::kDenoisedTrajectory, trajectoryBuffer(k + 1));
        if (!bound || !mContext->enqueueV3(stream))
        {
            LOG_ERROR("Action engine enqueue failed at flow step %d of %d", k, numCalls);
            return false;
        }
    }
    return true;
}

void Alpamayo1ActionRunner::captureDenoiseGraph(int32_t activeBatchSize, cudaStream_t stream)
{
    cudaGraph_t graph{nullptr};
    cudaGraphExec_t exec{nullptr};
    bool captured = cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal) == cudaSuccess;
    if (captured)
    {
        bool const enqueued = enqueueDenoiseSteps(stream);
        captured = cudaStreamEndCapture(stream, &graph) == cudaSuccess && enqueued
            && instantiateCudaGraph(&exec, graph) == cudaSuccess;
    }
    if (!captured)
    {
        if (graph != nullptr)
        {
            static_cast<void>(cudaGraphDestroy(graph));
        }
        static_cast<void>(cudaGetLastError());
        LOG_WARNING("Action CUDA graph capture failed; the flow steps keep running through enqueueV3.");
        mUseCudaGraph = false;
        return;
    }
    mDenoiseGraphs.push_back(DenoiseGraph{activeBatchSize, graph, exec});
    LOG_INFO("Captured the action flow loop into a CUDA graph (batch %d, %zu engine calls)", activeBatchSize,
        mTimeGrid.size() - 1U);
}

void Alpamayo1ActionRunner::resetDenoiseGraphs() noexcept
{
    for (DenoiseGraph const& g : mDenoiseGraphs)
    {
        static_cast<void>(cudaGraphExecDestroy(g.exec));
        static_cast<void>(cudaGraphDestroy(g.graph));
    }
    mDenoiseGraphs.clear();
}

bool Alpamayo1ActionRunner::parseModelConfig(std::string const& configPath)
{
    Json jsonConfig;
    std::ifstream configFile(configPath);
    if (!configFile.is_open())
    {
        LOG_ERROR("Failed to open config file: %s", configPath.c_str());
        return false;
    }
    try
    {
        jsonConfig = Json::parse(configFile);
        configFile.close();
    }
    catch (Json::parse_error const& e)
    {
        LOG_ERROR("Failed to parse config file %s with error: %s", configPath.c_str(), e.what());
        return false;
    }

    try
    {
        mConfig.ropeTheta = jsonConfig.at("rope_theta").get<float>();
        mConfig.numDecoderLayers = jsonConfig.at("num_hidden_layers").get<int32_t>();
        mConfig.numTrajTokens = jsonConfig.at("num_traj_tokens").get<int32_t>();
        mConfig.trajTokenStart = jsonConfig.at("traj_token_start").get<int32_t>();
        mConfig.numKVHeads = jsonConfig.at("num_key_value_heads").get<int32_t>();
        mConfig.headDim = jsonConfig.at("head_dim").get<int32_t>();
    }
    catch (Json::exception const& e)
    {
        LOG_ERROR("Failed to read required fields from %s: %s", configPath.c_str(), e.what());
        return false;
    }

    auto const& ropeParams = jsonConfig.contains("rope_scaling") ? jsonConfig["rope_scaling"] : jsonConfig;
    if (ropeParams.contains("mrope_section"))
    {
        auto section = ropeParams["mrope_section"].get<std::vector<int32_t>>();
        if (section.size() >= 3)
        {
            mConfig.mropeSectionH = section[1];
            mConfig.mropeSectionW = section[2];
        }
    }
    if (mConfig.mropeSectionH <= 0 || mConfig.mropeSectionW <= 0)
    {
        LOG_ERROR("Failed to parse mrope_section from action config. Got H=%d, W=%d", mConfig.mropeSectionH,
            mConfig.mropeSectionW);
        return false;
    }

    // Read max_kv_cache_capacity from builder_config section
    if (jsonConfig.contains("builder_config") && jsonConfig["builder_config"].contains("max_kv_cache_capacity"))
    {
        mConfig.maxKVCacheCapacity = jsonConfig["builder_config"]["max_kv_cache_capacity"].get<int32_t>();
    }
    else
    {
        LOG_ERROR("max_kv_cache_capacity not found in builder_config section of %s", configPath.c_str());
        return false;
    }

    return true;
}

void Alpamayo1ActionRunner::parseDenoiseSchedule()
{
    if (char const* grid = std::getenv("EDGELLM_ALPAMAYO_TIME_GRID"))
    {
        try
        {
            mTimeGrid = action_utils::parseFlowTimeGrid(grid);
        }
        catch (std::invalid_argument const& e)
        {
            LOG_ERROR("EDGELLM_ALPAMAYO_TIME_GRID=\"%s\": %s", grid, e.what());
            throw std::runtime_error("invalid EDGELLM_ALPAMAYO_TIME_GRID");
        }
        if (std::getenv("EDGELLM_ALPAMAYO_DENOISE_STEPS") != nullptr
            || std::getenv("EDGELLM_ALPAMAYO_CACHE_STEPS") != nullptr)
        {
            LOG_WARNING(
                "EDGELLM_ALPAMAYO_TIME_GRID overrides EDGELLM_ALPAMAYO_DENOISE_STEPS and EDGELLM_ALPAMAYO_CACHE_STEPS");
        }
    }
    else
    {
        int32_t numDenoiseSteps{kDEFAULT_DENOISE_STEPS};
        if (char const* steps = std::getenv("EDGELLM_ALPAMAYO_DENOISE_STEPS"))
        {
            numDenoiseSteps = std::max(1, std::atoi(steps));
        }
        std::vector<bool> cacheStep(static_cast<size_t>(numDenoiseSteps), false);
        if (char const* cache = std::getenv("EDGELLM_ALPAMAYO_CACHE_STEPS"))
        {
            std::string list(cache);
            size_t pos = 0;
            while (pos < list.size())
            {
                size_t const end = std::min(list.find(',', pos), list.size());
                std::string const item = list.substr(pos, end - pos);
                if (!item.empty())
                {
                    int32_t const step = std::atoi(item.c_str());
                    if (step <= 0 || step >= numDenoiseSteps)
                    {
                        LOG_ERROR("EDGELLM_ALPAMAYO_CACHE_STEPS: step %d must be in [1, %d)", step, numDenoiseSteps);
                        throw std::runtime_error("invalid cached denoise step");
                    }
                    cacheStep[static_cast<size_t>(step)] = true;
                }
                pos = end + 1;
            }
        }
        mTimeGrid = action_utils::foldCachedEulerSteps(cacheStep);
    }
    if (char const* scale = std::getenv("EDGELLM_ALPAMAYO_NOISE_SCALE"))
    {
        mNoiseScale = std::stof(scale);
        if (mNoiseScale < 0.0F)
        {
            throw std::runtime_error("EDGELLM_ALPAMAYO_NOISE_SCALE must be >= 0");
        }
    }
    std::string points;
    for (float const t : mTimeGrid)
    {
        char point[32];
        std::snprintf(point, sizeof(point), "%s%g", points.empty() ? "" : ", ", t);
        points += point;
    }
    LOG_INFO("Alpamayo flow schedule: %zu action engine calls on time grid {%s}, noise scale %.3f",
        mTimeGrid.size() - 1U, points.c_str(), mNoiseScale);
}

void Alpamayo1ActionRunner::allocateTensors(KVCacheManager::Config const& kvCacheConfig)
{
    parseDenoiseSchedule();
    Dims const maxNoise = mEngine->getProfileShape(binding_names::kNoiseTrajectory, 0, OptProfileSelector::kMAX);
    int32_t const maxBatch = std::min(static_cast<int32_t>(maxNoise.d[0]), kvCacheConfig.maxBatchSize);
    mMaxActionBatchSize = maxBatch;

    nvinfer1::Dims noiseShape = mContext->getTensorShape(binding_names::kNoiseTrajectory);
    mNumWaypoints = noiseShape.d[1];

    Dims const maxKCache
        = mEngine->getProfileShape(binding_names::formatKCacheName(0, true).c_str(), 0, OptProfileSelector::kMAX);
    mKVCacheCapacity = static_cast<int32_t>(maxKCache.d[2]);
    ELLM_CHECK(mKVCacheCapacity == mConfig.maxKVCacheCapacity && mKVCacheCapacity > mNumWaypoints,
        format::fmtstr("Action engine KV capacity %d must equal builder_config.max_kv_cache_capacity %d and exceed "
                       "the %d action tokens.",
            mKVCacheCapacity, mConfig.maxKVCacheCapacity, mNumWaypoints));
    rt::Coords noiseCoords({maxBatch, mNumWaypoints, 2});

    mNoiseTrajectoryDevice = rt::Tensor(
        noiseCoords, rt::DeviceType::kGPU, nvinfer1::DataType::kFLOAT, "Alpamayo1ActionRunner::mNoiseTrajectoryDevice");
    mNoiseTrajectoryHost = rt::Tensor(
        noiseCoords, rt::DeviceType::kCPU, nvinfer1::DataType::kFLOAT, "Alpamayo1ActionRunner::mNoiseTrajectoryHost");

    mDenoisedTrajectoryDevice = rt::Tensor(noiseCoords, rt::DeviceType::kGPU, nvinfer1::DataType::kFLOAT,
        "Alpamayo1ActionRunner::mDenoisedTrajectoryDevice");
    mDenoisedTrajectoryHost = rt::Tensor(noiseCoords, rt::DeviceType::kCPU, nvinfer1::DataType::kFLOAT,
        "Alpamayo1ActionRunner::mDenoisedTrajectoryHost");

    rt::Coords const timeGridShape{static_cast<int64_t>(mTimeGrid.size()), kTIME_SLOT_FLOATS};
    mTimeGridHost = rt::Tensor(
        timeGridShape, rt::DeviceType::kCPU, nvinfer1::DataType::kFLOAT, "Alpamayo1ActionRunner::mTimeGridHost");
    mTimeGridDevice = rt::Tensor(
        timeGridShape, rt::DeviceType::kGPU, nvinfer1::DataType::kFLOAT, "Alpamayo1ActionRunner::mTimeGridDevice");
    float* const timeSlots = mTimeGridHost.dataPointer<float>();
    std::fill(timeSlots, timeSlots + timeGridShape.volume(), 0.0F);
    for (size_t k = 0; k < mTimeGrid.size(); ++k)
    {
        timeSlots[static_cast<int64_t>(k) * kTIME_SLOT_FLOATS] = mTimeGrid[k];
    }
    CUDA_CHECK(cudaMemcpyAsync(mTimeGridDevice.rawPointer(), timeSlots,
        static_cast<size_t>(mTimeGridHost.getMemoryCapacity()), cudaMemcpyHostToDevice, mStream));

    nvinfer1::Dims ropeCosSinDims = mContext->getTensorShape(binding_names::kRopeCosSin);
    mRopeHeadDim = ropeCosSinDims.d[2];
    mRopeCosSinDevice = rt::Tensor({maxBatch, mNumWaypoints, mRopeHeadDim}, rt::DeviceType::kGPU,
        nvinfer1::DataType::kFLOAT, "Alpamayo1ActionRunner::mRopeCosSinDevice");

    mPositionIdsHost = rt::Tensor({maxBatch, mNumWaypoints}, rt::DeviceType::kCPU, nvinfer1::DataType::kINT32,
        "Alpamayo1ActionRunner::mPositionIdsHost");
    mPositionIdsDevice = rt::Tensor({maxBatch, mNumWaypoints}, rt::DeviceType::kGPU, nvinfer1::DataType::kINT32,
        "Alpamayo1ActionRunner::mPositionIdsDevice");
    int32_t* const positionIds = mPositionIdsHost.dataPointer<int32_t>();
    for (int32_t b = 0; b < maxBatch; ++b)
    {
        for (int32_t w = 0; w < mNumWaypoints; ++w)
        {
            positionIds[b * mNumWaypoints + w] = w;
        }
    }
    CUDA_CHECK(cudaMemcpyAsync(mPositionIdsDevice.rawPointer(), positionIds,
        static_cast<size_t>(mPositionIdsHost.getMemoryCapacity()), cudaMemcpyHostToDevice, mStream));

    mRopePositionIdsHost = rt::Tensor({maxBatch, 3, mNumWaypoints}, rt::DeviceType::kCPU, nvinfer1::DataType::kINT64,
        "Alpamayo1ActionRunner::mRopePositionIdsHost");
    mRopePositionIdsDevice = rt::Tensor({maxBatch, 3, mNumWaypoints}, rt::DeviceType::kGPU, nvinfer1::DataType::kINT64,
        "Alpamayo1ActionRunner::mRopePositionIdsDevice");

    rt::Coords const kvcacheShape{maxBatch, mConfig.numKVHeads, mKVCacheCapacity, mConfig.headDim};
    mKCacheLayers.resize(static_cast<size_t>(mConfig.numDecoderLayers));
    mVCacheLayers.resize(static_cast<size_t>(mConfig.numDecoderLayers));
    for (int32_t i = 0; i < mConfig.numDecoderLayers; ++i)
    {
        mKCacheLayers[i] = rt::Tensor(
            kvcacheShape, DeviceType::kGPU, kvCacheConfig.kvCacheType, "Alpamayo1ActionRunner::mKCacheLayer");
        mVCacheLayers[i] = rt::Tensor(
            kvcacheShape, DeviceType::kGPU, kvCacheConfig.kvCacheType, "Alpamayo1ActionRunner::mVCacheLayer");
    }
}

bool Alpamayo1ActionRunner::reshapeActionTensorsForActiveBatch(int32_t activeBatchSize)
{
    rt::Coords const noiseShape({activeBatchSize, mNumWaypoints, 2});
    bool ok = true;
    ok &= mNoiseTrajectoryDevice.reshape(noiseShape);
    ok &= mNoiseTrajectoryHost.reshape(noiseShape);
    ok &= mDenoisedTrajectoryDevice.reshape(noiseShape);
    ok &= mDenoisedTrajectoryHost.reshape(noiseShape);
    ok &= mRopeCosSinDevice.reshape({activeBatchSize, mNumWaypoints, mRopeHeadDim});
    ok &= mPositionIdsHost.reshape({activeBatchSize, mNumWaypoints});
    ok &= mPositionIdsDevice.reshape({activeBatchSize, mNumWaypoints});
    ok &= mRopePositionIdsHost.reshape({activeBatchSize, 3, mNumWaypoints});
    ok &= mRopePositionIdsDevice.reshape({activeBatchSize, 3, mNumWaypoints});
    if (!ok)
    {
        LOG_ERROR("Tensor reshape for active batch size failed.");
    }
    return ok;
}

void Alpamayo1ActionRunner::initializeNoiseTrajectory(int32_t randomSeed, int32_t activeBatchSize)
{
    size_t const elemCount = static_cast<size_t>(activeBatchSize) * static_cast<size_t>(mNumWaypoints) * 2ULL;
    float* data = static_cast<float*>(mNoiseTrajectoryHost.rawPointer());
    std::mt19937 generator(static_cast<std::mt19937::result_type>(randomSeed));
    std::normal_distribution<float> dist(0.0f, 1.0f);
    for (size_t i = 0; i < elemCount; ++i)
    {
        data[i] = mNoiseScale * dist(generator);
    }
}

} // namespace rt
} // namespace trt_edgellm
