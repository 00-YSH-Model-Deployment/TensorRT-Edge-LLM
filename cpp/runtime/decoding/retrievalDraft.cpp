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

#include "runtime/decoding/retrievalDraft.h"

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <system_error>

namespace trt_edgellm
{
namespace retrieval_draft
{

namespace
{
constexpr char kCorpusMagic[] = {'C', 'O', 'C', 'R'};
constexpr int32_t kCorpusVersion{1};
} // namespace

RetrievalDraft::Trie::Trie()
    : mNodes{{/*token=*/-1, /*count=*/0, /*best=*/-1}}
{
}

void RetrievalDraft::Trie::insert(int32_t const* tokens, int32_t length, int32_t depth)
{
    int32_t node = 0;
    mNodes[0].count += 1;
    for (int32_t i = 0; i < std::min(length, depth); ++i)
    {
        auto const [edge, created] = mEdges.try_emplace(edgeKey(node, tokens[i]), static_cast<int32_t>(mNodes.size()));
        int32_t const child = edge->second;
        if (created)
        {
            mNodes.push_back({tokens[i], 0, -1});
        }
        mNodes[child].count += 1;
        int32_t const best = mNodes[node].best;
        if (best < 0 || mNodes[child].count > mNodes[best].count)
        {
            mNodes[node].best = child;
        }
        node = child;
    }
}

int32_t RetrievalDraft::Trie::find(int32_t const* tokens, int32_t length) const
{
    int32_t node = 0;
    for (int32_t i = 0; i < length; ++i)
    {
        auto const edge = mEdges.find(edgeKey(node, tokens[i]));
        if (edge == mEdges.end())
        {
            return -1;
        }
        node = edge->second;
    }
    return node;
}

bool RetrievalDraft::Trie::confident(int32_t node, RetrievalDraftConfig const& config) const
{
    Node const& matched = mNodes[node];
    return matched.best >= 0 && matched.count >= config.minSupport
        && mNodes[matched.best].count >= config.minShare * matched.count;
}

int32_t RetrievalDraft::Trie::majorityPath(int32_t node, int32_t* out, int32_t capacity) const
{
    int32_t count = 0;
    for (int32_t child = mNodes[node].best; child >= 0 && count < capacity; child = mNodes[child].best)
    {
        out[count++] = mNodes[child].token;
    }
    return count;
}

RetrievalDraft::RetrievalDraft(RetrievalDraftConfig const& config)
    : mConfig(config)
{
}

bool RetrievalDraft::loadCorpus(std::string const& path)
{
    std::error_code error;
    auto const fileSize = std::filesystem::file_size(path, error);
    std::ifstream file(path, std::ios::binary);
    if (error || !file)
    {
        return false;
    }
    char magic[sizeof(kCorpusMagic)];
    int32_t header[3]; // version, sequences, tokens
    if (!file.read(magic, sizeof(magic)) || !std::equal(magic, magic + sizeof(magic), kCorpusMagic)
        || !file.read(reinterpret_cast<char*>(header), sizeof(header)) || header[0] != kCorpusVersion || header[1] < 0
        || header[2] < 0)
    {
        return false;
    }
    int64_t const numSequences = header[1];
    int64_t const numTokens = header[2];
    auto const expectedSize = static_cast<std::uintmax_t>(
        sizeof(magic) + sizeof(header) + (numSequences + 1 + numTokens) * static_cast<int64_t>(sizeof(int32_t)));
    if (fileSize != expectedSize)
    {
        return false;
    }
    std::vector<int32_t> offsets(static_cast<size_t>(numSequences + 1));
    std::vector<int32_t> tokens(static_cast<size_t>(numTokens));
    if (!file.read(
            reinterpret_cast<char*>(offsets.data()), static_cast<std::streamsize>(offsets.size() * sizeof(int32_t)))
        || !file.read(
            reinterpret_cast<char*>(tokens.data()), static_cast<std::streamsize>(tokens.size() * sizeof(int32_t))))
    {
        return false;
    }
    if (offsets.front() != 0 || offsets.back() != numTokens || !std::is_sorted(offsets.begin(), offsets.end())
        || std::any_of(tokens.begin(), tokens.end(), [](int32_t token) { return token < 0; }))
    {
        return false;
    }
    for (int64_t i = 0; i < numSequences; ++i)
    {
        add(tokens.data() + offsets[i], offsets[i + 1] - offsets[i]);
    }
    return true;
}

void RetrievalDraft::add(int32_t const* tokens, int32_t length)
{
    if (length <= 0)
    {
        return;
    }
    mPrefix.insert(tokens, length, length);
    int32_t const depth = mConfig.maxMatch + mConfig.maxContinuation;
    for (int32_t start = 0; start < length; ++start)
    {
        mSuffix.insert(tokens + start, length - start, depth);
    }
    mNumSequences += 1;
    mNumTokens += length;
}

int32_t RetrievalDraft::propose(int32_t const* generated, int32_t generatedLen, int32_t* out, int32_t capacity) const
{
    if (mNumSequences == 0 || generatedLen < 0 || capacity <= 0)
    {
        return 0;
    }
    // The whole output is the most specific context: once it has enough support, its verdict stands.
    int32_t const prefixNode = mPrefix.find(generated, generatedLen);
    if (prefixNode >= 0 && mPrefix.count(prefixNode) >= mConfig.minSupport)
    {
        return mPrefix.confident(prefixNode, mConfig) ? mPrefix.majorityPath(prefixNode, out, capacity) : 0;
    }
    for (int32_t n = std::min(mConfig.maxMatch, generatedLen); n >= std::max(1, mConfig.minMatch); --n)
    {
        int32_t const node = mSuffix.find(generated + generatedLen - n, n);
        if (node >= 0 && mSuffix.confident(node, mConfig))
        {
            return mSuffix.majorityPath(node, out, capacity);
        }
    }
    return 0;
}

int64_t RetrievalDraft::numNodes() const noexcept
{
    return mPrefix.size() + mSuffix.size();
}

} // namespace retrieval_draft
} // namespace trt_edgellm
