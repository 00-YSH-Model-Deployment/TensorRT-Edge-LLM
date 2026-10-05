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

#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace trt_edgellm
{
namespace retrieval_draft
{

struct RetrievalDraftConfig
{
    int32_t minSupport{2};       //!< Fewest corpus sequences through the matched node for a proposal
    double minShare{0.5};        //!< Smallest share of those sequences the majority next token must have
    int32_t minMatch{2};         //!< Shortest suffix of the output looked up in the suffix index
    int32_t maxMatch{4};         //!< Longest suffix of the output looked up in the suffix index
    int32_t maxContinuation{32}; //!< Suffix-index depth beyond maxMatch: the proposal length it guarantees
};

//! Proposes continuations of a partly decoded output from a corpus of generated sequences (the tokens after
//! <|cot_start|> of chain-of-causation outputs, see build_alpamayo1_5_coc_corpus.py), REST/SuffixDecoding style.
//!
//! The corpus is indexed twice: a prefix trie of whole sequences, and a suffix index (a trie of every suffix,
//! maxMatch + maxContinuation tokens deep). Every node counts the sequences (or occurrences) through it and
//! keeps its most frequent child. propose() matches the whole output in the prefix trie; when that path is
//! missing or has fewer than minSupport sequences, it matches the longest output suffix of maxMatch down to
//! minMatch tokens in the suffix index instead. The matched node proposes only when it has minSupport
//! sequences and its majority child holds at least minShare of them. The proposal then follows the majority
//! child to the end of the path whatever its share further on, since a verify round costs the same however many
//! of its slots hold proposed tokens. Proposals are only verified, so a wrong one costs acceptance, not output.
class RetrievalDraft
{
public:
    explicit RetrievalDraft(RetrievalDraftConfig const& config = {});

    //! Adds the sequences of a corpus file written by build_alpamayo1_5_coc_corpus.py. Returns false, adding
    //! nothing, when the file is missing or malformed.
    bool loadCorpus(std::string const& path);

    //! Adds one generated sequence.
    void add(int32_t const* tokens, int32_t length);

    //! Writes at most `capacity` tokens continuing `generated` to `out` and returns how many; 0 when no
    //! confident continuation exists.
    int32_t propose(int32_t const* generated, int32_t generatedLen, int32_t* out, int32_t capacity) const;

    int64_t numSequences() const noexcept
    {
        return mNumSequences;
    }
    int64_t numTokens() const noexcept
    {
        return mNumTokens;
    }
    //! Nodes in the prefix trie and the suffix index together.
    int64_t numNodes() const noexcept;

private:
    class Trie
    {
    public:
        Trie();
        //! Inserts the first min(length, depth) tokens of `tokens`.
        void insert(int32_t const* tokens, int32_t length, int32_t depth);
        //! Node reached by following `tokens` from the root, or -1.
        int32_t find(int32_t const* tokens, int32_t length) const;
        bool confident(int32_t node, RetrievalDraftConfig const& config) const;
        int32_t majorityPath(int32_t node, int32_t* out, int32_t capacity) const;
        int32_t count(int32_t node) const
        {
            return mNodes[node].count;
        }
        int64_t size() const noexcept
        {
            return static_cast<int64_t>(mNodes.size());
        }

    private:
        struct Node
        {
            int32_t token;
            int32_t count; //!< Sequences inserted through this node
            int32_t best;  //!< Child with the highest count (the first to reach it), -1 for a leaf
        };
        static uint64_t edgeKey(int32_t node, int32_t token) noexcept
        {
            return (static_cast<uint64_t>(static_cast<uint32_t>(node)) << 32U) | static_cast<uint32_t>(token);
        }
        std::vector<Node> mNodes;
        std::unordered_map<uint64_t, int32_t> mEdges; //!< edgeKey(parent, token) -> child
    };

    RetrievalDraftConfig mConfig;
    Trie mPrefix;
    Trie mSuffix;
    int64_t mNumSequences{0};
    int64_t mNumTokens{0};
};

} // namespace retrieval_draft
} // namespace trt_edgellm
