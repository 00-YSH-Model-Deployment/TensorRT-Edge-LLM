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

#include <gtest/gtest.h>

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <vector>

using namespace trt_edgellm;
using retrieval_draft::RetrievalDraft;
using retrieval_draft::RetrievalDraftConfig;

namespace
{

using Sequence = std::vector<int32_t>;

RetrievalDraft makeDraft(std::vector<Sequence> const& corpus, RetrievalDraftConfig const& config = {})
{
    RetrievalDraft draft(config);
    for (auto const& sequence : corpus)
    {
        draft.add(sequence.data(), static_cast<int32_t>(sequence.size()));
    }
    return draft;
}

Sequence propose(RetrievalDraft const& draft, Sequence const& generated, int32_t capacity)
{
    Sequence out(static_cast<size_t>(capacity), -1);
    int32_t const n = draft.propose(generated.data(), static_cast<int32_t>(generated.size()), out.data(), capacity);
    out.resize(static_cast<size_t>(n));
    return out;
}

// 1 = "Keep", 2 = "distance", 3 = "to", ...; 90 = <|cot_end|>, 91 = <|traj_future_start|>.
Sequence const kLead{1, 2, 3, 4, 5, 90, 91};
Sequence const kSlowing{1, 2, 3, 4, 6, 7, 90, 91};
Sequence const kLane{1, 8, 9, 90, 91};

TEST(RetrievalDraftTest, ProposesTheMajorityContinuationOfThePrefix)
{
    RetrievalDraft const draft = makeDraft({kLead, kLead, kLead, kSlowing, kSlowing, kLane});
    EXPECT_EQ(propose(draft, {1}, 8), (Sequence{2, 3, 4, 5, 90, 91}));
    EXPECT_EQ(propose(draft, {1, 2, 3}, 2), (Sequence{4, 5}));
    // Past the branch point the output's own branch is continued.
    EXPECT_EQ(propose(draft, {1, 2, 3, 4, 6}, 8), (Sequence{7, 90, 91}));
    EXPECT_TRUE(propose(draft, kLead, 8).empty());
}

TEST(RetrievalDraftTest, AmbiguousOrRareNodesDoNotPropose)
{
    // A 2:2 split meets a 0.5 share; a tie goes to the continuation that reached the count first.
    RetrievalDraft const tied = makeDraft({kLead, kLead, kSlowing, kSlowing});
    EXPECT_EQ(propose(tied, {1, 2, 3, 4}, 8), (Sequence{5, 90, 91}));

    // "3 4" is followed by 5 in 5 of 7 suffixes, but the whole output has enough support and is split 2:2, so
    // the suffix index is not consulted.
    RetrievalDraftConfig strict;
    strict.minShare = 0.6;
    Sequence const other{20, 3, 4, 5, 90, 91};
    RetrievalDraft const split = makeDraft({kLead, kLead, kSlowing, kSlowing, other, other, other}, strict);
    EXPECT_TRUE(propose(split, {1, 2, 3, 4}, 8).empty());
    EXPECT_EQ(propose(split, {30, 3, 4}, 8), (Sequence{5, 90, 91}));

    RetrievalDraftConfig rare;
    rare.minSupport = 3;
    RetrievalDraft const single = makeDraft({kLead, kLane}, rare);
    EXPECT_TRUE(propose(single, {1, 2}, 8).empty());
}

TEST(RetrievalDraftTest, DivergedOutputFallsBackToTheSuffixIndex)
{
    RetrievalDraft const draft = makeDraft({kLead, kLead, kSlowing});
    // "Decelerate" (50) for "Keep": no prefix path, but "2 3 4" occurs three times, twice followed by 5.
    EXPECT_EQ(propose(draft, {50, 2, 3, 4}, 8), (Sequence{5, 90, 91}));
    EXPECT_TRUE(propose(draft, {50, 51, 4}, 8).empty());
    RetrievalDraftConfig loose;
    loose.minMatch = 1;
    RetrievalDraft const looseDraft = makeDraft({kLead, kLead, kSlowing}, loose);
    EXPECT_EQ(propose(looseDraft, {50, 51, 4}, 8), (Sequence{5, 90, 91}));
}

TEST(RetrievalDraftTest, LoadsTheCorpusFileFormat)
{
    std::filesystem::path const path = std::filesystem::temp_directory_path() / "retrievalDraftTestCorpus.bin";
    {
        std::vector<int32_t> const offsets{
            0, static_cast<int32_t>(kLead.size()), static_cast<int32_t>(kLead.size() + kLane.size())};
        std::vector<int32_t> const header{/*version=*/1, /*sequences=*/2, /*tokens=*/offsets.back()};
        std::ofstream file(path, std::ios::binary);
        file.write("COCR", 4);
        file.write(reinterpret_cast<char const*>(header.data()), header.size() * sizeof(int32_t));
        file.write(reinterpret_cast<char const*>(offsets.data()), offsets.size() * sizeof(int32_t));
        file.write(reinterpret_cast<char const*>(kLead.data()), kLead.size() * sizeof(int32_t));
        file.write(reinterpret_cast<char const*>(kLane.data()), kLane.size() * sizeof(int32_t));
    }
    RetrievalDraftConfig config;
    config.minSupport = 1;
    RetrievalDraft draft(config);
    ASSERT_TRUE(draft.loadCorpus(path.string()));
    EXPECT_EQ(draft.numSequences(), 2);
    EXPECT_EQ(draft.numTokens(), static_cast<int64_t>(kLead.size() + kLane.size()));
    EXPECT_EQ(propose(draft, {1, 8}, 8), (Sequence{9, 90, 91}));

    std::filesystem::resize_file(path, std::filesystem::file_size(path) - sizeof(int32_t));
    RetrievalDraft truncated(config);
    EXPECT_FALSE(truncated.loadCorpus(path.string()));
    EXPECT_EQ(truncated.numSequences(), 0);
    EXPECT_FALSE(truncated.loadCorpus(path.string() + ".missing"));
    std::filesystem::remove(path);
}

TEST(RetrievalDraftTest, NothingToProposeWithoutACorpus)
{
    RetrievalDraft const empty;
    EXPECT_TRUE(propose(empty, {1, 2}, 8).empty());
    RetrievalDraft const draft = makeDraft({kLead, kLead});
    EXPECT_TRUE(propose(draft, {1}, 0).empty());
}

} // namespace
