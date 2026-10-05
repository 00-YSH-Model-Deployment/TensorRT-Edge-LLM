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

#include "runtime/decoding/temporalDraft.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <vector>

using namespace trt_edgellm;

namespace
{

std::vector<int32_t> propose(std::vector<int32_t> const& previous, std::vector<int32_t> const& generated,
    int32_t capacity, int32_t minMatch = 2)
{
    std::vector<int32_t> out(static_cast<size_t>(capacity), -1);
    int32_t const n = temporal_draft::propose(previous.data(), static_cast<int32_t>(previous.size()),
        generated.data(), static_cast<int32_t>(generated.size()), minMatch, out.data(), capacity);
    out.resize(static_cast<size_t>(n));
    return out;
}

TEST(TemporalDraftTest, PrefixProposesTheRestOfThePreviousOutput)
{
    std::vector<int32_t> const previous{10, 11, 12, 13, 14, 15};
    EXPECT_EQ(propose(previous, {10}, 8), (std::vector<int32_t>{11, 12, 13, 14, 15}));
    EXPECT_EQ(propose(previous, {10, 11, 12}, 2), (std::vector<int32_t>{13, 14}));
    EXPECT_TRUE(propose(previous, previous, 8).empty());
}

TEST(TemporalDraftTest, DivergedOutputRealignsOnASuffixMatch)
{
    // The window changed one word ("Decelerate" for "Keep"), then the reasoning continues as before.
    std::vector<int32_t> const previous{1, 2, 3, 4, 5, 6, 7};
    EXPECT_EQ(propose(previous, {9, 3, 4}, 8), (std::vector<int32_t>{5, 6, 7}));
    EXPECT_EQ(propose(previous, {9, 9, 4}, 8), (std::vector<int32_t>{}));
    EXPECT_EQ(propose(previous, {9, 9, 4}, 8, /*minMatch=*/1), (std::vector<int32_t>{5, 6, 7}));
}

TEST(TemporalDraftTest, PrefersTheOccurrenceNearestTheAlignedPosition)
{
    // "a b" occurs twice; with four tokens generated the occurrence ending at index 4 is the aligned one.
    std::vector<int32_t> const previous{1, 2, 30, 1, 2, 40, 50};
    EXPECT_EQ(propose(previous, {8, 8, 1, 2}, 8), (std::vector<int32_t>{40, 50}));
    EXPECT_EQ(propose(previous, {1, 2}, 8), (std::vector<int32_t>{30, 1, 2, 40, 50}));
}

TEST(TemporalDraftTest, NothingToProposeWithoutAPreviousOutput)
{
    EXPECT_TRUE(propose({}, {1, 2}, 8).empty());
    EXPECT_TRUE(propose({1, 2, 3}, {1}, 0).empty());
}

} // namespace
