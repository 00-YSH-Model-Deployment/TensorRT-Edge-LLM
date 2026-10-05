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

#include "runtime/llmRuntimeUtils.h"
#include "tokenizer/tokenEncoder.h"
#include <string>
#include <vector>

namespace trt_edgellm
{
namespace rt
{
namespace action_utils
{

//! Encode past trajectory points (x,y,z) to binned trajectory token IDs for Alpamayo-style input.
std::vector<tokenizer::Rank> trajectoryToTokenIds(std::vector<PastTrajectoryPoint> const& trajectoryHistory,
    tokenizer::Rank numTrajTokens, tokenizer::Rank trajTokenStart);

//! Flow time grid 0 = t_0 < ... < t_n = 1 for cachedSteps.size() uniform Euler steps, where a flagged step reuses the
//! velocity of the last computed step (FlashDrive's action caching). The velocity depends only on (x, t0), so a run of
//! reused steps is the preceding computed step stretched over them: 8 steps with {3, 4, 5, 6} cached give
//! {0, 1/8, 2/8, 7/8, 1}. Exact in real arithmetic, so it matches step-by-step velocity reuse up to the FP16 rounding
//! of the action engine's update.
//! \throws std::invalid_argument if cachedSteps is empty or flags step 0
std::vector<float> foldCachedEulerSteps(std::vector<bool> const& cachedSteps);

//! Parse a comma-separated list of flow time points, e.g. "0,0.125,0.875,1". The list must start at exactly 0, end at
//! exactly 1 and increase strictly after rounding to FP16, the precision in which the action engine reads time.
//! \throws std::invalid_argument naming the offending entry if the list is malformed
std::vector<float> parseFlowTimeGrid(std::string const& list);

} // namespace action_utils
} // namespace rt
} // namespace trt_edgellm
