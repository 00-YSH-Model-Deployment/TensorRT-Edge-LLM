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

#include "action/actionUtils.h"

#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

using namespace trt_edgellm::rt;

namespace
{

std::vector<bool> cachedSteps(int32_t numSteps, std::vector<int32_t> const& cached)
{
    std::vector<bool> flags(static_cast<size_t>(numSteps), false);
    for (int32_t const step : cached)
    {
        flags[static_cast<size_t>(step)] = true;
    }
    return flags;
}

// Toy velocity field that depends on both the state and the time.
double velocity(double x, double t)
{
    return std::sin(3.0 * x) - 2.0 * t * x + 0.5;
}

TEST(ActionTimeGridTest, FlashDriveCacheScheduleIsAFourCallGrid)
{
    EXPECT_EQ(action_utils::foldCachedEulerSteps(cachedSteps(8, {3, 4, 5, 6})),
        (std::vector<float>{0.0F, 0.125F, 0.25F, 0.875F, 1.0F}));
}

TEST(ActionTimeGridTest, UncachedStepsKeepTheUniformGrid)
{
    std::vector<float> const grid = action_utils::foldCachedEulerSteps(cachedSteps(10, {}));
    ASSERT_EQ(grid.size(), 11U);
    for (int32_t i = 0; i <= 10; ++i)
    {
        EXPECT_EQ(grid[static_cast<size_t>(i)], static_cast<float>(i) / 10.0F);
    }
}

TEST(ActionTimeGridTest, TrailingCachedStepsStretchTheLastCallToOne)
{
    EXPECT_EQ(action_utils::foldCachedEulerSteps(cachedSteps(4, {2, 3})), (std::vector<float>{0.0F, 0.25F, 1.0F}));
}

TEST(ActionTimeGridTest, FirstStepCannotBeCached)
{
    EXPECT_THROW(action_utils::foldCachedEulerSteps(cachedSteps(4, {0})), std::invalid_argument);
    EXPECT_THROW(action_utils::foldCachedEulerSteps({}), std::invalid_argument);
}

TEST(ActionTimeGridTest, FoldedGridMatchesStepwiseVelocityReuse)
{
    std::mt19937 generator(7U);
    for (int32_t trial = 0; trial < 200; ++trial)
    {
        int32_t const numSteps = 2 + static_cast<int32_t>(generator() % 12U);
        std::vector<bool> cached(static_cast<size_t>(numSteps), false);
        for (int32_t i = 1; i < numSteps; ++i)
        {
            cached[static_cast<size_t>(i)] = (generator() % 2U) == 0U;
        }
        double const x0 = static_cast<double>(generator() % 1000U) / 250.0 - 2.0;

        double stepwise = x0;
        double reused = 0.0;
        for (int32_t i = 0; i < numSteps; ++i)
        {
            if (!cached[static_cast<size_t>(i)])
            {
                reused = velocity(stepwise, static_cast<double>(i) / numSteps);
            }
            stepwise += reused / numSteps;
        }

        std::vector<float> const grid = action_utils::foldCachedEulerSteps(cached);
        double folded = x0;
        for (size_t k = 0; k + 1 < grid.size(); ++k)
        {
            folded += (static_cast<double>(grid[k + 1]) - grid[k]) * velocity(folded, grid[k]);
        }
        EXPECT_NEAR(folded, stepwise, 1e-6) << "trial " << trial;
    }
}

TEST(ActionTimeGridTest, ParsesAValidGrid)
{
    EXPECT_EQ(action_utils::parseFlowTimeGrid("0,0.125,0.875,1"), (std::vector<float>{0.0F, 0.125F, 0.875F, 1.0F}));
    EXPECT_EQ(action_utils::parseFlowTimeGrid(" 0 , 0.5,1 "), (std::vector<float>{0.0F, 0.5F, 1.0F}));
    EXPECT_EQ(action_utils::parseFlowTimeGrid("0,1"), (std::vector<float>{0.0F, 1.0F}));
}

TEST(ActionTimeGridTest, RejectsMalformedGrids)
{
    for (std::string const list : {"", "0", "1", "0,", "0,,1", "0,1,", " ,0,1", "0.1,1", "0,0.9", "0,1,1",
             "0,0.5,0.5,1", "0,0.6,0.5,1", "0,abc,1", "0,0.5x,1", "0,0.5 0.6,1", "0,nan,1", "0,inf,1", "0,1e40,1",
             "0,1e-40,1", "0,0.50001,0.50002,1", "1,0"})
    {
        EXPECT_THROW(action_utils::parseFlowTimeGrid(list), std::invalid_argument) << "'" << list << "'";
    }
}

} // namespace
