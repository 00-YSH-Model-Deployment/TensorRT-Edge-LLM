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
#include <cerrno>
#include <cmath>
#include <cstdlib>
#include <cuda_fp16.h>
#include <stdexcept>
#include <tuple>

namespace trt_edgellm
{
namespace rt
{
namespace action_utils
{

std::vector<tokenizer::Rank> trajectoryToTokenIds(std::vector<PastTrajectoryPoint> const& trajectoryHistory,
    tokenizer::Rank numTrajTokens, tokenizer::Rank trajTokenStart)
{
    std::vector<tokenizer::Rank> tokenIds;
    std::tuple<float, float, float> point0, point;
    auto const ego_xyz_max = std::make_tuple(4.0f, 4.0f, 10.0f);
    auto const ego_xyz_min = std::make_tuple(-4.0f, -4.0f, -10.0f);

    auto clamp = [numTrajTokens](
                     int value) { return value < 0 ? 0 : (value >= numTrajTokens - 1 ? numTrajTokens - 1 : value); };

    auto tokenizePoint = [&](std::tuple<float, float, float> const& pt) {
        int const bx = clamp(static_cast<int>(std::round((std::get<0>(pt) - std::get<0>(ego_xyz_min))
                           / (std::get<0>(ego_xyz_max) - std::get<0>(ego_xyz_min)) * (numTrajTokens - 1))))
            + trajTokenStart;
        int const by = clamp(static_cast<int>(std::round((std::get<1>(pt) - std::get<1>(ego_xyz_min))
                           / (std::get<1>(ego_xyz_max) - std::get<1>(ego_xyz_min)) * (numTrajTokens - 1))))
            + trajTokenStart;
        int const bz = clamp(static_cast<int>(std::round((std::get<2>(pt) - std::get<2>(ego_xyz_min))
                           / (std::get<2>(ego_xyz_max) - std::get<2>(ego_xyz_min)) * (numTrajTokens - 1))))
            + trajTokenStart;
        tokenIds.push_back(static_cast<tokenizer::Rank>(bx));
        tokenIds.push_back(static_cast<tokenizer::Rank>(by));
        tokenIds.push_back(static_cast<tokenizer::Rank>(bz));
    };

    for (size_t i = 0; i < trajectoryHistory.size(); ++i)
    {
        auto const& item = trajectoryHistory[i];
        if (i == 0)
        {
            point0 = item;
            point = item;
        }
        else
        {
            point = std::make_tuple(std::get<0>(item) - std::get<0>(point0), std::get<1>(item) - std::get<1>(point0),
                std::get<2>(item) - std::get<2>(point0));
            point0 = item;
        }
        tokenizePoint(point);
    }

    return tokenIds;
}

std::vector<float> foldCachedEulerSteps(std::vector<bool> const& cachedSteps)
{
    if (cachedSteps.empty() || cachedSteps.front())
    {
        throw std::invalid_argument("the first Euler step has no cached velocity to reuse");
    }
    int32_t const numSteps = static_cast<int32_t>(cachedSteps.size());
    std::vector<float> grid;
    for (int32_t i = 0; i < numSteps; ++i)
    {
        if (!cachedSteps[static_cast<size_t>(i)])
        {
            grid.push_back(static_cast<float>(i) / static_cast<float>(numSteps));
        }
    }
    grid.push_back(1.0F);
    return grid;
}

std::vector<float> parseFlowTimeGrid(std::string const& list)
{
    std::vector<std::string> items;
    std::vector<float> grid;
    size_t pos = 0;
    while (true)
    {
        size_t const comma = list.find(',', pos);
        std::string item = list.substr(pos, comma == std::string::npos ? std::string::npos : comma - pos);
        size_t const first = item.find_first_not_of(" \t");
        item = first == std::string::npos ? "" : item.substr(first, item.find_last_not_of(" \t") - first + 1);
        if (item.empty())
        {
            throw std::invalid_argument("time point " + std::to_string(items.size()) + " is empty");
        }
        char* parsedEnd = nullptr;
        errno = 0;
        float const value = std::strtof(item.c_str(), &parsedEnd);
        if (parsedEnd != item.c_str() + item.size() || errno == ERANGE || !std::isfinite(value))
        {
            throw std::invalid_argument("time point '" + item + "' is not a finite number");
        }
        items.push_back(item);
        grid.push_back(value);
        if (comma == std::string::npos)
        {
            break;
        }
        pos = comma + 1;
    }

    if (grid.size() < 2U || grid.front() != 0.0F || grid.back() != 1.0F)
    {
        throw std::invalid_argument("the time grid must start at 0 and end at 1");
    }
    for (size_t k = 1; k < grid.size(); ++k)
    {
        if (__half2float(__float2half_rn(grid[k])) <= __half2float(__float2half_rn(grid[k - 1])))
        {
            throw std::invalid_argument(
                "time point '" + items[k] + "' does not exceed '" + items[k - 1] + "' at FP16 precision");
        }
    }
    return grid;
}

} // namespace action_utils
} // namespace rt
} // namespace trt_edgellm
