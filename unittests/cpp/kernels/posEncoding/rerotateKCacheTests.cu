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

#include <gtest/gtest.h>

#include "kernels/posEncoding/rerotateKCache.h"

#include <cmath>
#include <cuda_fp16.h>
#include <random>
#include <vector>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

using namespace trt_edgellm;

namespace
{

constexpr int32_t kRotaryDim = 128;
constexpr int32_t kHeads = 2;
constexpr int32_t kRows = 40;

//! Per-row angles from arbitrary (non-arithmetic) positions, as M-RoPE rows are: cos at [i], sin at [i + 64].
std::vector<float> cosSinTable(std::vector<float> const& positions)
{
    std::vector<float> table(positions.size() * kRotaryDim);
    for (size_t r = 0; r < positions.size(); ++r)
    {
        for (int32_t i = 0; i < kRotaryDim / 2; ++i)
        {
            float const angle = positions[r] / std::pow(5.0e6F, 2.0F * i / kRotaryDim);
            table[r * kRotaryDim + i] = std::cos(angle);
            table[r * kRotaryDim + i + kRotaryDim / 2] = std::sin(angle);
        }
    }
    return table;
}

//! Rotate-half RoPE of one head vector at table row `row`.
std::vector<float> rotate(std::vector<float> const& x, std::vector<float> const& table, int32_t row)
{
    std::vector<float> y(kRotaryDim);
    for (int32_t i = 0; i < kRotaryDim / 2; ++i)
    {
        float const c = table[row * kRotaryDim + i];
        float const s = table[row * kRotaryDim + i + kRotaryDim / 2];
        y[i] = x[i] * c - x[i + kRotaryDim / 2] * s;
        y[i + kRotaryDim / 2] = x[i + kRotaryDim / 2] * c + x[i] * s;
    }
    return y;
}

TEST(RerotateKCacheTest, MovedKeysMatchKeysRotatedAtTheirNewRows)
{
    std::mt19937 rng(7);
    std::uniform_real_distribution<float> unit(-1.0F, 1.0F);
    std::vector<float> positions(kRows);
    for (int32_t r = 0; r < kRows; ++r)
    {
        positions[r] = 100.0F + 37.0F * r + (r % 3) * 5.0F;
    }
    std::vector<float> const table = cosSinTable(positions);

    // Pre-RoPE keys; row r of the cache holds key r rotated at row r. Then rows [24, 32) move to [4, 12).
    std::vector<std::vector<float>> raw(kRows * kHeads, std::vector<float>(kRotaryDim));
    for (auto& v : raw)
    {
        for (auto& e : v)
        {
            e = unit(rng);
        }
    }
    thrust::host_vector<half> cache(kRows * kHeads * kRotaryDim);
    auto put = [&](int32_t row, int32_t head, std::vector<float> const& v) {
        for (int32_t i = 0; i < kRotaryDim; ++i)
        {
            cache[(row * kHeads + head) * kRotaryDim + i] = __float2half(v[i]);
        }
    };
    for (int32_t r = 0; r < kRows; ++r)
    {
        for (int32_t h = 0; h < kHeads; ++h)
        {
            put(r, h, rotate(raw[r * kHeads + h], table, r));
        }
    }
    int32_t const src = 24;
    int32_t const dst = 4;
    int32_t const len = 8;
    for (int32_t t = 0; t < len; ++t)
    {
        for (int32_t h = 0; h < kHeads; ++h)
        {
            put(dst + t, h, rotate(raw[(src + t) * kHeads + h], table, src + t));
        }
    }

    thrust::device_vector<half> dCache = cache;
    thrust::device_vector<float> dTable(table.begin(), table.end());
    kernel::KRerotateSegments segs;
    segs.dst[0] = dst;
    segs.src[0] = src;
    segs.len[0] = len;
    segs.count = 1;
    kernel::launchRerotateKCache(thrust::raw_pointer_cast(dCache.data()), thrust::raw_pointer_cast(dTable.data()),
        kRotaryDim, kHeads, kRotaryDim, segs, len, nullptr);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    thrust::host_vector<half> out = dCache;

    for (int32_t t = 0; t < len; ++t)
    {
        for (int32_t h = 0; h < kHeads; ++h)
        {
            std::vector<float> const expected = rotate(raw[(src + t) * kHeads + h], table, dst + t);
            for (int32_t i = 0; i < kRotaryDim; ++i)
            {
                EXPECT_NEAR(__half2float(out[((dst + t) * kHeads + h) * kRotaryDim + i]), expected[i], 4e-3F)
                    << "row " << dst + t << " head " << h << " dim " << i;
            }
        }
    }
    // Rows outside the segment are untouched.
    EXPECT_EQ(__half2float(out[(0 * kHeads) * kRotaryDim]), __half2float(cache[0]));
    EXPECT_EQ(__half2float(out[(20 * kHeads + 1) * kRotaryDim + 5]), __half2float(cache[(20 * kHeads + 1) * kRotaryDim + 5]));
}

} // namespace
