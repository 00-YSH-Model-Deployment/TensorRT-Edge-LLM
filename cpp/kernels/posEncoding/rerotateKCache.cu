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

#include "common/checkMacros.h"
#include "kernels/posEncoding/rerotateKCache.h"

#include <algorithm>

namespace trt_edgellm
{
namespace kernel
{

namespace
{

__global__ void rerotateKCacheKernel(
    half* kRows, float const* cosSin, int32_t rotaryDim, int32_t numKVHeads, int32_t headDim, KRerotateSegments segs)
{
    int32_t const seg = blockIdx.y;
    int32_t const t = blockIdx.x;
    if (t >= segs.len[seg])
    {
        return;
    }
    int32_t const halfDim = rotaryDim / 2;
    int64_t const dstRow = segs.dst[seg] + t;
    int64_t const srcRow = segs.src[seg] + t;
    float const* cd = cosSin + dstRow * rotaryDim;
    float const* cs = cosSin + srcRow * rotaryDim;
    half* k = kRows + (dstRow * numKVHeads + blockIdx.z) * headDim;
    // Each thread owns whole (i, i + halfDim) pairs, so the update is in place without races.
    for (int32_t i = threadIdx.x; i < halfDim; i += blockDim.x)
    {
        float const c = cd[i] * cs[i] + cd[i + halfDim] * cs[i + halfDim];
        float const s = cd[i + halfDim] * cs[i] - cd[i] * cs[i + halfDim];
        float const x1 = __half2float(k[i]);
        float const x2 = __half2float(k[i + halfDim]);
        k[i] = __float2half(x1 * c - x2 * s);
        k[i + halfDim] = __float2half(x2 * c + x1 * s);
    }
}

} // namespace

void launchRerotateKCache(half* kRows, float const* cosSin, int32_t rotaryDim, int32_t numKVHeads,
    int32_t headDim, KRerotateSegments const& segments, int32_t maxLen, cudaStream_t stream)
{
    if (segments.count <= 0 || maxLen <= 0)
    {
        return;
    }
    check::check(segments.count <= KRerotateSegments::kMaxSegments, "too many K re-rotation segments");
    check::check(rotaryDim % 2 == 0 && rotaryDim <= headDim, "rotaryDim must be even and at most headDim");
    dim3 const grid(static_cast<uint32_t>(maxLen), static_cast<uint32_t>(segments.count),
        static_cast<uint32_t>(numKVHeads));
    dim3 const block(static_cast<uint32_t>(std::min(rotaryDim / 2, 64)));
    rerotateKCacheKernel<<<grid, block, 0, stream>>>(kRows, cosSin, rotaryDim, numKVHeads, headDim, segments);
}

} // namespace kernel
} // namespace trt_edgellm
