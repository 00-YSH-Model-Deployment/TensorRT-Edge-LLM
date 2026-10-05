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
#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace trt_edgellm
{
namespace kernel
{

//! Row ranges of a KV cache whose keys were moved: rows [dst, dst + len) now hold keys written at rows
//! [src, src + len). Passed by value, so no host-to-device copy.
struct KRerotateSegments
{
    static constexpr int32_t kMaxSegments{32};
    int32_t dst[kMaxSegments];
    int32_t src[kMaxSegments];
    int32_t len[kMaxSegments];
    int32_t count{0};
};

//! Re-rotates moved keys in place so each carries the rotary position of its new row: k <- R(dst) R(src)^T k.
//! Both rotations come from the request's cos/sin table (row = sequence index; cos at [i], sin at
//! [i + rotaryDim / 2]), so the result is what a prefill at the new rows would have written, whatever the
//! position ids (M-RoPE included). Rotate-half pairing (i, i + rotaryDim / 2); dimensions past rotaryDim are
//! untouched. kRows: one layer's FP16 K rows, [rows, numKVHeads, headDim], contiguous.
void launchRerotateKCache(half* kRows, float const* cosSin, int32_t rotaryDim, int32_t numKVHeads,
    int32_t headDim, KRerotateSegments const& segments, int32_t maxLen, cudaStream_t stream);

} // namespace kernel
} // namespace trt_edgellm
