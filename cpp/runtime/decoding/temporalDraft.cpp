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

#include <algorithm>
#include <cstdlib>

namespace trt_edgellm
{
namespace temporal_draft
{

namespace
{
int32_t copyFrom(int32_t const* previous, int32_t previousLen, int32_t from, int32_t* out, int32_t capacity)
{
    int32_t const count = std::max(0, std::min(capacity, previousLen - from));
    std::copy(previous + from, previous + from + count, out);
    return count;
}
} // namespace

int32_t propose(int32_t const* previous, int32_t previousLen, int32_t const* generated, int32_t generatedLen,
    int32_t minMatch, int32_t* out, int32_t capacity)
{
    if (previous == nullptr || previousLen <= 0 || capacity <= 0 || generatedLen < 0)
    {
        return 0;
    }
    if (generatedLen < previousLen && std::equal(generated, generated + generatedLen, previous))
    {
        return copyFrom(previous, previousLen, generatedLen, out, capacity);
    }
    int32_t constexpr kMaxMatch = 4;
    for (int32_t n = std::min(kMaxMatch, generatedLen); n >= std::max(1, minMatch); --n)
    {
        int32_t const* suffix = generated + generatedLen - n;
        int32_t best = -1;
        for (int32_t end = n; end < previousLen; ++end)
        {
            if (std::equal(suffix, suffix + n, previous + end - n)
                && (best < 0 || std::abs(end - generatedLen) < std::abs(best - generatedLen)))
            {
                best = end;
            }
        }
        if (best >= 0)
        {
            return copyFrom(previous, previousLen, best, out, capacity);
        }
    }
    return 0;
}

} // namespace temporal_draft
} // namespace trt_edgellm
