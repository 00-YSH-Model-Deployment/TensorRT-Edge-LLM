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

namespace trt_edgellm
{
namespace temporal_draft
{

//! Proposes the continuation of `generated` (the tokens decoded so far in this window) from `previous` (the
//! tokens the previous window decoded). Consecutive camera windows of a driving clip mostly repeat their
//! reasoning, so the previous output is a draft that needs no model.
//!
//! - If `generated` is a prefix of `previous`, the rest of `previous` is proposed.
//! - Otherwise the longest suffix of `generated` (at most 4 tokens, at least `minMatch`) found in `previous` is
//!   located, preferring the occurrence that ends nearest index `generated` (the aligned position), and the
//!   tokens after it are proposed.
//!
//! Writes at most `capacity` tokens to `out` and returns how many; 0 when nothing matches.
int32_t propose(int32_t const* previous, int32_t previousLen, int32_t const* generated, int32_t generatedLen,
    int32_t minMatch, int32_t* out, int32_t capacity);

} // namespace temporal_draft
} // namespace trt_edgellm
