# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""CPU checks that the Qwen3-VL patch embed's Linear forms equal its Conv3d."""

import copy
import os
import sys

import pytest
import torch

_REPO_ROOT = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

try:
    from tensorrt_edgellm.models.qwen3_vl.modeling_qwen3_vl_visual import (
        PATCH_EMBED_ENV, Qwen3VLPatchEmbed, patch_embed_mode)
except ImportError as exc:  # pragma: no cover
    pytest.skip(f"tensorrt_edgellm not importable: {exc}",
                allow_module_level=True)

# Qwen3-VL ViT geometry (Alpamayo 1.5): 3 x 2 x 16 x 16 voxels -> 1152.
_C, _T, _P, _HIDDEN, _MERGE = 3, 2, 16, 1152, 2
# FP32 accumulation-order noise for |output| ~ 4; observed max ~1e-5.
_ATOL = 1e-4


def _random_patch_embed(seed: int,
                        dtype: torch.dtype = torch.float32
                        ) -> Qwen3VLPatchEmbed:
    torch.manual_seed(seed)
    embed = Qwen3VLPatchEmbed(_C, _HIDDEN, _P, _T).to(dtype).eval()
    torch.nn.init.normal_(embed.proj.weight, std=0.02)
    torch.nn.init.normal_(embed.proj.bias, std=0.02)
    return embed


def _converted(embed: Qwen3VLPatchEmbed,
               fold_temporal: bool) -> Qwen3VLPatchEmbed:
    converted = copy.deepcopy(embed)
    converted.to_linear(fold_temporal=fold_temporal)
    return converted


def _patchify(frames: torch.Tensor, temporal: int) -> torch.Tensor:
    """HF Qwen2VLImageProcessor rows (the runtime kernel's order) of [t, C, H, W]."""
    t, c, h, w = frames.shape
    x = frames.reshape(t // temporal, temporal, c, h // _P // _MERGE, _MERGE,
                       _P, w // _P // _MERGE, _MERGE, _P)
    x = x.permute(0, 3, 6, 4, 7, 2, 1, 5, 8)
    return x.reshape(-1, c * temporal * _P * _P)


@pytest.mark.parametrize("dtype,atol", [(torch.float32, _ATOL),
                                        (torch.float64, 1e-12)])
def test_linear_matches_conv3d(dtype, atol):
    conv = _random_patch_embed(0, dtype)
    linear = _converted(conv, fold_temporal=False)
    rows = torch.randn(1024, _C * _T * _P * _P, dtype=dtype)
    with torch.no_grad():
        ref, out = conv(rows), linear(rows)
    assert linear.input_dim == _C * _T * _P * _P
    torch.testing.assert_close(out, ref, rtol=0, atol=atol)


@pytest.mark.parametrize("dtype,atol", [(torch.float32, _ATOL),
                                        (torch.float64, 1e-12)])
def test_fold_matches_conv3d_on_repeated_frame(dtype, atol):
    conv = _random_patch_embed(1, dtype)
    folded = _converted(conv, fold_temporal=True)
    frame = torch.randn(1024, _C, 1, _P, _P, dtype=dtype)
    repeated = frame.expand(-1, -1, _T, -1, -1).reshape(1024, -1)
    with torch.no_grad():
        ref, out = conv(repeated), folded(frame.reshape(1024, -1))
    assert folded.input_dim == _C * _P * _P
    torch.testing.assert_close(out, ref, rtol=0, atol=atol)


def test_folded_runtime_rows_match_duplicated_frame_rows():
    # The runtime patchifies a still image once (temporal 1) for a folded
    # engine, and twice-replicated (temporal 2) otherwise.
    conv = _random_patch_embed(2)
    folded = _converted(conv, fold_temporal=True)
    image = torch.randn(1, _C, 4 * _P * _MERGE, 6 * _P * _MERGE)
    with torch.no_grad():
        ref = conv(_patchify(image.repeat(_T, 1, 1, 1), _T))
        out = folded(_patchify(image, 1))
    torch.testing.assert_close(out, ref, rtol=0, atol=_ATOL)


def test_to_linear_keeps_weights_and_rejects_second_call():
    conv = _random_patch_embed(3, torch.float16)
    linear = _converted(conv, fold_temporal=False)
    assert torch.equal(linear.proj.weight,
                       conv.proj.weight.reshape(_HIDDEN, -1))
    assert torch.equal(linear.proj.bias, conv.proj.bias)
    with pytest.raises(RuntimeError):
        linear.to_linear()


def test_patch_embed_mode(monkeypatch):
    monkeypatch.delenv(PATCH_EMBED_ENV, raising=False)
    assert patch_embed_mode() == "conv3d"
    monkeypatch.setenv(PATCH_EMBED_ENV, "Fold")
    assert patch_embed_mode() == "fold"
    monkeypatch.setenv(PATCH_EMBED_ENV, "gemm")
    with pytest.raises(ValueError, match=PATCH_EMBED_ENV):
        patch_embed_mode()
