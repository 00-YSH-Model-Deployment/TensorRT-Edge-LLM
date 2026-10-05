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
"""DFlash draft precision options: FP16 fc projector and W4A16 NVFP4 layers."""

import json
import os
import sys

import pytest

torch = pytest.importorskip("torch")
safetensors_torch = pytest.importorskip("safetensors.torch")

_REPO_ROOT = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from tensorrt_edgellm import config  # noqa: E402

_HIDDEN = 64
_NUM_TARGET_LAYERS = 5


def _write_draft(path, num_layers, quant_algo=None):
    cfg = {
        "architectures": ["DFlashDraftModel"],
        "model_type": "qwen3",
        "hidden_size": _HIDDEN,
        "intermediate_size": 128,
        "num_attention_heads": 1,
        "num_key_value_heads": 1,
        "head_dim": 64,
        "num_hidden_layers": num_layers,
        "layer_types": ["full_attention"] * num_layers,
        "vocab_size": 128,
        "rms_norm_eps": 1e-6,
        "rope_theta": 1e6,
        "tie_word_embeddings": False,
        "block_size": 8,
        "mask_token_id": 127,
        "target_layer_ids": [1, 2, 3, 4, 5],
    }
    (path / "config.json").write_text(json.dumps(cfg))
    tensors = {
        "fc.weight":
        torch.zeros(_HIDDEN, _NUM_TARGET_LAYERS * _HIDDEN,
                    dtype=torch.bfloat16)
    }
    if quant_algo is not None:
        proj = "layers.0.mlp.down_proj"
        tensors[f"{proj}.weight"] = torch.zeros(_HIDDEN, 64, dtype=torch.uint8)
        tensors[f"{proj}.weight_scale"] = torch.ones(
            _HIDDEN, 8, dtype=torch.float8_e4m3fn)
        tensors[f"{proj}.weight_scale_2"] = torch.ones((), dtype=torch.float32)
        (path / "hf_quant_config.json").write_text(
            json.dumps({
                "quantization": {
                    "quant_algo": quant_algo,
                    "group_size": 16,
                    "exclude_modules": ["fc", "lm_head"],
                }
            }))
    safetensors_torch.save_file(tensors, str(path / "model.safetensors"))
    return config.make_dflash_draft_config(str(path), lambda hd: hd**-0.5)


@pytest.mark.parametrize(("quant_algo", "expected"),
                         [("W4A16_NVFP4", config.QUANT_NVFP4_A16),
                          ("NVFP4", config.QUANT_NVFP4)],
                         ids=["weight-only", "w4a4"])
def test_dflash_draft_w4a16_nvfp4_uses_marlin_a16(tmp_path, quant_algo,
                                                  expected):
    cfg = _write_draft(tmp_path, 1, quant_algo)
    assert cfg.quant.quant_type == expected
    assert config.module_quant_type("layers.0.mlp.down_proj", cfg) == expected
    assert config.module_quant_type("fc", cfg) == config.QUANT_FP16
    assert config.module_quant_type("lm_head", cfg) == config.QUANT_FP16


def _run_fc(model, target_hidden):
    """Return the draft's h_delta (fc -> hidden_norm) for ``target_hidden``."""
    captured = {}
    hook = model.hidden_norm.register_forward_hook(
        lambda _m, _i, out: captured.setdefault("h", out))
    batch, length = target_hidden.shape[:2]

    def index(*shape):
        return torch.zeros(*shape, dtype=torch.int32)

    with torch.no_grad():
        model(torch.zeros(batch, 8, _HIDDEN, dtype=torch.float16),
              target_hidden, torch.zeros(1, 16, 64), index(batch),
              index(batch), index(batch, 2, 1),
              torch.full((batch, ), length, dtype=torch.int32),
              index(batch, 8, 1), index(batch, 8), [])
    hook.remove()
    return captured["h"].to(torch.float16).float()


def test_enable_fp16_fc_matches_fp32_projection_without_overflow(tmp_path):
    from tensorrt_edgellm.models.dflash.modeling_dflash_draft import \
        DFlashDraftModel

    cfg = _write_draft(tmp_path, 0)
    torch.manual_seed(0)
    weight = (torch.randn(_HIDDEN, _NUM_TARGET_LAYERS * _HIDDEN) *
              0.25).to(torch.float16)
    models = []
    for _ in range(3):
        model = DFlashDraftModel(cfg).eval()
        model.fc.weight.data = weight.clone()
        model.hidden_norm.weight.data.fill_(1)
        models.append(model)
    reference, scaled, unscaled = models

    scale_log2 = scaled.enable_fp16_fc()
    row_l1 = weight.double().abs().sum(dim=1).max().item()
    assert 2**(scale_log2 - 1) < row_l1 * (1 + 2**-6) <= 2**scale_log2
    assert unscaled.enable_fp16_fc(0) == 0
    with pytest.raises(RuntimeError):
        scaled.enable_fp16_fc()
    with pytest.raises(ValueError):
        DFlashDraftModel(cfg).enable_fp16_fc(15)

    # Typical activations plus one row aligned with an fc row at FP16 max:
    # the worst case the default exponent is chosen for.
    typical = torch.randn(1, 4, _NUM_TARGET_LAYERS * _HIDDEN) * 4
    worst = 65504.0 * torch.sign(weight[0].float())[None, None]
    target_hidden = torch.cat([typical, worst], dim=1).to(torch.float16)

    h_ref = _run_fc(reference, target_hidden)
    h_scaled = _run_fc(scaled, target_hidden)
    assert torch.isfinite(h_scaled).all()
    assert not torch.isfinite(_run_fc(unscaled, target_hidden)).all()
    torch.testing.assert_close(h_scaled, h_ref, rtol=2e-2, atol=2e-2)
