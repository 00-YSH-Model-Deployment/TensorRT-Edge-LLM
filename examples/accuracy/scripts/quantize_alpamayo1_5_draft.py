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
"""Quantize the DFlash draft's decoder layers to weight-only NVFP4 (W4A16).

The draft only proposes; the base verifies, so a coarser draft can change how
often proposals are accepted but not the output. Weight-only NVFP4 needs no
activation calibration: every projection weight is packed exactly as ModelOpt
exports ``W4A16_NVFP4`` (E2M1 codes, an E4M3 scale per 16 inputs and an FP32
per-tensor scale), so this runs on the CPU without Alpamayo or driving clips.

Only the q/k/v/o and gate/up/down projections of ``layers.*`` are quantized.
``fc`` stays dense, as the exporter requires: it projects the base's hidden
states, whose ~1e4 activations need the FP32 projection or
``--dflash-fc-fp16`` scaling. The draft takes ``lm_head`` from the base, and it
stays FP16 (the reduced draft vocabulary slices it). ``tensorrt-edgellm-export
--dflash-draft --dflash-draft-dir OUT`` then emits the projections as
``Nvfp4A16GemmPlugin`` (Marlin W4A16) GEMMs.

The output directory links the snapshot's other files (``config.json``,
``mask_embedding.pt``) and holds one ``model.safetensors`` and
``hf_quant_config.json``.
"""
import argparse
import json
import re
from pathlib import Path

import modelopt
import torch
from modelopt.torch.quantization.qtensor import NVFP4QTensor
from safetensors import safe_open
from safetensors.torch import save_file

GROUP_SIZE = 16
PROJECTION = re.compile(
    r"layers\.\d+\.(self_attn\.[qkvo]_proj|mlp\.(gate|up|down)_proj)\.weight")
E2M1 = torch.tensor(
    [0, 0.5, 1, 1.5, 2, 3, 4, 6, 0, -0.5, -1, -1.5, -2, -3, -4, -6])


def quantize(weight: torch.Tensor) -> dict[str, torch.Tensor]:
    """ModelOpt's W4A16_NVFP4 export tensors for one ``[out, in]`` weight."""
    w = weight.float()
    scale_2 = NVFP4QTensor.get_weights_scaling_factor_2(w)
    scale, _ = NVFP4QTensor.get_weights_scaling_factor(w, GROUP_SIZE, scale_2)
    packed = NVFP4QTensor.quantize(w, GROUP_SIZE, scale,
                                   scale_2)[0]._quantized_data
    return {"weight": packed, "weight_scale": scale, "weight_scale_2": scale_2}


def dequantize(q: dict[str, torch.Tensor]) -> torch.Tensor:
    # Low nibble holds the even input column, high nibble the odd one.
    packed = q["weight"]
    codes = torch.stack([packed & 0x0F, packed >> 4], dim=-1).flatten(-2)
    values = E2M1[codes.long()].unflatten(-1, (-1, GROUP_SIZE))
    scale = q["weight_scale"].float().unsqueeze(-1) * q["weight_scale_2"].float()
    return (values * scale).flatten(-2)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--draft-dir", type=Path, required=True,
                   help="DFlash draft snapshot, e.g. z-lab/Alpamayo-1.5-10B-DFlash.")
    p.add_argument("--out-dir", type=Path, required=True)
    args = p.parse_args()
    src, out_dir = args.draft_dir, args.out_dir
    assert src.resolve() != out_dir.resolve(), "--out-dir must differ from --draft-dir"

    tensors = {}
    for shard in sorted(src.glob("*.safetensors")):
        with safe_open(str(shard), framework="pt") as f:
            tensors.update({k: f.get_tensor(k) for k in f.keys()})
    names = sorted(k for k in tensors if PROJECTION.fullmatch(k))
    assert names, f"no decoder-layer projections in {src}"

    out, errors, dense_bytes, packed_bytes = {}, {}, 0, 0
    for key, tensor in tensors.items():
        if key not in names:
            out[key] = tensor
            continue
        module = key[:-len(".weight")]
        q = quantize(tensor)
        w = tensor.float()
        errors[module] = float((dequantize(q) - w).norm() / w.norm())
        dense_bytes += tensor.numel() * tensor.element_size()
        packed_bytes += sum(v.numel() * v.element_size() for v in q.values())
        out.update({f"{module}.{name}": v for name, v in q.items()})

    out_dir.mkdir(parents=True, exist_ok=True)
    for f in src.iterdir():
        if f.name.startswith(".") or ".safetensors" in f.name or f.name == "hf_quant_config.json":
            continue
        link = out_dir / f.name
        if not link.exists():
            link.symlink_to(f.resolve())
    save_file({k: v.contiguous() for k, v in out.items()}, str(out_dir / "model.safetensors"))
    (out_dir / "hf_quant_config.json").write_text(
        json.dumps(
            {
                "producer": {
                    "name": "modelopt",
                    "version": modelopt.__version__
                },
                "quantization": {
                    "quant_algo": "W4A16_NVFP4",
                    "kv_cache_quant_algo": None,
                    "group_size": GROUP_SIZE,
                    "exclude_modules": ["fc", "lm_head"],
                },
                "source": str(src.resolve()),
            },
            indent=2))
    summary = {
        "quantized_linears": len(names),
        "dense_mb": round(dense_bytes / 2**20, 1),
        "packed_mb": round(packed_bytes / 2**20, 1),
        "rel_error_mean": round(sum(errors.values()) / len(errors), 4),
        "rel_error_max": round(max(errors.values()), 4),
    }
    print(json.dumps(summary))
    print(f"wrote {out_dir}", flush=True)


if __name__ == "__main__":
    main()
