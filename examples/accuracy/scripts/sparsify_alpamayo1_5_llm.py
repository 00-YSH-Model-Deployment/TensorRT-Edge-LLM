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
"""2:4-sparsify Alpamayo 1.5's language model with ModelOpt, calibrated on driving clips.

Run like quantize_alpamayo1_5_llm.py (same Python, same calibration clips and
rollout loop). Semi-structured 2:4 sparsity keeps every matrix's shape -- two
of each four consecutive input weights become zero -- so the checkpoint loads
into the same architecture, and TensorRT can run the GEMMs on sparse tensor
cores when the engine is built with sparse weights enabled.

``sparsegpt`` updates the surviving weights from a Hessian of the layer
inputs; ``sparse_magnitude`` only drops the smallest. Only
``vlm.model.language_model`` linears change. The output directory links the
source snapshot and adds one bf16 shard with the sparse linears.
"""
import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("TORCH_DISABLE_NATIVE_JIT", "1")

import modelopt.torch.sparsity as mts  # noqa: E402
import physical_ai_av  # noqa: E402
import torch  # noqa: E402
from safetensors.torch import save_file  # noqa: E402
from torch import nn  # noqa: E402

from alpamayo1_5 import helper  # noqa: E402
from alpamayo1_5.models.alpamayo1_5 import Alpamayo1_5  # noqa: E402
from quantize_alpamayo1_5_llm import (DATASET_REVISION, LLM_PREFIX,  # noqa: E402
                                      make_forward_loop, pick_clips)

SPARSE_SHARD = "model-edgellm-sparse.safetensors"


def check_2to4(weight: torch.Tensor) -> bool:
    groups = weight.reshape(-1, 4)
    return bool(((groups == 0).sum(dim=1) >= 2).all())


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--model", type=Path, required=True, help="Alpamayo 1.5 snapshot directory.")
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--method", choices=["sparsegpt", "sparse_magnitude"], default="sparsegpt")
    p.add_argument("--clip-list", required=True)
    p.add_argument("--split", default="train")
    p.add_argument("--exclude", action="append", default=[])
    p.add_argument("--num-clips", type=int, default=32)
    p.add_argument("--t0-us", type=int, default=5_100_000)
    p.add_argument("--data-cache", default=None)
    p.add_argument("--temperature", type=float, default=0.6)
    p.add_argument("--top-p", type=float, default=0.98)
    p.add_argument("--max-generation-length", type=int, default=256)
    args = p.parse_args()

    clip_ids = pick_clips(args)
    print(f"{len(clip_ids)} calibration clips", flush=True)
    avdi = physical_ai_av.PhysicalAIAVDatasetInterface(cache_dir=args.data_cache,
                                                       revision=DATASET_REVISION)
    model = Alpamayo1_5.from_pretrained(str(args.model), dtype=torch.bfloat16,
                                        attn_implementation="sdpa").to("cuda").eval()
    processor = helper.get_processor(model.tokenizer)
    language_model = model.vlm.model.language_model

    loop = make_forward_loop(model, processor, avdi, clip_ids, args)
    config = {"forward_loop": loop} if args.method == "sparsegpt" else {}
    mts.sparsify(language_model, args.method, config=config)
    mts.export(language_model)

    tensors = {}
    for name, module in language_model.named_modules():
        if isinstance(module, nn.Linear):
            weight = module.weight.detach().to(torch.bfloat16)
            assert check_2to4(weight), f"{name} is not 2:4 sparse"
            tensors[f"{LLM_PREFIX}{name}.weight"] = weight.cpu().contiguous()
    print(f"sparse linears: {len(tensors)}", flush=True)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for f in args.model.iterdir():
        if f.name != "model.safetensors.index.json" and not (args.out_dir / f.name).exists():
            (args.out_dir / f.name).symlink_to(f.resolve())
    save_file(tensors, str(args.out_dir / SPARSE_SHARD))
    index = json.loads((args.model / "model.safetensors.index.json").read_text())
    for key in tensors:
        index["weight_map"][key] = SPARSE_SHARD
    (args.out_dir / "model.safetensors.index.json").write_text(json.dumps(index, indent=2))
    (args.out_dir / "sparsity.json").write_text(json.dumps(
        {"method": args.method, "pattern": "2:4", "scope": "vlm.model.language_model linears",
         "calibration_clips": clip_ids}, indent=2))
    print(f"wrote {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
