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
"""Quantize Alpamayo 1.5's language model with ModelOpt, calibrated on driving clips.

Run with the Python of the YSH-Inference-Alpamayo-1.5 repo on a GPU; it provides
``alpamayo1_5``, ``physical_ai_av`` and nvidia-modelopt. ``tensorrt-edgellm-export``
then reads the output directory as an NVFP4 (or FP8) checkpoint.

Only ``vlm.model.language_model`` linears are quantized, which is what the llm engine
holds; the vision tower, ``lm_head`` and the action expert are untouched. Calibration
runs the full rollout per clip as the alpamayo1_5_quant recipe does (prefill, reasoning
generation, diffusion), so the quantizers see prompt and decode activations. Q/K/V and
gate/up share their per-tensor scales so the exporter can fuse them.

The output directory links every file of the source snapshot and adds one shard with the
quantized linears (packed ``weight``, ``weight_scale``, ``weight_scale_2`` for NVFP4,
``input_scale``); the index points those keys at the new shard. ``hf_quant_config.json``
names the format.
"""
import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("TORCH_DISABLE_NATIVE_JIT", "1")

import modelopt  # noqa: E402
import modelopt.torch.quantization as mtq  # noqa: E402
import pandas as pd  # noqa: E402
import physical_ai_av  # noqa: E402
import torch  # noqa: E402
from modelopt.torch.export.quant_utils import (  # noqa: E402
    get_activation_scaling_factor, get_quantization_format,
    get_weight_scaling_factor, get_weight_scaling_factor_2,
    preprocess_linear_fusion, to_quantized_weight)
from safetensors.torch import save_file  # noqa: E402

from alpamayo1_5 import helper  # noqa: E402
from alpamayo1_5.load_physical_aiavdataset import \
    load_physical_aiavdataset  # noqa: E402
from alpamayo1_5.models.alpamayo1_5 import Alpamayo1_5  # noqa: E402

DATASET_REVISION = "b719eea7f0a63619ef51ec7f54178af0937ef050"
LLM_PREFIX = "vlm.model.language_model."
FORMATS = {
    "nvfp4": (mtq.NVFP4_DEFAULT_CFG, "NVFP4", 16),
    "fp8": (mtq.FP8_DEFAULT_CFG, "FP8", None),
}
QUANT_SHARD = "model-edgellm-quant.safetensors"


def pick_clips(args) -> list[str]:
    clips = pd.read_parquet(args.clip_list)
    if args.split:
        clips = clips[clips["split"] == args.split]
    excluded = set()
    for path in args.exclude:
        excluded |= set(pd.read_parquet(path)["clip_id"])
    return sorted(set(clips["clip_id"]) - excluded)[:args.num_clips]


def make_forward_loop(model, processor, avdi, clip_ids, args):

    def forward_loop(_language_model):
        with torch.no_grad():
            for i, clip_id in enumerate(clip_ids):
                data = load_physical_aiavdataset(clip_id,
                                                 t0_us=args.t0_us,
                                                 avdi=avdi)
                messages = helper.create_message(
                    frames=data["image_frames"].flatten(0, 1),
                    camera_indices=data["camera_indices"])
                inputs = processor.apply_chat_template(
                    messages,
                    tokenize=True,
                    add_generation_prompt=False,
                    continue_final_message=True,
                    return_dict=True,
                    return_tensors="pt")
                model_inputs = helper.to_device(
                    {
                        "tokenized_data": inputs,
                        "ego_history_xyz": data["ego_history_xyz"],
                        "ego_history_rot": data["ego_history_rot"],
                    }, "cuda")
                # The engines run FP16 activations, as the recipe calibrates.
                with torch.autocast("cuda", dtype=torch.float16):
                    model.sample_trajectories_from_data_with_vlm_rollout(
                        data=model_inputs,
                        top_p=args.top_p,
                        temperature=args.temperature,
                        num_traj_samples=1,
                        max_generation_length=args.max_generation_length)
                print(f"calibrated {i + 1}/{len(clip_ids)} {clip_id}",
                      flush=True)

    return forward_loop


def unify_fused_scales(language_model) -> None:
    for layer in language_model.layers:
        attn, mlp = layer.self_attn, layer.mlp
        preprocess_linear_fusion([attn.q_proj, attn.k_proj, attn.v_proj])
        preprocess_linear_fusion([mlp.gate_proj, mlp.up_proj])


def pack(language_model, block_size) -> dict[str, torch.Tensor]:
    tensors = {}
    for name, module in language_model.named_modules():
        # get_quantization_format also answers for parents of quantized linears.
        quantizer = getattr(module, "weight_quantizer", None)
        if quantizer is None or not quantizer.is_enabled:
            continue
        fmt = get_quantization_format(module)
        key = LLM_PREFIX + name
        weight_scale = get_weight_scaling_factor(module)
        weight_scale_2 = get_weight_scaling_factor_2(module)
        tensors[f"{key}.weight"] = to_quantized_weight(
            module.weight.detach(), weight_scale, fmt, weight_scale_2,
            block_size).cpu()
        tensors[f"{key}.weight_scale"] = weight_scale.cpu()
        if weight_scale_2 is not None:
            tensors[f"{key}.weight_scale_2"] = weight_scale_2.cpu()
        tensors[f"{key}.input_scale"] = get_activation_scaling_factor(
            module).cpu()
    return tensors


def write_checkpoint(src: Path, out: Path, tensors: dict[str, torch.Tensor],
                     algo: str, block_size, clip_ids: list[str]) -> None:
    out.mkdir(parents=True, exist_ok=True)
    for f in src.iterdir():
        if f.name == "model.safetensors.index.json":
            continue
        link = out / f.name
        if not link.exists():
            link.symlink_to(f.resolve())
    save_file({k: v.contiguous()
               for k, v in tensors.items()}, str(out / QUANT_SHARD))
    index = json.loads((src / "model.safetensors.index.json").read_text())
    for key in tensors:
        index["weight_map"][key] = QUANT_SHARD
    (out / "model.safetensors.index.json").write_text(
        json.dumps(index, indent=2))
    quantization = {
        "quant_algo": algo,
        "kv_cache_quant_algo": None,
        "exclude_modules": ["lm_head", "vlm.lm_head", "vlm.model.visual*",
                            "expert*", "action_*"],
    }
    if block_size:
        quantization["group_size"] = block_size
    (out / "hf_quant_config.json").write_text(
        json.dumps(
            {
                "producer": {
                    "name": "modelopt",
                    "version": modelopt.__version__
                },
                "quantization": quantization,
                "calibration_clips": clip_ids,
            },
            indent=2))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--model", type=Path, required=True,
                   help="Alpamayo 1.5 snapshot directory.")
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--format", choices=sorted(FORMATS), default="nvfp4")
    p.add_argument("--clip-list", required=True,
                   help="Parquet with clip_id (and split) columns.")
    p.add_argument("--split", default="train")
    p.add_argument("--exclude", action="append", default=[],
                   help="Parquet of clips never to calibrate on (eval sets).")
    p.add_argument("--num-clips", type=int, default=64)
    p.add_argument("--t0-us", type=int, default=5_100_000)
    p.add_argument("--data-cache", default=None)
    p.add_argument("--temperature", type=float, default=0.6)
    p.add_argument("--top-p", type=float, default=0.98)
    p.add_argument("--max-generation-length", type=int, default=256)
    args = p.parse_args()

    clip_ids = pick_clips(args)
    assert clip_ids, "no calibration clips left after the exclusions"
    print(f"{len(clip_ids)} calibration clips", flush=True)
    avdi = physical_ai_av.PhysicalAIAVDatasetInterface(
        cache_dir=args.data_cache, revision=DATASET_REVISION)
    model = Alpamayo1_5.from_pretrained(str(args.model),
                                        dtype=torch.bfloat16,
                                        attn_implementation="sdpa").to(
                                            "cuda").eval()
    processor = helper.get_processor(model.tokenizer)
    language_model = model.vlm.model.language_model

    cfg, algo, block_size = FORMATS[args.format]
    mtq.quantize(language_model, cfg,
                 make_forward_loop(model, processor, avdi, clip_ids, args))
    mtq.print_quant_summary(language_model)
    unify_fused_scales(language_model)
    tensors = pack(language_model, block_size)
    print(f"packed {sum(k.endswith('.input_scale') for k in tensors)} linears",
          flush=True)
    write_checkpoint(args.model, args.out_dir, tensors, algo, block_size,
                     clip_ids)
    print(f"wrote {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
