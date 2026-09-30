# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
# PhysicalAI-AV clip 을 Alpamayo 1.5 용 action_inference 입력 JSON + GT JSON 으로 변환한다
"""Build action_inference inputs for Alpamayo 1.5 from PhysicalAI-AV clips.

Run with the Python of the YSH-Inference-Alpamayo-1.5 repo (it provides the
``alpamayo1_5`` loader and ``physical_ai_av``); this script needs no GPU.

The prompt mirrors ``alpamayo1_5.helper.create_message``: every image is
preceded by ``"frame {i} "`` and the first frame of each camera by
``"{camera name}: "``. That is 38 content items per user message, so the
runtime must be built with ``kMaxContentItemsPerMessage >= 38``.

Each clip is emitted ``--num-traj-samples`` times in a row and ``batch_size``
is set to the same number. The runtime reseeds the diffusion noise for every
batch, so the samples of one clip only differ when they share a batch.

Outputs under ``--out-dir``:
    images/<clip_id>/cam<idx>_f<i>.png
    inputs/input_<NNNN>.json      (``--clips-per-file`` clips each; the runtime
                                   loads every image of a file into memory)
    gt.json                       (format read by compute_minade.py)
"""
import argparse
import json
from pathlib import Path

import pandas as pd
import physical_ai_av
from PIL import Image

from alpamayo1_5.helper import CAMERA_DISPLAY_NAMES
from alpamayo1_5.load_physical_aiavdataset import load_physical_aiavdataset

# Same pinned dataset snapshot as scripts/run_inference_tracked.py.
DATASET_REVISION = "b719eea7f0a63619ef51ec7f54178af0937ef050"
SYSTEM_PROMPT = "You are a driving assistant that generates safe and accurate actions."
USER_PROMPT = (
    "output the chain-of-thought reasoning of the driving process, "
    "then output the future trajectory."
)


def build_request(clip_id: str, data: dict, image_dir: Path) -> dict:
    image_dir.mkdir(parents=True, exist_ok=True)
    content = []
    for cam_idx, frames in zip(data["camera_indices"].tolist(), data["image_frames"]):
        cam_name = CAMERA_DISPLAY_NAMES.get(cam_idx, f"Camera {cam_idx}")
        content.append({"type": "text", "text": f"{cam_name}: "})
        for frame_idx, frame in enumerate(frames):
            path = image_dir / f"cam{cam_idx}_f{frame_idx}.png"
            if not path.exists():
                Image.fromarray(frame.permute(1, 2, 0).numpy()).save(path)
            content.append({"type": "text", "text": f"frame {frame_idx} "})
            content.append({"type": "image", "image": str(path.resolve())})
    content.append({"type": "trajectory", "trajectory": data["ego_history_xyz"][0, 0].tolist()})
    content.append({"type": "text", "text": USER_PROMPT})
    return {
        "id": clip_id,
        "messages": [
            {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
            {"role": "user", "content": content},
        ],
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--clip-id", action="append", help="Repeat for several clips.")
    p.add_argument("--clip-list", help="Parquet with a clip_id column.")
    p.add_argument("--limit", type=int, help="Use only the first N clips.")
    p.add_argument("--t0-us", type=int, default=5_100_000)
    p.add_argument("--data-cache", default=None, help="PhysicalAI-AV cache dir.")
    p.add_argument("--dataset-revision", default=DATASET_REVISION)
    p.add_argument("--allow-stream", action="store_true", help="Download missing data from HF.")
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--num-traj-samples", type=int, default=6)
    p.add_argument("--clips-per-file", type=int, default=8)
    p.add_argument("--temperature", type=float, default=0.6)
    p.add_argument("--top-p", type=float, default=0.98)
    p.add_argument("--max-generate-length", type=int, default=256)
    args = p.parse_args()

    clip_ids = list(args.clip_id or [])
    if args.clip_list:
        clip_ids += pd.read_parquet(args.clip_list)["clip_id"].tolist()
    clip_ids = clip_ids[: args.limit]
    assert clip_ids, "give --clip-id or --clip-list"

    avdi = physical_ai_av.PhysicalAIAVDatasetInterface(
        cache_dir=args.data_cache, revision=args.dataset_revision
    )
    (args.out_dir / "inputs").mkdir(parents=True, exist_ok=True)
    header = {
        "batch_size": args.num_traj_samples,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": 0,  # disabled, as upstream (top_k=None)
        "max_generate_length": args.max_generate_length,
    }

    gt = {}
    for file_idx, start in enumerate(range(0, len(clip_ids), args.clips_per_file)):
        requests = []
        for clip_id in clip_ids[start : start + args.clips_per_file]:
            data = load_physical_aiavdataset(
                clip_id, t0_us=args.t0_us, avdi=avdi, maybe_stream=args.allow_stream
            )
            request = build_request(clip_id, data, args.out_dir / "images" / clip_id)
            requests += [request] * args.num_traj_samples
            gt[clip_id] = {
                "gt_xy": data["ego_future_xyz"][0, 0, :, :2].tolist(),
                "ego_history_xyz": data["ego_history_xyz"][0, 0].tolist(),
                "ego_history_rot": data["ego_history_rot"][0, 0].tolist(),
            }
        path = args.out_dir / "inputs" / f"input_{file_idx:04d}.json"
        path.write_text(json.dumps({**header, "requests": requests}))
        print(f"{path}: {len(requests)} requests ({start + len(requests) // args.num_traj_samples}/{len(clip_ids)} clips)")

    (args.out_dir / "gt.json").write_text(json.dumps(gt))
    print(f"{args.out_dir / 'gt.json'}: {len(gt)} clips")


if __name__ == "__main__":
    main()
