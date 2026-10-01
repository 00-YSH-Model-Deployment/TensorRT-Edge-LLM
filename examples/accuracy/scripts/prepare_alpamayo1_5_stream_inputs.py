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
"""Build action_inference inputs for consecutive windows of one clip (a streaming run).

Window k predicts at ``t0 + k * step``. With the default 100 ms step and the
loader's four frames 100 ms apart, consecutive windows share three of each
camera's four frames, decoded from the same video frames, so their pixels are
byte-identical and the runtime's encoder embedding cache can serve them.

Same request format and options as prepare_alpamayo1_5_inputs.py; each window
gets its own image directory so frames are never overwritten. Every window is
written ``--num-traj-samples`` times so run_alpamayo1_5_k1_latency.sh can take
one copy per window.
"""
import argparse
import json
from pathlib import Path

import physical_ai_av

from alpamayo1_5.load_physical_aiavdataset import load_physical_aiavdataset
from prepare_alpamayo1_5_inputs import DATASET_REVISION, build_request


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--clip-id", required=True)
    p.add_argument("--t0-us", type=int, default=5_100_000, help="First window's prediction time.")
    p.add_argument("--step-us", type=int, default=100_000)
    p.add_argument("--windows", type=int, default=40)
    p.add_argument("--data-cache", default=None)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--num-traj-samples", type=int, default=6)
    p.add_argument("--temperature", type=float, default=0.6)
    p.add_argument("--top-p", type=float, default=0.98)
    p.add_argument("--max-generate-length", type=int, default=256)
    args = p.parse_args()

    avdi = physical_ai_av.PhysicalAIAVDatasetInterface(cache_dir=args.data_cache,
                                                       revision=DATASET_REVISION)
    requests, gt = [], {}
    for k in range(args.windows):
        t0 = args.t0_us + k * args.step_us
        data = load_physical_aiavdataset(args.clip_id, t0_us=t0, avdi=avdi)
        request = build_request(args.clip_id, data, args.out_dir / "images" / args.clip_id / str(t0))
        window_id = f"{args.clip_id}@{t0}"
        request["id"] = window_id
        requests += [request] * args.num_traj_samples
        gt[window_id] = {
            "gt_xy": data["ego_future_xyz"][0, 0, :, :2].tolist(),
            "ego_history_xyz": data["ego_history_xyz"][0, 0].tolist(),
            "ego_history_rot": data["ego_history_rot"][0, 0].tolist(),
        }
        print(f"window {k + 1}/{args.windows} t0={t0}", flush=True)

    (args.out_dir / "inputs").mkdir(parents=True, exist_ok=True)
    header = {
        "batch_size": args.num_traj_samples,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": 0,
        "max_generate_length": args.max_generate_length,
    }
    (args.out_dir / "inputs" / "input_0000.json").write_text(json.dumps({**header, "requests": requests}))
    (args.out_dir / "gt.json").write_text(json.dumps(gt))
    print(f"{args.out_dir}: {args.windows} windows")


if __name__ == "__main__":
    main()
