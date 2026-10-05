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
"""Build the DFlash draft's reduced vocabulary for Alpamayo 1.5.

The draft only proposes; the base verifies against its full vocabulary, so a
smaller draft lm_head cannot change the output, only how often proposals are
accepted. Alpamayo's reasoning uses a tiny vocabulary (861 distinct ids over
112k generated tokens on 7k non-evaluation samples), so the kept set is the
union of:

- every token the model generated in the given PyTorch runs' predictions
  (``token_ids``); clips in ``--exclude`` are skipped so evaluation clips do not
  pick the set,
- the ``--top-ids`` lowest ids as a general-language margin (byte-level BPE ids
  are roughly frequency-ordered), and/or a ``tensorrt-edgellm-reduce-vocab``
  output via ``--frequency-dir``,
- every special token (cot and trajectory markers must stay proposable),
  except the 4,000 trajectory bins, which generation never emits.

Writes ``vocab_map.safetensors`` (sorted int32 ids, padded to a multiple of 128
with the next unused low ids) for ``--draft-reduced-vocab-dir``.
"""
import argparse
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from safetensors import safe_open
from safetensors.torch import save_file


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--predictions", nargs="+", required=True,
                   help="predictions.parquet files (token_ids, clip_id columns).")
    p.add_argument("--exclude", action="append", default=[],
                   help="Parquet of clip ids whose tokens must not pick the set.")
    p.add_argument("--top-ids", type=int, default=8192)
    p.add_argument("--frequency-dir", type=Path, default=None)
    p.add_argument("--special-start", type=int, default=151643)
    p.add_argument("--traj-start", type=int, default=151669)
    p.add_argument("--traj-size", type=int, default=4000)
    p.add_argument("--vocab-size", type=int, default=155697)
    p.add_argument("--out-dir", type=Path, required=True)
    args = p.parse_args()

    excluded = set()
    for path in args.exclude:
        excluded |= set(pd.read_parquet(path)["clip_id"])
    generated, n_tokens = set(), 0
    for path in sorted({f for pattern in args.predictions for f in glob.glob(pattern)}):
        table = pd.read_parquet(path, columns=["clip_id", "token_ids"])
        for ids in table.loc[~table["clip_id"].isin(excluded), "token_ids"].dropna():
            ids = np.asarray(ids)
            generated.update(ids.tolist())
            n_tokens += ids.size

    traj = set(range(args.traj_start, args.traj_start + args.traj_size))
    special = set(range(args.special_start, args.vocab_size)) - traj
    kept = (generated - traj) | special | set(range(args.top_ids))
    if args.frequency_dir:
        with safe_open(args.frequency_dir / "vocab_map.safetensors", framework="pt") as f:
            kept |= set(f.get_tensor("vocab_map").tolist())
    next_id = args.top_ids
    while len(kept) % 128:
        if next_id not in kept and next_id not in traj:
            kept.add(next_id)
        next_id += 1
    vocab_map = torch.tensor(sorted(kept), dtype=torch.int32)
    assert int(vocab_map.max()) < args.vocab_size

    args.out_dir.mkdir(parents=True, exist_ok=True)
    save_file({"vocab_map": vocab_map}, str(args.out_dir / "vocab_map.safetensors"))
    info = {"reduced_vocab_size": len(kept), "vocab_size": args.vocab_size,
            "generated_distinct": len(generated), "generated_tokens": n_tokens,
            "generated_outside_top": len(generated - set(range(args.top_ids)) - special),
            "top_ids": args.top_ids, "excluded_clips": len(excluded)}
    (args.out_dir / "build_info.json").write_text(json.dumps(info, indent=2))
    print(json.dumps(info))


if __name__ == "__main__":
    main()
