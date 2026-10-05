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
"""Build the chain-of-causation corpus for the runtime's retrieval draft.

Alpamayo's reasoning is short and templated, so the continuation of a partly
decoded CoC can be looked up in CoCs the model generated before
(``EDGELLM_RETRIEVAL_DRAFT=<corpus>``, see cpp/runtime/decoding/retrievalDraft.h).
The base model verifies every proposal, so the corpus changes how many tokens a
verify round accepts, never the output.

Reads the generated ``token_ids`` (the tokens after ``<|cot_start|>``) of the
given PyTorch predictions.parquet runs, the same inputs as
build_alpamayo1_5_draft_vocab.py. Clips in ``--exclude`` are skipped so the
clips the speed is measured on do not seed their own drafts, e.g.::

    N=11_Inference/YSH-Inference-Alpamayo-1.5/notebooks
    python build_alpamayo1_5_coc_corpus.py --predictions 'runs/*/predictions.parquet' \\
        --exclude $N/clip_ids_latency30.parquet --exclude $N/clip_ids_gold644.parquet \\
        --exclude $N/clip_ids_pilot400.parquet --out coc_corpus.bin

Each sequence is cut after its first ``<|traj_future_start|>`` (the token the
runtime stops on), or after ``<|cot_end|>`` when that marker is missing;
sequences with neither, or longer than ``--max-tokens``, are dropped. Every
sample is kept, duplicates included: the runtime gates proposals on how often a
continuation occurs.

Writes ``--out`` (the bytes "COCR", then little-endian int32: version 1, number
of sequences N, number of tokens T, offsets[N + 1], tokens[T]) and a JSON
summary next to it.
"""
import argparse
import collections
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

MAGIC = b"COCR"
VERSION = 1


def cut(ids: np.ndarray, cot_end: int, traj_future_start: int):
    """Generated ids through the first stop marker, or None when there is none."""
    for marker in (traj_future_start, cot_end):
        hits = np.flatnonzero(ids == marker)
        if hits.size:
            return ids[:hits[0] + 1], marker
    return None, None


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--predictions", nargs="+", required=True,
                   help="predictions.parquet files or glob patterns (clip_id, token_ids columns).")
    p.add_argument("--exclude", action="append", default=[],
                   help="Parquet of clip ids whose outputs must not enter the corpus.")
    p.add_argument("--cot-end", type=int, default=155678, help="<|cot_end|> id.")
    p.add_argument("--traj-future-start", type=int, default=155681,
                   help="<|traj_future_start|> id.")
    p.add_argument("--max-tokens", type=int, default=128,
                   help="Drop longer sequences (degenerate samples).")
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    excluded = set()
    for path in args.exclude:
        excluded |= set(pd.read_parquet(path)["clip_id"])

    paths = sorted({f for pattern in args.predictions for f in glob.glob(pattern)})
    assert paths, "no predictions.parquet matched --predictions"
    sequences, cots, sources, clips = [], collections.Counter(), [], set()
    stats = collections.Counter()
    for path in paths:
        names = pq.ParquetFile(path).schema_arrow.names
        columns = [c for c in ("clip_id", "token_ids", "cot") if c in names]
        table = pd.read_parquet(path, columns=columns)
        stats["rows"] += len(table)
        keep = ~table["clip_id"].isin(excluded)
        stats["rows_excluded"] += int((~keep).sum())
        used = 0
        for row in table.loc[keep].itertuples(index=False):
            if getattr(row, "token_ids", None) is None:
                stats["rows_without_token_ids"] += 1
                continue
            ids, marker = cut(np.asarray(row.token_ids, dtype=np.int64), args.cot_end,
                              args.traj_future_start)
            if ids is None:
                stats["dropped_without_marker"] += 1
                continue
            if ids.size > args.max_tokens:
                stats["dropped_too_long"] += 1
                continue
            ended = "traj_future_start" if marker == args.traj_future_start else "cot_end"
            stats[f"ended_at_{ended}"] += 1
            sequences.append(ids.astype("<i4"))
            cots[getattr(row, "cot", None) or ""] += 1
            clips.add(row.clip_id)
            used += 1
        sources.append({"path": path, "rows": len(table), "used": used})

    assert sequences, "no sequence left after exclusion"
    lengths = np.array([s.size for s in sequences], dtype=np.int64)
    offsets = np.concatenate([[0], np.cumsum(lengths)]).astype("<i4")
    tokens = np.concatenate(sequences).astype("<i4")
    assert int(tokens.min()) >= 0 and tokens.size < 2**31

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "wb") as f:
        f.write(MAGIC)
        np.array([VERSION, len(sequences), tokens.size], dtype="<i4").tofile(f)
        offsets.tofile(f)
        tokens.tofile(f)

    distinct = len({s.tobytes() for s in sequences})
    summary = {
        "corpus": str(args.out),
        "num_sequences": len(sequences),
        "num_tokens": int(tokens.size),
        "distinct_sequences": distinct,
        "distinct_tokens": int(np.unique(tokens).size),
        "mean_length": float(lengths.mean()),
        "max_length": int(lengths.max()),
        "clips": len(clips),
        "excluded_clips": len(excluded),
        **{k: int(v) for k, v in sorted(stats.items())},
        "top_cots": [{"cot": c, "count": n} for c, n in cots.most_common(10)],
        "sources": sources,
        "exclude": args.exclude,
    }
    args.out.with_suffix(".json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k not in ("top_cots", "sources")}))


if __name__ == "__main__":
    main()
