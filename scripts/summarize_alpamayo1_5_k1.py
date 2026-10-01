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
"""Summarize run_alpamayo1_5_k1_latency.sh work dirs.

Per run: wall time per request (log timestamps), GPU time per request for each profiled
stage (vision per encoder call), decode forwards per request, and K=1 minADE at 6.4 s
against the PyTorch baseline's per-clip minADE. Runs after the first are also paired
with the first run clip by clip.

Run with the YSH-Inference-Alpamayo-1.5 Python (pandas):
    python summarize_alpamayo1_5_k1.py --baseline <per_clip.parquet> L1 [L2 ...]
"""
import argparse
import json
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
STAMP = r"^\[(\d\d:\d\d:\d\d\.\d+)\] \[INFO\] "
DECODE_STAGES = ("llm_generation", "spec_decode_base_verification")


def stamps(pattern: str, text: str) -> list[datetime]:
    return [datetime.strptime(m, "%H:%M:%S.%f") for m in re.findall(STAMP + pattern, text, re.M)]


def walls(run: Path) -> list[float]:
    out = []
    for log in sorted(run.glob("k1_outputs/output_*.log")):
        text = log.read_text()
        warmup = int(re.search(r"Warmup runs: (\d+)", text).group(1)) if "Warmup runs" in text else 0
        starts = stamps(r"\[llmRankRuntime\.cpp:\d+:multiModalRuntimePreprocess\] Processing trajectory", text)
        ends = stamps(r".*Processing complete", text)
        ts = starts + ends[-1:]
        # Warmup repeats a file's first request, so its measured run hits the encoder embedding
        # cache and skips the ViT; it is dropped with the warmup runs.
        out += [(b - a).total_seconds() * 1e3 for a, b in zip(ts, ts[1:])][warmup + 1:]
    return out


def stages(run: Path, n_requests: int) -> dict[str, float]:
    total: dict[str, float] = {}
    runs: dict[str, int] = {}
    for f in sorted(run.glob("k1_outputs/profile_*.json")):
        for s in json.loads(f.read_text())["stages"]:
            total[s["stage_id"]] = total.get(s["stage_id"], 0.0) + s["total_gpu_time_ms"]
            runs[s["stage_id"]] = runs.get(s["stage_id"], 0) + s["total_runs"]
    per_request = {k: v / n_requests for k, v in total.items()}
    per_request["vision_encoder"] = total.get("vision_encoder", np.nan) / max(runs.get("vision_encoder", 0), 1)
    per_request["decode_forwards"] = sum(runs.get(k, 0) for k in DECODE_STAGES) / n_requests
    return per_request


def min_ade(run: Path, python: str) -> pd.Series:
    requests, responses = [], []
    for out in sorted(run.glob("k1_outputs/output_*.json")):
        inp = run / "k1_inputs" / out.name.replace("output_", "input_")
        requests += json.loads(inp.read_text())["requests"]
        responses += json.loads(out.read_text())["responses"]
    (run / "k1_merged_input.json").write_text(json.dumps({"requests": [{"id": r["id"]} for r in requests]}))
    (run / "k1_merged_output.json").write_text(json.dumps({"responses": responses}))
    subprocess.run([
        python, str(REPO / "examples/accuracy/scripts/compute_minade.py"), "--input",
        str(run / "k1_merged_input.json"), "--output", str(run / "k1_merged_output.json"), "--gt",
        str(run / "gt.json"), "--num_traj_samples", "1", "--csv_out", str(run / "k1_minade.csv")
    ], check=True, capture_output=True)
    table = pd.read_csv(run / "k1_minade.csv")
    table["clip"] = table["clip_id"].astype(str).str.split("_").str[0]
    return table.set_index("clip")["min_ade_6.4s"]


def ratio_ci(a: np.ndarray, b: np.ndarray, rng: np.random.Generator) -> tuple[float, float, float]:
    """Paired mean ratio a/b - 1 with a bootstrap 95% interval, in percent."""
    idx = rng.integers(0, len(a), (4000, len(a)))
    boot = a[idx].mean(1) / b[idx].mean(1) - 1
    return 100 * (a.mean() / b.mean() - 1), 100 * np.percentile(boot, 2.5), 100 * np.percentile(boot, 97.5)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--baseline", required=True, help="PyTorch per_clip.parquet with clip_id and min_ade.")
    p.add_argument("--python", default=sys.executable, help="Python that runs compute_minade.py.")
    p.add_argument("runs", nargs="+", type=Path)
    args = p.parse_args()

    baseline = pd.read_parquet(args.baseline).set_index("clip_id")["min_ade"]
    rng = np.random.default_rng(0)
    rows, ades = [], {}
    for run in args.runs:
        w = np.array(walls(run))
        ade = min_ade(run, args.python)
        ades[run.name] = ade
        s = stages(run, len(ade))
        pair = pd.concat([ade, baseline], axis=1, join="inner").to_numpy()
        diff, lo, hi = ratio_ci(pair[:, 0], pair[:, 1], rng)
        rows.append({
            "run": run.name,
            "n": len(w),
            "wall_mean": w.mean(),
            "wall_median": np.median(w),
            "wall_p90": np.percentile(w, 90),
            "fps": 1000 / w.mean(),
            "vision": s.get("vision_encoder", np.nan),
            "prefill": s.get("llm_prefill", np.nan),
            "decode": sum(s.get(k, 0.0) for k in DECODE_STAGES) + s.get("spec_decode_draft_proposal", 0.0),
            "decode_forwards": s["decode_forwards"],
            "action": s.get("action_inference", np.nan),
            "ade6.4": ade.mean(),
            "vs_pytorch_%": f"{diff:+.1f} [{lo:+.1f}, {hi:+.1f}]",
        })
    pd.set_option("display.width", 250)
    print(pd.DataFrame(rows).round(1).to_string(index=False))
    first = args.runs[0].name
    for name, ade in list(ades.items())[1:]:
        pair = pd.concat([ade, ades[first]], axis=1, join="inner").to_numpy()
        diff, lo, hi = ratio_ci(pair[:, 0], pair[:, 1], rng)
        print(f"ADE6.4 {name} vs {first}: {diff:+.1f}% [{lo:+.1f}, {hi:+.1f}] over {len(pair)} clips")


if __name__ == "__main__":
    main()
