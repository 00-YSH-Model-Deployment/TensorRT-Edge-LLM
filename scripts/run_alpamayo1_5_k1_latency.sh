#!/usr/bin/env bash
# K=1 latency per clip: one request per clip, batch 1, with a profile per input file.
#
# Usage: L=<work dir holding inputs/ and engines/ from run_alpamayo1_5_thor.sh> bash $0
# The converter writes each clip 6 times (batch 6, one noise per sample); this keeps one
# copy per clip, cools the board to 50 C before each file and warms up WARMUP runs (default 25,
# about 60 s: the Thor GPU steps up only after ~40 s of load, so a few runs measure the slow state).
# TEMPERATURE overrides the inputs' sampling temperature (0 = greedy, which DFlash decoding forces).
# Results: $L/k1_outputs/{output,profile}_NNNN.json and .log.
set -euo pipefail
: "${L:?set L to the work dir}"
REPO="$(cd "$(dirname "$0")/.." && pwd)"
TRT_PACKAGE_DIR="${TRT_PACKAGE_DIR:-$HOME/Documents/Alpamayo/00_Model/TensorRT-11.0-pip}"
export LD_LIBRARY_PATH="$TRT_PACKAGE_DIR/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
BUILD_DIR="${BUILD_DIR:-$REPO/build-trt11}"
export EDGELLM_PLUGIN_PATH="$BUILD_DIR/libNvInfer_edgellm_plugin.so"
INFER_REPO="${INFER_REPO:-$HOME/Documents/Alpamayo/11_Inference/YSH-Inference-Alpamayo-1.5}"
PY="${PYTHON:-$INFER_REPO/a1_5_venv/bin/python}"
mkdir -p "$L/k1_inputs" "$L/k1_outputs"
# The converter writes each clip 6 times (batch 6, one noise per sample); keep one copy.
"$PY" - "$L" "${TEMPERATURE:-}" <<'PY'
import json, sys
from pathlib import Path
L = Path(sys.argv[1])
for f in sorted((L / "inputs").glob("input_*.json")):
    d = json.loads(f.read_text())
    d["batch_size"] = 1
    d["requests"] = d["requests"][::6]
    if sys.argv[2]:
        d["temperature"] = float(sys.argv[2])
    (L / "k1_inputs" / f.name).write_text(json.dumps(d))
    print(f.name, len(d["requests"]), "requests")
PY
for f in "$L"/k1_inputs/input_*.json; do
    n=$(basename "$f" .json | sed 's/input_//')
    PYTHONPATH="$INFER_REPO/src" "$PY" -c \
        "from alpamayo1_5.trace import thermal as T; print('cool', T.wait_until_cool(50))"
    "$BUILD_DIR/examples/multimodal/action_inference" \
        --engineDir "$L/engines/llm" --multimodalEngineDir "$L/engines" \
        --inputFile "$f" --outputFile "$L/k1_outputs/output_$n.json" \
        --warmup "${WARMUP:-25}" --profileOutputFile "$L/k1_outputs/profile_$n.json" --dumpProfile \
        > "$L/k1_outputs/output_$n.log" 2>&1
    echo "file $n done $(date +%T)"
done
