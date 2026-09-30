#!/usr/bin/env bash
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
# Jetson Thor 에서 Alpamayo 1.5 의 C++ 빌드 → 엔진 빌드 → 입력 변환 → 추론 → minADE 채점을 단계별로 실행한다
#
# 사용법 (레포 루트에서):
#   RUN_GPU=1 ONNX_DIR=... WORK_DIR=... PYTHON=... bash scripts/run_alpamayo1_5_thor.sh <stage>...
#   stage: check | build | engines | inputs | infer | score | all
#
# 필수 환경변수:
#   ONNX_DIR  A100 에서 export 한 onnx/ (llm, visual, action 포함) 를 복사한 경로
#   WORK_DIR  엔진·입력·출력을 둘 경로 (엔진 약 22 GB, 이미지는 clip 당 약 30 MB)
#   PYTHON    YSH-Inference-Alpamayo-1.5 venv 의 python (inputs, score 단계에서만 사용)
# 선택:
#   CLIP_LIST (기본 gold644), LIMIT (앞 N clip 만), DATA_CACHE, ALLOW_STREAM=1, NOISE_SEED (기본 5)
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
: "${ONNX_DIR:?set ONNX_DIR}" "${WORK_DIR:?set WORK_DIR}"
INFER_REPO="${INFER_REPO:-$HOME/Documents/Alpamayo/YSH-Inference-Alpamayo-1.5}"
CLIP_LIST="${CLIP_LIST:-$INFER_REPO/notebooks/clip_ids_gold644.parquet}"
NUM_SAMPLES=6            # 엔진 maxBatchSize 와 입력 batch_size 가 같아야 한다
KV_CAPACITY=4096         # export 의 --max-kv-cache-capacity 와 같아야 한다
ENGINES="$WORK_DIR/engines"
BIN="$REPO/build/examples"

need_gpu() {
    [ "${RUN_GPU:-0}" = "1" ] || { echo "GPU 단계다. RUN_GPU=1 을 붙여 실행한다." >&2; exit 1; }
}

stage_check() {
    nvcc --version | tail -1
    dpkg -l | grep -E "libnvinfer10|tensorrt " || true
    local trt; trt="$(dpkg-query -W -f='${Version}' libnvinfer10 2>/dev/null || echo 0)"
    # action ONNX 의 trt::Attention / RotaryEmbedding / TensorScatter 는 TRT 10.15 이상에서만 파싱된다
    dpkg --compare-versions "$trt" ge 10.15 || echo "WARN: TensorRT $trt < 10.15, action_build 가 실패할 수 있다" >&2
    for d in llm visual action; do [ -f "$ONNX_DIR/$d/model.onnx" ] || { echo "missing $ONNX_DIR/$d" >&2; exit 1; }; done
    free -g | sed -n 2p
}

stage_build() {
    local cuda; cuda="$(nvcc --version | grep -oE 'release [0-9]+\.[0-9]+' | cut -d' ' -f2)"
    git -C "$REPO" submodule update --init --recursive
    cmake -S "$REPO" -B "$REPO/build" \
        -DCMAKE_BUILD_TYPE=Release \
        -DTRT_PACKAGE_DIR=/usr \
        -DCMAKE_TOOLCHAIN_FILE="$REPO/cmake/aarch64_linux_toolchain.cmake" \
        -DEMBEDDED_TARGET=jetson-thor \
        -DCUDA_CTK_VERSION="$cuda" \
        -DENABLE_CUTE_DSL=ALL
    cmake --build "$REPO/build" --parallel "$(nproc)" \
        --target llm_build visual_build action_build action_inference
}

stage_engines() {
    need_gpu
    ls "$ENGINES"/llm/*.engine >/dev/null 2>&1 || "$BIN/llm/llm_build" \
        --onnxDir "$ONNX_DIR/llm" --engineDir "$ENGINES/llm" \
        --maxInputLen 3424 --maxKVCacheCapacity "$KV_CAPACITY" --maxBatchSize "$NUM_SAMPLES"
    # 160 / 192 token 은 upstream MIN_PIXELS 163840 / MAX_PIXELS 196608 과 같은 픽셀 범위다
    [ -f "$ENGINES/visual/visual.engine" ] || "$BIN/multimodal/visual_build" \
        --onnxDir "$ONNX_DIR/visual" --engineDir "$ENGINES" \
        --minImageTokens 160 --maxImageTokens 18432 --maxImageTokensPerImage 192
    [ -f "$ENGINES/action/action.engine" ] || "$BIN/multimodal/action_build" \
        --onnxDir "$ONNX_DIR/action" --engineDir "$ENGINES" --maxBatchSize "$NUM_SAMPLES"
    du -sh "$ENGINES"/*
}

stage_inputs() {
    : "${PYTHON:?set PYTHON}"
    "$PYTHON" "$REPO/examples/accuracy/scripts/prepare_alpamayo1_5_inputs.py" \
        --clip-list "$CLIP_LIST" ${LIMIT:+--limit "$LIMIT"} \
        ${DATA_CACHE:+--data-cache "$DATA_CACHE"} ${ALLOW_STREAM:+--allow-stream} \
        --num-traj-samples "$NUM_SAMPLES" --out-dir "$WORK_DIR"
}

stage_infer() {
    need_gpu
    mkdir -p "$WORK_DIR/outputs"
    for input in "$WORK_DIR"/inputs/input_*.json; do
        local output; output="$WORK_DIR/outputs/$(basename "$input" | sed 's/^input_/output_/')"
        [ -f "$output" ] && continue     # 끊겨도 이어서 돈다
        "$BIN/multimodal/action_inference" \
            --engineDir "$ENGINES/llm" --multimodalEngineDir "$ENGINES" \
            --inputFile "$input" --outputFile "$output.tmp" --noiseSeed "${NOISE_SEED:-5}" \
            2>&1 | tee "$WORK_DIR/outputs/$(basename "$output" .json).log"
        mv "$output.tmp" "$output"
    done
}

stage_score() {
    : "${PYTHON:?set PYTHON}"
    # compute_minade.py 는 입력·출력 파일 한 쌍만 받으므로 나눠 돌린 파일을 순서대로 합친다
    "$PYTHON" - "$WORK_DIR" <<'EOF'
import json, sys
from pathlib import Path
work = Path(sys.argv[1])
requests, responses = [], []
for out in sorted((work / "outputs").glob("output_*.json")):
    inp = work / "inputs" / out.name.replace("output_", "input_")
    req, resp = json.loads(inp.read_text())["requests"], json.loads(out.read_text())["responses"]
    assert len(req) == len(resp), f"{out}: {len(req)} requests vs {len(resp)} responses"
    requests += req
    responses += resp
(work / "merged_input.json").write_text(json.dumps({"requests": [{"id": r["id"]} for r in requests]}))
(work / "merged_output.json").write_text(json.dumps({"responses": responses}))
print(f"merged {len(requests)} requests")
EOF
    "$PYTHON" "$REPO/examples/accuracy/scripts/compute_minade.py" \
        --input "$WORK_DIR/merged_input.json" --output "$WORK_DIR/merged_output.json" \
        --gt "$WORK_DIR/gt.json" --num_traj_samples "$NUM_SAMPLES" \
        --csv_out "$WORK_DIR/minade_results.csv" | tee "$WORK_DIR/minade_summary.txt"
}

[ $# -gt 0 ] || { sed -n '/^# Jetson Thor/,/^set -euo/{/^set -euo/!p}' "$0"; exit 1; }
for stage in "$@"; do
    case "$stage" in
        all) stage_check; stage_build; stage_engines; stage_inputs; stage_infer; stage_score ;;
        check|build|engines|inputs|infer|score) "stage_$stage" ;;
        *) echo "unknown stage: $stage" >&2; exit 1 ;;
    esac
done
