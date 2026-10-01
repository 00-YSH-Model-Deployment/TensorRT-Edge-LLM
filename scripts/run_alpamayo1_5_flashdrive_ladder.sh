#!/usr/bin/env bash
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
# FlashDrive techniques stacked in Edge-LLM, one config after another on the 30 latency30 clips, K=1.
# Each file cools to 50 C and warms up about 60 s (run_alpamayo1_5_k1_latency.sh).
# Under WORK_ROOT: lat30-trt11/ (inputs, images, gt.json) and the engine dirs work-trt11/ (nvidia),
# zlab/ (z-lab action head), dflash/ (DFlash base + draft), nvfp4/ (NVFP4 DFlash base + the same draft).
# INPUT_SET picks the input dir under WORK_ROOT (default lat30-trt11; lat30-3cam drops the telephoto)
# and RUN_PREFIX the run dir prefix (default lat30w).
# Usage: bash run_alpamayo1_5_flashdrive_ladder.sh <config>...   (default: all, in order)
# Summarize with scripts/summarize_alpamayo1_5_k1.py on the lat30w-* dirs.
set -uo pipefail
R="$(cd "$(dirname "$0")/.." && pwd)"
M="${WORK_ROOT:-$HOME/Documents/Alpamayo/00_Model/Alpamayo-1.5_EdgeLLM}"
INPUT_SET="${INPUT_SET:-lat30-trt11}"
RUN_PREFIX="${RUN_PREFIX:-lat30w}"
S8C4=(EDGELLM_ALPAMAYO_DENOISE_STEPS=8 EDGELLM_ALPAMAYO_CACHE_STEPS=3,4,5,6)
DFLASH=(EDGELLM_DFLASH_BLOCK=8 EDGELLM_DFLASH_CONTEXT_WINDOW=8 TEMPERATURE=0)

run() {  # <dir name> <engines dir> <warmup> [ENV=VALUE...]
    local name=$1 engines=$2 warmup=$3; shift 3
    local L=$M/$RUN_PREFIX-$name
    mkdir -p "$L"
    ln -sfn "$engines" "$L/engines"
    for x in gt.json images inputs; do ln -sfn "../$INPUT_SET/$x" "$L/$x"; done
    rm -rf "$L/k1_outputs"
    echo "== $name start $(date +%T) $*"
    env WARMUP="$warmup" L="$L" "$@" bash "$R/scripts/run_alpamayo1_5_k1_latency.sh" > "$L/ladder.log" 2>&1
    echo "== $name done $(date +%T) rc=$?"
}

configs=("$@")
[ ${#configs[@]} -gt 0 ] || configs=(nv-s10 s10 s8c4 s8c4-greedy s8c4-dflash s8c4-dflash-nvfp4)
for c in "${configs[@]}"; do
    case $c in
        nv-s10)            run nv-s10            ../work-trt11/engines 25 ;;
        s10)               run s10               ../zlab/engines 25 ;;
        s8c4)              run s8c4              ../zlab/engines 25 "${S8C4[@]}" ;;
        s8c4-greedy)       run s8c4-greedy       ../zlab/engines 25 "${S8C4[@]}" TEMPERATURE=0 ;;
        s8c4-dflash)       run s8c4-dflash       ../dflash/engines 40 "${S8C4[@]}" "${DFLASH[@]}" ;;
        s8c4-dflash-nvfp4) run s8c4-dflash-nvfp4 ../nvfp4/engines 45 "${S8C4[@]}" "${DFLASH[@]}" ;;
        s10-dflash-nvfp4|s5-dflash-nvfp4|s2-dflash-nvfp4)
            steps=${c%%-*}
            run "$c" ../nvfp4/engines 45 EDGELLM_ALPAMAYO_DENOISE_STEPS="${steps#s}" "${DFLASH[@]}"
            ;;
        *) echo "unknown config $c" >&2 ;;
    esac
done
