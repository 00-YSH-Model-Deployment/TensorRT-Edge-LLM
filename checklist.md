# Alpamayo 1.5 → TensorRT-Edge-LLM 포팅 체크리스트

대상: `nvidia/Alpamayo-1.5-10B` (원본). 이번 범위는 CPU 만 (2026-09-27 사용자 결정).
결정과 근거는 [context-notes.md](context-notes.md).

## Phase 0 — CPU (이번 세션)

- [x] 전용 venv `.venv` (Python 3.12, torch 2.13.0 CPU wheel, `.[export]`)
- [x] `alpamayo1_5` model_type 인식 패치 (Python dispatch 8곳)
- [x] 패치 단위 검증 (`_is_alpamayo`, `_is_alpamayo_1_model`, LLM config promote, registry)
- [x] CPU ONNX export → `/home/Humble/extra2/trt/alpamayo1_5_edgellm/onnx`
- [x] ONNX 구조 검증 (llm / visual / action 존재, checker, action I/O 이름, kv capacity 4096)
- [x] GPU 요구사항 표 확정

## Phase 0.5 — Thor 실행 준비 (CPU, 2026-09-30)

- [x] `kMaxContentItemsPerMessage` 18 → 64 (1.5 프롬프트는 user 메시지 항목 38개)
- [x] 입력 변환기 `examples/accuracy/scripts/prepare_alpamayo1_5_inputs.py` (1 clip 스트리밍 시험, upstream `create_message` 와 user content 문자열 일치)
- [x] Thor 단계 실행 스크립트 `scripts/run_alpamayo1_5_thor.sh` (score 단계만 가짜 출력으로 시험)

## Phase 1 — Thor (2026-09-30 사용자 결정, A100 은 여유 없어 보류)

- [x] ONNX — 복사 대신 Thor 에서 다시 export (2026-10-01, CPU 약 6분, `~/Documents/Alpamayo/00_Model/Alpamayo-1.5_EdgeLLM/onnx`, llm 16 GB / visual 1.1 GB / action 4.3 GB)
- [x] `check` — JetPack 7.1 의 TensorRT 는 10.13.3.9 다. 예상대로 action 엔진만 실패했다 (TensorScatter / Attention 파싱 불가). pip TensorRT 11.0 으로 우회했다 (아래 노트)
- [x] `build` — C++ 빌드. 플러그인 `NvInfer_edgellm_plugin` 이 빌드 대상에서 빠져 있어 추가했다. TensorRT 11 빌드는 `build-trt11/`
- [x] `engines` — TensorRT 11 로 llm 16 GB / visual 1.1 GB / action 4.3 GB (`00_Model/Alpamayo-1.5_EdgeLLM/work-trt11/engines`)
- [x] `LIMIT=1` 스모크 (clip 021a4585, 6 sample) → CoC 텍스트가 PyTorch 와 같은 문장, minADE6@6.4s 0.038 m. visual token 수는 미확인
- [ ] `inputs` + `infer` + `score` — gold644 × 6 sample, minADE6
- [ ] 같은 644 clip PyTorch 1.5 (`run_inference_tracked.py`) 와 분포 비교 + 지연

## 보류 — A100 1장 (GPU 여유 ≥ 40GB + 사용자 허락 후)

- [ ] CUDA toolkit (nvcc) 유저 공간 설치 + TensorRT 10.15 이상 tarball (시스템 TRT 는 10.8 이라 부족)
- [ ] `kernelSrcs/build_cutedsl.py --gpu_arch sm_80`
- [ ] C++ 빌드 (`-DCUTE_DSL_ARTIFACT_TAG=sm_80 -DENABLE_CUTE_DSL=ALL`)
- [ ] `llm_build` / `visual_build` / `action_build`
