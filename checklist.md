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

- [ ] ONNX (`/home/Humble/extra2/trt/alpamayo1_5_edgellm/onnx`, 약 22 GB) 를 Thor 로 복사
- [ ] `check` — JetPack 의 TensorRT 가 10.15 이상인지 (미만이면 action_build 실패 가능)
- [ ] `build` — C++ 빌드 (sm_110 CuTe DSL prebuilt 동봉, 커널 빌드 불필요)
- [ ] `engines` — llm / visual / action, maxBatchSize 6
- [ ] `LIMIT=1` 스모크 → CoC 텍스트와 궤적이 그럴듯한지, visual token 이 이미지당 180 인지
- [ ] `inputs` + `infer` + `score` — gold644 × 6 sample, minADE6
- [ ] 같은 644 clip PyTorch 1.5 (`run_inference_tracked.py`) 와 분포 비교 + 지연

## 보류 — A100 1장 (GPU 여유 ≥ 40GB + 사용자 허락 후)

- [ ] CUDA toolkit (nvcc) 유저 공간 설치 + TensorRT 10.15 이상 tarball (시스템 TRT 는 10.8 이라 부족)
- [ ] `kernelSrcs/build_cutedsl.py --gpu_arch sm_80`
- [ ] C++ 빌드 (`-DCUTE_DSL_ARTIFACT_TAG=sm_80 -DENABLE_CUTE_DSL=ALL`)
- [ ] `llm_build` / `visual_build` / `action_build`
