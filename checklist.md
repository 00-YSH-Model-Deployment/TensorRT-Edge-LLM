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

## Phase 1 — A100 1장 (GPU 여유 ≥ 40GB + 사용자 허락 후)

- [ ] CUDA 12.8 toolkit (nvcc) 유저 공간 설치
- [ ] `kernelSrcs/build_cutedsl.py --gpu_arch sm_80`
- [ ] C++ 빌드 (`-DCUTE_DSL_ARTIFACT_TAG=sm_80 -DENABLE_CUTE_DSL=ALL`)
- [ ] `llm_build` / `visual_build` / `action_build`
- [ ] visual token 수 PyTorch 대비 확인 (processor pixel 범위 R1 하드코딩 100352 / 2097152 vs upstream 163840 / 196608)
- [ ] `action_inference` 로 PhysicalAI-AV 소수 clip, PyTorch 대비 paired minADE + 지연

## Phase 2 — Thor

- [ ] ONNX 복사 → Thor 에서 엔진 재빌드 + 추론
