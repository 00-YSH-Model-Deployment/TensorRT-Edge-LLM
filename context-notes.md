# Alpamayo 1.5 → TensorRT-Edge-LLM 포팅 작업 노트

진행 상황은 [checklist.md](checklist.md). 여기는 결정과 그 이유를 쌓는다.

## 2026-09-27 — 레시피 선택

- `13_Model-Optimization/flashdrive/` 본체 (z-lab FlashDrive) 는 TensorRT 가 아니다. PyTorch + vLLM Marlin W4A8 + DFlash + torch.compile. 모든 단계가 GPU 필수이고, `-PARO`/`-DFlash` companion 이 stock 가중치 전용이라 fine-tune/pruned 체크포인트엔 쓸 수 없다.
- Alpamayo TRT 레시피는 이 레포 (TensorRT-Edge-LLM 0.10.1) 뿐이다. 공식 지원은 Alpamayo-R1-10B, FP16 만 (`docs/source/user_guide/examples/vla/alpamayo.md`).
- TensorRT-LLM fork 에는 Alpamayo 코드가 없다.
- 사용자 결정. 이번엔 CPU 단계 (ONNX export) 까지만, 대상은 원본 `nvidia/Alpamayo-1.5-10B`.

## 2026-09-27 — 1.5 포팅 범위

- 1.5 의 `model_type` 은 `alpamayo1_5`. Python dispatch 가 전부 `== "alpamayo_r1"` 이라 1.5 는 일반 모델로 취급되어 visual/action export 가 빠진다.
- 수정 대상 8곳 (계획 시 7곳으로 봤으나 `export.py` 의 LLM key remap 분기가 하나 더 있음).
  - `tensorrt_edgellm/scripts/export.py` — `_VISION_MODEL_TYPES`, `_ACTION_MODEL_TYPES`, `_is_alpamayo()`, LLM key remap 분기
  - `tensorrt_edgellm/chat_template.py` — `_is_alpamayo_1_model()`
  - `tensorrt_edgellm/checkpoint/checkpoint_utils.py` — LLM config promote, tokenizer 빌드
  - `experimental/builder/models/registry.py` — direct builder 매핑
- C++ 런타임은 `model_type` 문자열을 읽지 않는다 (grep 0건). 엔진 빌드/런타임은 수정 불필요.
- 호환 근거. 가중치 prefix (`vlm.model.language_model.*`, `vlm.model.visual.*`, `expert.*`), action expert 구성 (2048 / 16 head / 128 hd / 8256, 64 waypoint, Euler), special token 29개 위치, delta history tokenizer 가 R1 과 같다.
- base VLM 은 1.5 config 의 `vlm_name_or_path = nvidia/Cosmos-Reason2-8B` 를 그대로 읽는다 (코드가 config 값 우선, R1 기본값은 fallback).
- **min/max_pixels 는 수정하지 않는다.** `export.py` 의 `_save_alpamayo_visual_processor` 는 128·28·28 / 2048·32·32 로 하드코딩돼 있지만, upstream R1 과 1.5 모두 `helper.py` 에서 163840 / 196608 을 쓴다. 즉 1.5 고유 불일치가 아니라 R1 에도 똑같이 있는 선택이다. 이미지당 192 token (`--maxImageTokensPerImage 192` = 196608 / 32²) 이 되도록 입력 이미지를 미리 맞추는 쪽이 R1 가이드의 전제로 보인다. GPU 단계에서 PyTorch 와 visual token 수를 비교해 확인할 것.
- 범위 밖 (그대로 미지원). nav 텍스트 CFG 2-pass, noise temperature, camera-name / `frame i` 프롬프트 자동 생성 (입력 JSON 의 text 항목으로 수동 주입은 가능, 미검증).

## 2026-09-27 — 환경 구성

- venv 는 레포 안 `.venv` (gitignore 됨). Python 3.12, torch `2.13.0+cpu` (pytorch CPU index), `pip install -e ".[export]"` → transformers 5.14.1, onnx 1.19.0.
- **`uv pip` 에는 반드시 `--no-config`.** uv 는 `[tool.uv]` 가 없는 pyproject 를 건너뛰고 상위로 올라가 부모 `flashdrive/pyproject.toml` 의 `override-dependencies = ["torch==2.9.1"]` 를 적용한다. 처음엔 이것 때문에 `torch==2.13.0` 을 요청했는데 2.9.1 이 설치됐다.
- `pip install -e .` 는 `EDGELLM_PYTHON_ONLY_WHEEL=ON` 이라 CUDA/컴파일러 없이 된다. 단 `license-files` 검사 때문에 `git submodule update --init --recursive` 가 먼저 필요하다 (googletest / nlohmannJson / NVTX, 핀 리비전 그대로).
- 검증. 1.5 snapshot 으로 `_is_alpamayo`, `_VLM_MODEL_TYPES`, `_ACTION_MODEL_TYPES`, `_is_alpamayo_1_model`, registry components (LLM/VISUAL/ACTION) 모두 인식. `load_checkpoint_config_dicts` 가 Cosmos-Reason2-8B 에서 `qwen3_vl_text` (hidden 4096, 36 layer, 32/8 head, head_dim 128, FFN 12288) 를 끌어오고 vocab 을 155697 로 덮어씀. `AutoConfig` 실패 경고는 R1 과 같은 raw config fallback 경로라 정상.
- 단위 테스트 (`LLM_SDK_DIR=$PWD`) export_config / checkpoint_utils / chat_template_* / direct_builder_checkpoint_contract → 45 passed, 1 failed. 실패 1건은 `ModuleNotFoundError: tensorrt` (CPU venv 에 TRT Python 없음) 로 패치와 무관.
