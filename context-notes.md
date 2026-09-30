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
- **`.[export]` 만으로는 Alpamayo export 가 안 된다. `.[export,tools]` 로 설치해야 한다.** 1차 시도에서 두 곳이 깨졌다.
  - tokenizer 빌드가 `Qwen2VLImageProcessor` 를 쓰는데 pillow / torchvision 이 없어 실패 → WARNING 만 찍고 넘어가서 tokenizer 파일이 안 생기고 chat template 이 fallback 으로 떨어짐 (`<|image_pad|>` token ID 못 찾음). **조용히 넘어가므로 로그에서 `Failed to build Alpamayo tokenizer` 를 반드시 확인.**
  - visual export 가 `tensorrt_edgellm.quantization` 을 import → modelopt / requests / datasets 등 tools 스택 전체가 필요 (`ModuleNotFoundError: modelopt`).
  - 1차 로그는 `/home/Humble/extra2/trt/alpamayo1_5_edgellm/export.attempt1.log`. LLM ONNX 는 1차에서도 7분 (16:58→17:05) 에 16GB 로 완료됐었다.
- `pip install -e .` 는 `EDGELLM_PYTHON_ONLY_WHEEL=ON` 이라 CUDA/컴파일러 없이 된다. 단 `license-files` 검사 때문에 `git submodule update --init --recursive` 가 먼저 필요하다 (googletest / nlohmannJson / NVTX, 핀 리비전 그대로).
- 검증. 1.5 snapshot 으로 `_is_alpamayo`, `_VLM_MODEL_TYPES`, `_ACTION_MODEL_TYPES`, `_is_alpamayo_1_model`, registry components (LLM/VISUAL/ACTION) 모두 인식. `load_checkpoint_config_dicts` 가 Cosmos-Reason2-8B 에서 `qwen3_vl_text` (hidden 4096, 36 layer, 32/8 head, head_dim 128, FFN 12288) 를 끌어오고 vocab 을 155697 로 덮어씀. `AutoConfig` 실패 경고는 R1 과 같은 raw config fallback 경로라 정상.
- 단위 테스트 (`LLM_SDK_DIR=$PWD`) export_config / checkpoint_utils / chat_template_* / direct_builder_checkpoint_contract → 45 passed, 1 failed. 실패 1건은 `ModuleNotFoundError: tensorrt` (CPU venv 에 TRT Python 없음) 로 패치와 무관.

## 2026-09-27 — export 2차 (성공 경로)

- `processed_chat_template.json` 은 **이미 있으면 다시 만들지 않는다** (`checkpoint_utils.py` `write_runtime_artifacts` 끝부분 `if not os.path.exists(template_dst)`). 1차의 fallback (`User: / Assistant:`) 이 2차에도 그대로 남아 있었다. 같은 출력 디렉터리로 재실행할 땐 이 파일을 먼저 치울 것.
  - 조치. stale 파일을 `/home/Humble/extra2/trt/alpamayo1_5_edgellm/processed_chat_template.attempt1_fallback.json` 로 옮기고 `process_chat_template(model_dir, out/llm)` 만 다시 호출. 결과는 Qwen `<|im_start|>` 형식 + `generation_prompt = "<|im_start|>assistant\n<|cot_start|>"` + image/video content type.
  - `<|cot_start|>` 가 1.5 에도 맞다. upstream 1.5 `helper.create_message` (주행 추론) 는 `<|cot_start|>`, `<|answer_start|>` 는 `create_vqa_message` (VQA) 전용.
- 소스 snapshot 기준 `Could not find token ID for '<|image_pad|>'` WARNING 은 정상 (1.5 snapshot 엔 tokenizer 가 없고 VLM 쪽에 있음). 출력 디렉터리 기준 경고가 없어야 정상.

## 2026-09-27 — export 결과와 구조 검증

- 명령 (CPU, `CUDA_VISIBLE_DEVICES=""`, `OMP_NUM_THREADS=16`, `HF_HUB_OFFLINE=1`, `HF_HOME=/home/Humble/extra2/Model`).
  `tensorrt-edgellm-export <1.5 snapshot 7aba829> /home/Humble/extra2/trt/alpamayo1_5_edgellm/onnx --max-kv-cache-capacity 4096`
- 소요 약 9분 (17:07 → 17:16, LLM 7분 / visual 20초 / action 2분). exit 0, traceback 0.
- 산출물 (`/home/Humble/extra2/trt/alpamayo1_5_edgellm/onnx`, 로그 `../export.log`).

  | 구성 | ONNX 외부 가중치 | 추가 파일 | 비고 |
  |---|---|---|---|
  | llm | 15.17 GB | embedding 1.28 GB, tokenizer (155697), chat template | qwen3_vl_text 36L, 36× AttentionPlugin |
  | visual | 1.16 GB | preprocessor_config | qwen3_vl vision 27L, deepstack 3 |
  | action | 4.56 GB | config (kv capacity 4096) | 36L expert 1 step, trt::Attention/RotaryEmbedding/TensorScatter |

  action 크기는 DL4AGX head 엔진 (4.57 GB, A100 FP16) 과 일치.
- 구조 검증. 세 ONNX 모두 `onnx.checker.check_model` 통과. action I/O 이름이 `cpp/common/bindingNames.h` 와 일치. `traj_token_start = 151669 + 3000 = 154669` (history delta tokenizer 1000 bin), `traj_vocab_size 4000` 과 정합.
- **수치 검증은 못 했다.** 세 모델 모두 TRT 전용 커스텀 op 를 써서 onnxruntime 으로 실행 불가. AGENTS.md 도 "export 만으로는 모델 동작 증거가 아니다, export → build → inference 까지" 라고 명시. 다음 GPU 단계에서 PyTorch 대비 paired minADE 로 확인해야 한다.

## GPU 요구사항 (export 산출물 크기 기반, 엔진 빌드/추론은 미실측)

FP16 가중치 합계 = 15.17 + 1.28 + 1.16 + 4.56 ≈ **22.2 GB**. KV cache 4096 tok × 36 L × 8 kv head × 128 × K,V × 2 B ≈ 0.6 GB / 시퀀스 (batch 6 이면 3.6 GB).

| 작업 | GPU | VRAM | 비고 |
|---|---|---|---|
| ONNX export | 불필요 | — | 완료 (CPU, RAM 약 32 GB 이상) |
| 엔진 빌드 | sm_80+ (Ampere 이상) | **40 GB 이상 권장** | 빌드 피크 가중치의 1.5–2×. 엔진은 빌드한 GPU 아키텍처 + TRT 버전에서만 동작 |
| 추론 FP16 (batch 1–6) | 빌드와 같은 GPU | **32 GB 이상** | 가중치 22 GB + KV 0.6–3.6 GB + 활성화 |
| Thor 배포 | Jetson / DRIVE Thor | 통합 128 GB | Thor 에서 재빌드 필수 |

- 적합 예. A100 40/80 GB, L40S 48 GB, RTX 6000 Ada 48 GB, RTX PRO 6000 96 GB, H100.
- 24 GB 급 (RTX 4090 / 3090) 은 FP16 가중치만 22 GB 라 비권장. Alpamayo 는 FP16 only 라 양자화로 줄일 수도 없음.
- 추가로 필요한 것. CUDA 12.8 toolkit (nvcc), TensorRT 10.x dev, `kernelSrcs/build_cutedsl.py --gpu_arch sm_80` (x86 sm_80 CuTe DSL 커널은 동봉 안 됨).
- 현재 이 호스트 (2026-09-27 16:37 기준) 는 A100 3장 모두 다른 테넌트가 약 67 GB 사용 → 장당 여유 약 13 GB 라 불가.

## 2026-09-30 — Thor 로 진행 결정과 실행 준비

- **사용자 결정.** 엔진 빌드·추론은 Thor 에서 한다. A100 3장은 다른 테넌트가 장당 약 66 GB 를 쓰고 있어 여유가 약 15 GB 뿐이다. RTX 4090 (24 GB) 은 FP16 가중치만 22.2 GB 라 불가. 검증 규모는 gold644 전체.
- 이 호스트 driver 는 580.126.09 (CUDA 13.0) 로 올라가 있다. 시스템 TensorRT 는 10.8.0.43 이라 action ONNX 를 못 읽는다.
- **TensorRT 10.15 이상 필요.** action ONNX 는 `trt::RotaryEmbedding` ×72, `trt::TensorScatter` ×72, `trt::Attention` ×36 을 쓰고, schema 주석이 "consumed by TRT >= 10.15" 다 (`tensorrt_edgellm/onnx/onnx_custom_schemas.py`). `limitations.md` 에 JetPack 7.1 이 TRT 10.13.3.9 라고 적혀 있어, JetPack 7.0/7.1 Thor 에서는 action_build 가 실패할 수 있다. `check` 단계가 경고한다.
- **`kMaxContentItemsPerMessage` 18 → 64.** 1.5 는 `helper._build_image_content` 가 카메라마다 `"{name}: "`, 프레임마다 `"frame {i} "` text 를 넣어 user 메시지가 4 + 16 + 16 + trajectory + text = 38 항목이다. R1 은 정확히 18 이었다. 상수만 바꿨고 이 호스트엔 툴체인이 없어 컴파일은 Thor 에서 처음 한다.
- **이전 노트 정정 두 건.**
  - min/max_pixels 는 문제가 아니다. C++ 런타임은 `preprocessor_config.json` 을 쓰지 않고 visual 엔진의 `--minImageTokens 160 --maxImageTokensPerImage 192` 로 크기를 정한다. 160·32² = 163840, 192·32² = 196608 로 upstream 과 같다. 1920×1080 은 320×576, 180 token 이 된다.
  - `compute_minade.py` 의 정규화 상수는 1.5 에서도 그대로 맞다. 1.5 config 값을 bf16 으로 반올림하면 정확히 그 값이고, PyTorch 모델도 버퍼를 bf16 으로 캐스팅한다.
- **clip 당 6 sample 은 한 배치로 묶어야 한다.** `initializeNoiseTrajectory` 가 배치마다 같은 seed 로 generator 를 새로 만든다. batch 1 로 6번 반복하면 6개가 같은 noise 를 받아 CoC 샘플링 차이만 남는다. 그래서 변환기는 같은 request 를 6번 연속으로 쓰고 `batch_size 6`, 엔진은 `--maxBatchSize 6` 이다. 모든 clip 이 같은 6개 noise 를 쓴다는 점은 남는다 (`--noiseSeed` 로만 바뀜).
- **입력 파일을 나눈다.** `action_inference` 는 파싱 시점에 파일 안 모든 request 의 이미지를 메모리에 올린다 (같은 파일을 6번 읽어도 6번 올림). 1920×1080 RGB 16장 × 6 × 8 clip 이면 약 4.8 GB 라 기본 `--clips-per-file 8`. 파일마다 엔진을 다시 로드하므로 지연 측정은 파일 첫 배치를 빼고 봐야 한다.
- 샘플링은 upstream 기본값에 맞춘다. temperature 0.6, top_p 0.98, top_k 0 (비활성, upstream 은 None), max_generate_length 256.
- PyTorch 와는 noise 를 맞출 수 없어 (Edge-LLM 은 seed 만 받음) clip 별 1:1 이 아니라 분포 비교다.
- 미확인 위험. action expert 가 쓰는 TRT 네이티브 Attention 레이어가 Thor TRT 버전에서 빌드되는지, batch 6 추론의 실제 메모리.
