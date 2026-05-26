# PROMPTS.md — Claude Code Prompt Log

본 파일은 Claude Code를 사용하여 `facebookresearch/ov-seg` 공식 저장소를 Google Colab 환경에서 학습 없이 추론만 수행하도록 만드는 과정에서 주고받은 프롬프트와, 각 프롬프트가 산출물(`OVSeg.ipynb`)에 어떻게 반영되었는지를 단계별로 기록한 것입니다.

프롬프트들은 가독성을 위해 일부 합치거나 다듬어 정리하였으며 클로드 상 세션의 대화 흐름은 본 로그가 정리한 순서입니다.


---

## 1. 프로젝트 셋업 및 목표 정의

### Prompt 1.1 — 초기 요구사항 정의

> OVSeg 논문(arXiv:2210.04150, CVPR 2023)의 공식 구현(`facebookresearch/ov-seg`)을 Google Colab에서 inference만 실행하는 Jupyter notebook을 만들자. 다음 조건을 만족.
>
> 1. Colab 런타임이 GPU로 설정되어 있는지 먼저 검증할 것
> 2. `ovseg` 라는 이름의 conda 환경을 만들고 그 안에서 모든 작업을 수행할 것
> 3. 공식 README가 명시한 PyTorch/Detectron2 버전을 그대로 사용할 것
> 4. 공식 Hugging Face Space `facebook/ov-seg` 에 올라가 있는 toy example 5개(`sample_01.jpeg` ~ `sample_05.png`)를 자동 다운로드하여 일괄 inference 할 것
> 5. 결과는 example ID 별로 별도 폴더에 저장하여 같은 이미지에 prompt set이 여러 개일 때 덮어쓰기가 발생하지 않도록 할 것

**Claude Code 응답 요약**
- 9단계 셀 구성을 제안: GPU 확인 → Miniconda 설치 → `ovseg` 환경 생성 → Repo clone & CLIP 설치 → 호환성 패치 → checkpoint 다운로드 → toy example 다운로드/메타데이터 작성 → 일괄 inference → 사용자 이미지 inference.
- 각 단계마다 셀이 단독으로 재실행 가능하도록 만들 것을 권고.


---

## 2. 환경 구성 (Miniconda + ovseg env)

### Prompt 2.1 — Miniconda 설치 셀

> Colab은 기본적으로 시스템 Python 3.10/3.11을 사용하지만, OVSeg 공식 README는 Python 3.8 + PyTorch 1.10.1 + CUDA 11.3 을 요구합니다. 시스템 Python을 건드리지 않고 conda 환경 안에서만 작업하고 싶습니다.
>
> 1. `/usr/local/miniconda` 가 이미 존재하면 재설치하지 말고 건너뛸 것
> 2. Miniconda3-py38 버전(`Miniconda3-py38_23.3.1-0-Linux-x86_64.sh`)을 사용할 것
> 3. `set -e` 로 어느 단계에서든 실패하면 즉시 중단되도록 할 것
>
> bash magic(`%%bash`) 셀로 작성해 주세요.

**Claude Code 응답 요약**
- `if [ ! -d /usr/local/miniconda ]` 로 idempotent 가드.
- `wget -q` 로 silent 다운로드, `bash ... -b -p` 로 비대화형 설치.
- 설치 후 `conda --version` 으로 sanity check.


---

### Prompt 2.2 — `ovseg` 환경 생성과 의존성 설치

> conda env 생성과 PyTorch + Detectron2 + OVSeg 의존성 설치를 하나의 bash 셀로 묶어 주세요. 다음 제약이 있습니다.
>
> 1. `RESET_ENV=0` 일 때는 기존 ovseg 환경을 재사용하고 패키지 설치도 건너뛸 것 (Colab 세션은 자주 끊기는데, 매번 재설치하면 30분 이상 걸려서 시간 낭비입니다)
> 2. `RESET_ENV=1` 일 때만 환경을 지우고 새로 만들 것
> 3. PyTorch는 `conda install` 이 너무 느려서 매번 solver 가 멈춥니다. 반드시 `pip wheel` (https://download.pytorch.org/whl/cu113/torch_stable.html) 로 설치할 것
> 4. 설치된 패키지 중 핵심(`torch==1.10.1`, `torchvision==0.11.2`, `detectron2`, `PIL`)을 `python -c` 로 import 해서 사전 점검할 것. 이게 통과하면 재설치를 건너뛸 것
> 5. NumPy 는 1.x 만 사용할 수 있으므로 `numpy==1.23.5` 로 고정할 것 (NumPy 2.x 는 detectron2 와 호환 안 됨)
> 6. Pillow 도 `9.5.0` 으로 고정할 것 (Pillow 10 은 `Image.LINEAR` 를 제거해서 detectron2 import 시 깨짐)

**Claude Code 응답 요약**
- heredoc (`python - <<'PY'`) 을 사용해 import-기반 sanity check 작성.
- detectron2 는 facebookresearch wheel 인덱스 (`https://dl.fbaipublicfiles.com/detectron2/wheels/cu113/torch1.10/index.html`) 에서 설치.
- `setuptools==59.5.0` 을 명시적으로 pin 해 detectron2 빌드 시 distutils 관련 에러 회피.
- 마지막에 버전 출력 블록을 추가해 `torch.cuda.is_available()` 까지 확인.


---

### Prompt 2.3 — 왜 numpy 1.23.5 인가?

> `numpy==1.23.5` 로 고정하는 이유를 코드 주석이 아니라 README 수준으로 한 문단 정리해 주세요. 발표 자료에 넣어야 합니다.

**Claude Code 응답 요약**
- NumPy 1.24 부터 `np.int`, `np.float`, `np.bool` 같은 deprecated alias 가 제거됨.
- OVSeg `predictor.py` 내부에 `np.int` 가 남아 있어 1.24+ 에서는 `AttributeError`.
- 더불어 detectron2 도 NumPy 2.x 와 ABI 가 달라 import 자체가 실패함.
- 따라서 1.23.5 가 detectron2/torch/OVSeg 세 가지 모두를 만족하는 마지막 안전 버전.


---

## 3. OVSeg Repo Clone, CLIP 설치, 호환성 패치

### Prompt 3.1 — Repo clone과 modified CLIP 설치

> 다음 작업을 한 셀에 담아 주세요.
>
> 1. `/content/ov-seg` 가 없으면 `git clone https://github.com/facebookresearch/ov-seg.git`
> 2. `cd /content/ov-seg/third_party/CLIP && pip install -Ue .` (OVSeg가 mask prompt tuning을 위해 패치한 CLIP)
> 3. 모두 `ovseg` conda env 안에서 실행

**Claude Code 응답 요약**
- `conda activate ovseg` 를 셀 상단에 명시.
- `if [ ! -d ov-seg ]` 가드.
- `pip install -Ue .` (editable + upgrade) 로 modified CLIP 설치.


---

### Prompt 3.2 — `np.int6464` 오타와 `np.int` deprecation 동시 패치

> 실제로 demo.py 를 돌려보니까 `np.int6464` 라는 이상한 attribute error 가 떴습니다. OVSeg 저장소의 `open_vocab_seg/utils/predictor.py` 를 grep 해보니 한 줄에 `np.int64` 가 들어가 있어야 할 자리에 `np.int6464` 가 적힌 오타가 있고, 다른 줄들에는 `np.int` 가 그대로 남아 있어서 numpy 1.24+ 에서는 어차피 깨지게 되어 있습니다.
>
> 1. 정규식 `r"np\.int(?![0-9])"` 로 `np.int` 만 `np.int64` 로 치환 (뒤에 숫자가 붙은 `np.int64`, `np.int32` 는 보존)
> 2. `np.int6464` 라는 문자열은 별도로 직접 치환
> 3. 패치 적용 후 어느 줄이 어떻게 바뀌었는지 출력
> 4. 파일이 없으면 silent 하게 warning만 찍고 통과
>
> 그리고 이걸 노트북 셀 안에서 실행해야 하니 `python - <<'PY' ... PY` heredoc 형식으로 부탁드립니다.

**Claude Code 응답 요약**
- `re.sub(r"np\.int(?![0-9])", "np.int64", s)` 패치 작성.
- `s.replace("np.int6464", "np.int64")` 를 먼저 적용 (정규식이 잘못 잡지 않도록 순서 보장).
- `predictor.read_text().splitlines()` 로 line-by-line print 추가하여 어떤 줄이 남았는지 검증.


---

### Prompt 3.3 — Pillow 10 호환 (`Image.LINEAR` → `Image.BILINEAR`)

> Detectron2 가 `data/transforms/transform.py` 안에서 `Image.LINEAR` 를 사용하는데 Pillow 10 부터 이게 제거되어서 import 자체가 실패합니다. 우리는 `Pillow==9.5.0` 으로 고정했지만, 누군가가 환경을 재사용할 때 Pillow 버전이 어긋날 수 있으니 detectron2 의 transform.py 도 자동으로 패치해 두고 싶습니다.
>
> - `detectron2.__file__` 로 설치 경로를 찾고
> - `data/transforms/transform.py` 내의 `Image.LINEAR` 를 `Image.BILINEAR` 로 치환
> - 이미 치환되어 있으면 그대로 통과
> - detectron2 가 아예 import 안 되면 silent 하게 skip

**Claude Code 응답 요약**
- `Path(detectron2.__file__).parent / "data/transforms/transform.py"` 로 경로 해석.
- `try/except` 로 detectron2 가 없는 환경도 안전하게 통과.
- 이미 패치된 경우(`"Image.LINEAR" not in s`)에는 "already OK" 출력.


---

## 4. Checkpoint 다운로드 (gdown → Hugging Face Hub)

### Prompt 4.1 — gdown 실패 우회

> 공식 OVSeg README 는 Google Drive 링크에서 `ovseg_swinbase_vitL14_ft_mpt.pth` (약 1.3GB) 를 받게 안내합니다. 그런데 `gdown` 으로 받으면 "Too many users have viewed this file" 에러가 거의 항상 나서 안정성이 떨어집니다.
>
> 이 체크포인트가 사실 `facebook/ov-seg` 라는 Hugging Face Space 에도 동일하게 올라가 있더라고요 (Space → Files 탭에서 확인). `huggingface_hub.hf_hub_download` 로 받는 방식으로 바꿔 주세요. 단, `repo_type="space"` 라는 것 잊지 말고, 이미 받아져 있으면 다시 받지 않도록 idempotent 하게.

**Claude Code 응답 요약**
- `hf_hub_download(repo_id="facebook/ov-seg", filename="ovseg_swinbase_vitL14_ft_mpt.pth", repo_type="space")` 사용.
- 다운로드 후 `shutil.copy` 로 `checkpoints/` 폴더로 이동 (HF 캐시는 `/root/.cache/...` 인데 demo.py 의 cwd 가 ov-seg repo 이므로 상대 경로 통일 필요).
- `target.exists()` 가드로 idempotency 보장.
- 최종적으로 `ls -lh checkpoints/` 로 파일 크기 확인.


---

## 5. Toy Examples 메타데이터와 일괄 추론

### Prompt 5.1 — 공식 toy example 5장 + 메타데이터 JSON

> Hugging Face Space `facebook/ov-seg` 의 examples 탭을 보면 5개의 demo example 이 있습니다. 각각 이미지 경로, proposal generator (`MaskFormer` 또는 `Segment_Anything`), granularity, class prompt list 가 있습니다.
>
> 1. 이 5개 example 을 list of dict 형태(`toy_examples`)로 노트북 안에 inline 정의해 주세요.
> 2. 같은 이미지(`sample_01.jpeg`)에 대해 prompt set 이 두 개(MaskFormer 용 짧은 prompt vs SAM 용 dense prompt) 인 경우가 있으므로 `id` 필드로 구분 가능하게 만들어 주세요.
> 3. 이미지는 `resources/demo_samples/` 아래에 받아두고, 이미 존재하면 다시 받지 말 것.
> 4. `proposal_gen_original` 은 metadata 로만 저장하고, 실제 inference 는 전부 demo.py + MaskFormer 흐름으로 통일하겠습니다 (공식 demo 도 Colab 에서는 MaskFormer 만 권장하므로).
> 5. 마지막에 `toy_examples.json` 으로도 저장해서 이후 inference 셀이 다시 읽을 수 있게.

**Claude Code 응답 요약**
- 5개 example 의 dict 구조 정의: `id`, `image`, `proposal_gen_original`, `granularity_original`, `class_names`.
- `sorted({ex["image"] for ex in toy_examples})` 로 중복 없는 이미지 다운로드.
- `Path("toy_examples.json").write_text(json.dumps(...))` 로 직렬화.


---

### Prompt 5.2 — 결과 폴더 분리

> sample_01 이미지로 inference 를 두 번(MaskFormer prompt set, SAM dense prompt set) 돌리면 demo.py 가 같은 파일명(`sample_01.jpeg`)으로 출력을 덮어씁니다. example ID 별로 폴더를 분리해 주세요.
>
> - 출력 root 는 `pred_toy_examples/`
> - 각 example 의 출력 폴더는 `pred_toy_examples/{example_id}/`

**Claude Code 응답 요약**
- `out_dir = pred_root / ex["id"]` 로 example 별 폴더.
- `subprocess.run(cmd, cwd=str(repo), check=True)` 로 demo.py 호출.
- 일괄 처리 후 `find pred_toy_examples -maxdepth 2 -type f` 로 산출물 목록 확인.


---

## 6. 결과 시각화

### Prompt 6.1 — 원본/예측 페어 시각화

> toy_examples.json 을 다시 읽어서 각 example 마다 (원본, OVSeg 예측) 두 장을 matplotlib 로 띄워 주세요. 제목에는 example_id 와 class_names 를 모두 표시.

**Claude Code 응답 요약**
- `for ex in examples:` 루프, 두 개의 `plt.figure()`.
- 원본 이미지가 없거나 prediction 이 없으면 "missing:" 으로 출력하고 skip.
- title 은 multi-line f-string 으로 example_id + proposal_gen_original + class_names 표기.


---

## 7. 사용자 정의 이미지 추론

### Prompt 7.1 — Custom inference 셀

> 마지막에 사용자가 자기 이미지를 넣어서 돌릴 수 있도록 `CUSTOM_IMAGE_PATH` 와 `CUSTOM_CLASS_NAMES` 두 변수만 수정하면 되는 inference 셀과, 결과 시각화 셀을 분리해서 만들어 주세요.

**Claude Code 응답 요약**
- 변수 두 개를 셀 상단에 명시.
- assertion (`CUSTOM_IMAGE_PATH.exists()`, `ckpt.exists()`) 으로 fail-fast.
- 출력 폴더 `pred_custom/` 분리.


---

## 8. 트러블슈팅 & 에러 처리

### Prompt 8.1 — CUDA OOM 가이드

> Colab T4 (15GB) 에서 `sample_01_sam_dense_prompts` 처럼 class prompt 가 9개인 example 을 돌리면 가끔 CUDA out-of-memory 가 납니다. 사용자에게 "prompt 수를 줄이거나 sample_02, sample_03 같은 짧은 prompt example 부터 시도해 보라"고 안내하는 markdown 셀을 하나 추가해 주세요.

**Claude Code 응답 요약**
- `## 트러블슈팅 > CUDA OOM` 섹션 추가.
- 짧은 prompt example 부터 검증 권장.


---


## 본 PROMPTS.md 생성


> 위에서 진행한 모든 프롬프트와 응답 요약을 단계별로 정리해서 PROMPTS.md 한 파일로 만들어 주세요. 표지에 학번/이름/타겟논문/도구/저장소를 넣고, 각 단계는 "Prompt N.M" 형식으로 번호를 매겨서 사람이 어느 단계에서 어떤 의사결정이 일어났는지 추적할 수 있게.

---

## Appendix A. 노트북 셀 인덱스 ↔ 프롬프트 매핑

| Cell | 종류 | 내용 | 근거 프롬프트 |
|------|------|------|---------------|
| 0 | md | 노트북 개요 / 9 단계 목차 | 1.1 |
| 1 | md | GPU 확인 안내 | 1.1 |
| 2 | code | `!nvidia-smi` | 1.1 |
| 3 | md | Miniconda 설치 섹션 | 2.1 |
| 4 | code | Miniconda 설치 (idempotent) | 2.1 |
| 5 | md | `ovseg` env 섹션 | 2.2 |
| 6 | code | conda env 생성 + PyTorch/Detectron2/OVSeg deps | 2.2, 2.3 |
| 7 | md | Repo clone / CLIP 설치 / 호환성 패치 섹션 | 1.1, 3.1, 3.2, 3.3 |
| 8 | code | git clone + CLIP editable install + np.int 패치 + Image.LINEAR 패치 | 3.1, 3.2, 3.3 |
| 9 | md | Checkpoint 섹션 | 4.1 |
| 10 | code | HF Hub 에서 checkpoint 다운로드 | 4.1 |
| 11 | md | Toy examples 섹션 | 5.1 |
| 12 | code | 5개 example 메타데이터 + 이미지 다운로드 | 5.1 |
| 13 | md | 원본 시각화 섹션 | 6.1 |
| 14 | code | sample 이미지 5장 시각화 | 6.1 |
| 15 | md | 일괄 inference 섹션 | 5.2 |
| 16 | code | example 별 폴더 분리하여 일괄 inference | 5.2 |
| 17 | md | 결과 시각화 섹션 | 6.1 |
| 18 | code | (원본, OVSeg 예측) 페어 시각화 | 6.1 |
| 19 | md | Custom inference 사용법 | 7.1 |
| 20 | code | Custom 이미지 inference | 7.1 |
| 21 | code | Custom 이미지 결과 시각화 | 7.1 |
| 22 | md | 트러블슈팅 (CUDA OOM) | 8.1 |
| 23 | code | (예약, 추가 실험용) | — |

---

## Appendix B. 사용 도구 환경

| 항목 | 값 |
|------|-----|
| AI 코딩 툴 | Claude Code |
| 모델 | claude-sonnet-4-5 |
| 작업 환경 | Google Colab (Tesla T4, 15GB VRAM) |
| 호스트 Python | 3.10 (Colab 기본값) |
| Conda env Python | 3.8 (`ovseg`) |
| 핵심 의존성 | PyTorch 1.10.1+cu113, torchvision 0.11.2+cu113, detectron2 (FB wheel), numpy 1.23.5, Pillow 9.5.0, setuptools 59.5.0 |
| 체크포인트 | `ovseg_swinbase_vitL14_ft_mpt.pth` (~1.3GB, Swin-B + ViT-L/14, FT+MPT) |
| Public 산출물 | https://github.com/jaehoshin123/OVSeg |

— *문서 끝*
