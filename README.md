# [OVSeg] Open-Vocabulary Semantic Segmentation with Mask-adapted CLIP

> **This repository is forked from [facebookresearch/ov-seg](https://github.com/facebookresearch/ov-seg) for the Smart Factory Capstone Design course.**
> Forked and maintained by **신재호 (Jaeho Shin)**.

<hr><hr>
**Open-Vocabulary Semantic Segmentation with Mask-adapted CLIP**
[Feng Liang](https://jeff-liangf.github.io/), [Bichen Wu](https://www.linkedin.com/in/bichenwu), [Xiaoliang Dai](https://sites.google.com/view/xiaoliangdai/), [Kunpeng Li](https://kunpengli1994.github.io/), [Yinan Zhao](https://yinan-zhao.github.io/), [Hang Zhang](https://hangzhang.org/), [Peizhao Zhang](https://www.linkedin.com/in/peizhao-zhang-14846042/), [Peter Vajda](https://sites.google.com/site/vajdap), [Diana Marculescu](https://www.ece.utexas.edu/people/faculty/diana-marculescu) <br>
Computer Vision and Pattern Recognition Conference (CVPR), 2023


---

## Quick Inference on Google Colab

> **No local setup required.** Run OVSeg inference directly in your browser using Google Colab.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jaehoshin123/OVSeg/blob/main/OVSeg_colab_inference.ipynb)

### What `OVSeg_colab_inference.ipynb` does

| Step | Description |
|------|-------------|
| **1. Environment setup** | Installs Miniconda, creates a Python 3.8 conda environment (`ovseg`) |
| **2. Dependency install** | Installs PyTorch 1.10.1+cu113, Detectron2, CLIP, and all required packages |
| **3. Repo & patch** | Clones this repo and patches a `numpy` API compatibility issue (`np.int` → `np.int64`) |
| **4. Checkpoint download** | Downloads the pretrained `ovseg_swinbase_vitL14_ft_mpt.pth` (~2 GB) from HuggingFace Hub |
| **5. Inference** | Runs `demo.py` with user-specified class names on a sample image |
| **6. Visualization** | Displays the original image alongside the OVSeg segmentation prediction |

**Requirements:** Google Colab with GPU runtime (T4 GPU or better recommended)

---

## Paper Overview

### Open-Vocabulary Semantic Segmentation with Mask-adapted CLIP

**Motivation**

Traditional semantic segmentation models are closed-vocabulary — they can only segment categories seen during training. OVSeg addresses this limitation by enabling segmentation of *arbitrary* categories specified as free-form text at inference time, without any retraining.

**Core Problem: CLIP's Mask-Region Mismatch**

[CLIP](https://openai.com/research/clip) is a vision-language model trained to align *full images* with text descriptions. When naively applied to segmentation, CLIP is asked to classify *masked regions* (cropped or blurred patches) rather than full images — a distribution shift that significantly degrades performance. The key insight of OVSeg is that this mismatch is the primary bottleneck for open-vocabulary segmentation.

**Proposed Solution: Mask-adapted CLIP**

OVSeg fine-tunes CLIP on a curated dataset of **(masked image region, category text)** pairs so that CLIP learns to recognize partially-visible, mask-cropped objects. This fine-tuning is lightweight and preserves CLIP's broad open-vocabulary knowledge while teaching it to handle masked regions effectively.

**Architecture**

```
Input Image
    │
    ▼
MaskFormer (Swin-B backbone)
    │  generates class-agnostic mask proposals
    ▼
For each mask proposal:
    ├─ Crop & mask the region from the image
    ├─ Encode with mask-adapted CLIP image encoder
    └─ Compare with CLIP text embeddings of target class names
    │
    ▼
Assign each mask the highest-similarity class name
    │
    ▼
Final Segmentation Map
```

- **Mask proposal network:** MaskFormer with a Swin-B backbone generates class-agnostic segment proposals covering the image.
- **Mask-adapted CLIP:** The CLIP image encoder is fine-tuned to encode masked/cropped regions robustly. The text encoder remains frozen to preserve open-vocabulary generalization.
- **Classification:** Each mask proposal is classified by computing cosine similarity between its CLIP image embedding and the CLIP text embeddings of all candidate class names.

**Training Data for Mask-Adaptation**

The fine-tuning dataset is constructed by pairing existing segmentation annotations (COCO, CC3M captions, etc.) with their corresponding masked image crops, creating ~300K (mask, text) pairs spanning diverse categories.

**Key Results (CVPR 2023)**

| Benchmark | Metric | OVSeg Score |
|-----------|--------|-------------|
| ADE20K-150 | mIoU | 24.8 |
| ADE20K-847 | mIoU | 9.0 |
| Pascal Context-59 | mIoU | 55.7 |
| Pascal Context-459 | mIoU | 12.4 |

OVSeg achieved state-of-the-art performance on all major open-vocabulary segmentation benchmarks at the time of publication.

**Why It Matters**

- **Zero-shot generalization:** Segment any category by just typing its name — no annotation or retraining needed.
- **Practical deployment:** A single model handles unlimited vocabularies, replacing the need for category-specific models.
- **CLIP compatibility:** The approach is model-agnostic; any CLIP-style vision-language model can be adapted the same way.

