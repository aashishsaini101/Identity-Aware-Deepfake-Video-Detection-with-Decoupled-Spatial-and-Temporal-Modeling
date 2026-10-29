# Identity-Aware Deepfake Video Detection with Decoupled Spatial and Temporal Modeling

**A Controlled Analysis Under Source-Aware Evaluation**

Official code for the paper *"Identity-Aware Deepfake Video Detection with Decoupled Spatial and Temporal Modeling: A Controlled Analysis Under Source-Aware Evaluation"* (under review at TMLR).

![PyTorch](https://img.shields.io/badge/PyTorch-2.5.1-ee4c2c?logo=pytorch&logoColor=white)
![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?logo=python&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)

---

## Overview

Many video deepfake detectors use temporal modeling, but its benefit is usually mixed up with the training procedure of the frame encoder. This repository implements a framework that keeps the two parts **separate**, so each can be tested under an identical training recipe:

- **Spatial encoder:** a ConvNeXt-Tiny backbone encodes every frame independently.
- **Temporal encoder:** a two-layer Transformer models the sequence of frame embeddings.

Because the stages are separable, the Transformer can be swapped for plain mean pooling while everything else stays the same. This gives a controlled measurement of what temporal modeling actually adds.

The pipeline is **identity-aware**: faces are detected with InsightFace and tracked with a ByteTrack-style tracker, so every 16-frame clip contains a single person.

### Main findings

| Finding | Evidence |
|---|---|
| The detector reaches pooled video AUROC **0.9924** and EER **1.98%** on 2,838 test videos. | Table 4 / Table 6 |
| Pooled metrics are inflated by one dominant source (DeeperForensics, 56.5% of test videos). The **macro-source AUROC is 0.9480**. | Table 6 |
| Under matched training, the temporal Transformer **does not improve** over frame averaging (spatial-only: pooled 0.9946 / macro 0.9487; full model: pooled 0.9924 / macro 0.9480). | Table 8 |
| Backbones trained *with* the Transformer give slightly stronger frozen frame embeddings (linear-probe macro AUROC 0.9834 vs 0.9683). This is a **hypothesis**: single run, single seed. | Table 8 |
| The model tolerates JPEG compression but is **vulnerable to blur and downscaling**. | Table 10 |
| Grad-CAM shows sparse temporal responses and some activation on alignment padding and hair, which is a possible shortcut. | Section 5.7 |

> **Scope of evaluation.** The test split is video-disjoint and source-stratified. Results measure generalization to *unseen videos from known sources*, **not** to unseen datasets or generation methods. Leave-one-dataset-out evaluation is future work.

---

## Method

```
Raw video
  -> frame sampling (~5 FPS)
  -> InsightFace detection + 5 landmarks
  -> ByteTrack-style identity association
  -> quality filter (Q >= 0.40)
  -> 5-point similarity alignment (224x224)
  -> non-overlapping 16-frame single-identity clips
  -> ConvNeXt-Tiny (shared weights, per frame)  -> F in R^{16x768}
  -> [CLS] + positional embeddings -> 2-layer Transformer -> [CLS] output
  -> LayerNorm -> Dropout(0.2) -> Linear(768,1) -> clip probability
  -> mean of clip probabilities -> video probability
  -> fake if p_video >= tau (tau fixed on validation set)
```

### 1. Identity-aware preprocessing

- **Frame sampling:** every `floor(f_native / 5)`-th frame, giving about 5 FPS.
- **Detection:** InsightFace returns boxes, confidence, and five landmarks.
- **Association:** ByteTrack-style tracker. High-confidence detections (score >= 0.6) are matched first, low-confidence ones (score >= 0.1) are recovered in a second pass, IoU matching threshold 0.8. Association happens *before* clip construction, so no clip mixes identities.
- **Quality filter:** `Q = 0.6 * min(Var(Laplacian(gray)) / 500, 1) + 0.4 * min(10 * A_face / A_img, 1)`. Faces with `Q < 0.40` are discarded.
- **Alignment:** similarity transform from the five landmarks to a fixed template, output 224 x 224.
- **Clips:** non-overlapping runs of T = 16 consecutive faces per identity (about 3.2 s at 5 FPS). Incomplete tails are discarded. Clips are stored as half-precision tensors in memory-mapped shards so decoding, detection, and tracking run once.

### 2. Spatial encoder

ConvNeXt-Tiny (ImageNet-22k pretrained, ImageNet-1k fine-tuned), classifier removed, global average pooling kept, drop-path 0.2. Each frame becomes a 768-d vector. A batch `(B, 16, 3, 224, 224)` is reshaped to `(B*16, 3, 224, 224)`, encoded in one pass, and reshaped back to `(B, 16, 768)`.

### 3. Temporal Transformer

A learnable `[CLS]` token is prepended and learnable positional embeddings `P in R^{17x768}` are added. The encoder has 2 pre-LayerNorm layers, 8 heads (d_k = 96), a 3072-d GELU feed-forward block, and dropout 0.1. Sequence length is 17 and the positional embeddings fix the clip length at 16.

### 4. Classification and video-level aggregation

`s_clip = w^T Dropout_0.2(LN(z_CLS)) + b`, `p_clip = sigmoid(s_clip)`.
The video score is the **mean** of the clip probabilities across all tracked identities in the video:

```
p_video = (1/N) * sum_i p_clip^(i)
```

Median and max pooling are evaluated as ablations (see below).

### 5. Training

Loss: binary cross-entropy on clip labels inherited from the source video.

1. **Backbone warm-up:** the backbone is first fine-tuned as a frame-level classifier on the training images for 5 epochs (lr 1e-5).
2. **Stage 1 (epochs 1-5):** the backbone is frozen, only the Transformer and head are trained.
3. **Stage 2 (epoch 6 onward):** joint training with differential learning rates, backbone 1e-5, Transformer 1e-4, head 5e-4.

| Setting | Value |
|---|---|
| Optimizer | AdamW, weight decay 1e-2 |
| Schedule | Cosine, 25-epoch budget |
| Batch size | 8, mixed precision |
| Early stopping | Patience 5 on validation video AUROC |
| Actual run | Stopped after epoch 19, best checkpoint at epoch 14 |
| Augmentation | None |
| Hardware | 1x NVIDIA GeForce RTX 3080 Ti, PyTorch 2.5.1, CUDA 12.1 |

---

## Data and evaluation protocol

### Datasets

Eight public sources are pooled: DeeperForensics-1.0, HiDF, FaceForensics++, Celeb-DF, DFDC, UADFV, DFD, and a small AI-generated set from social media (AI_DEEPFAKE). Partitioning is done at the **video-ID level**, stratified by source, so all clips of a video fall in one partition. No video ID is shared between validation and test. Source and video IDs are metadata only, never model inputs.

| Partition | Processed samples | Role |
|---|---|---|
| Training | 69,103 | Model optimization |
| Validation | 8,066 | Checkpoint selection and threshold calibration |
| Test | 8,446 (7,540 clips from 2,838 videos + 906 images) | Final held-out evaluation |

| Source | Test videos | Real | Fake |
|---|---|---|---|
| DeeperForensics | 1,604 | 1,384 | 220 |
| HiDF | 650 | 303 | 347 |
| FaceForensics++ | 316 | 58 | 258 |
| Celeb-DF | 158 | 22 | 136 |
| DFDC | 94 | 11 | 83 |
| UADFV | 9 | 4 | 5 |
| AI_DEEPFAKE | 5 | 0 | 5 |
| DFD | 2 | 2 | 0 |
| **Total** | **2,838** | **1,784** | **1,054** |

### Metrics

- Threshold-independent: AUROC, PR-AUC, EER. Threshold-dependent: accuracy, precision, recall, F1.
- **Pooled AUROC** is computed over all test videos.
- **Macro-source AUROC** averages per-source AUROC over the six sources containing both classes (AI_DEEPFAKE and DFD are single-class and excluded).
- 95% CIs come from 2,000 class-stratified bootstrap resamples (within each source for the macro score).
- Thresholds are **never** tuned on test labels. Three operating points are reported: default tau = 0.5, validation max-F1 tau = 0.7507, validation EER tau = 0.8363. Unless noted, tau = 0.5.

---

## Results

### Overall (tau = 0.5)

| Level | AUROC | PR-AUC | EER | Acc. | Prec. | Rec. | F1 |
|---|---|---|---|---|---|---|---|
| Clip (7,540) | 0.9852 | 0.9626 | 0.0311 | 0.9711 | 0.9491 | 0.9855 | 0.9670 |
| Video (2,838) | 0.9924 | 0.9759 | 0.0198 | 0.9789 | 0.9518 | 0.9934 | 0.9721 |

Pooled video AUROC 95% CI: 0.989-0.995.

### Per source (video level, tau = 0.5)

| Source | Videos (R/F) | AUROC [95% CI] | PR-AUC | EER | Acc. | F1 |
|---|---|---|---|---|---|---|
| DeeperForensics | 1,604 (1,384/220) | 1.0000 [1.000, 1.000] | 1.0000 | 0.0000 | 1.0000 | 1.0000 |
| HiDF | 650 (303/347) | 0.9843 [0.974, 0.993] | 0.9757 | 0.0233 | 0.9677 | 0.9704 |
| FaceForensics++ | 316 (58/258) | 0.8912 [0.835, 0.941] | 0.9560 | 0.1405 | 0.9209 | 0.9524 |
| Celeb-DF | 158 (22/136) | 0.9502 [0.879, 0.999] | 0.9846 | 0.0602 | 0.9494 | 0.9708 |
| DFDC | 94 (11/83) | 0.8620 [0.724, 1.000] | 0.9647 | 0.1424 | 0.9468 | 0.9708 |
| UADFV | 9 (4/5) | 1.0000 [1.000, 1.000] | 1.0000 | 0.0000 | 0.8889 | 0.9091 |
| **Macro-source (6)** | | **0.9480 [0.920, 0.972]** | | | | |
| **Pooled** | 2,838 | **0.9924 [0.989, 0.995]** | 0.9759 | 0.0198 | 0.9789 | 0.9721 |

Small sources (DFDC: 11 real videos, UADFV: 9 videos total) have wide confidence intervals; their per-source numbers are indicative, not definitive.

### Operating points (thresholds frozen on validation)

| Threshold | TN | FP | FN | TP | Acc. | Prec. | Rec. | F1 |
|---|---|---|---|---|---|---|---|---|
| Default, tau = 0.5000 | 1,731 | 53 | 7 | 1,047 | 0.9789 | 0.9518 | 0.9934 | 0.9721 |
| Validation max-F1, tau = 0.7507 | 1,736 | 48 | 14 | 1,040 | 0.9782 | 0.9559 | 0.9867 | 0.9711 |
| Validation EER, tau = 0.8363 | 1,740 | 44 | 16 | 1,038 | 0.9789 | 0.9593 | 0.9848 | 0.9719 |

### Comparison with reported results

Baseline numbers are copied from BMNet (Xiong et al., 2025). The protocols are **not identical**: baselines were trained and tested per dataset, whereas this model is trained once on pooled sources and tested on its own video-disjoint split. The table therefore shows relative standing, **not** a controlled head-to-head comparison. UADFV AUROC 100.00 is based on 9 videos. Retraining baselines on identical splits (e.g., with DeepfakeBench) is future work.

| Method | FF++ ACC/AUC | UADFV ACC/AUC | Celeb-DF ACC/AUC | DFDC ACC/AUC | Avg ACC/AUC |
|---|---|---|---|---|---|
| FWA | 77.60 / 81.17 | 88.70 / 95.21 | 50.74 / 55.71 | 82.71 / 84.23 | 74.94 / 79.08 |
| CViT | 84.09 / 90.27 | 52.67 / 58.37 | 51.67 / 60.22 | 71.02 / 75.61 | 64.86 / 71.12 |
| Capsule | 72.33 / 82.11 | 59.34 / 63.11 | 63.35 / 62.70 | 54.19 / 50.40 | 62.30 / 64.58 |
| MesoNet | 63.18 / 55.03 | 84.63 / 88.42 | 42.80 / 52.31 | 67.39 / 70.10 | 64.50 / 66.47 |
| Xception | 91.12 / 96.32 | 84.29 / 79.23 | 55.92 / 67.23 | 77.94 / 78.26 | 77.32 / 80.26 |
| BMNet | 95.54 / 98.60 | 92.18 / 97.21 | 80.20 / 75.72 | 84.72 / 88.51 | 88.16 / 90.01 |
| **Ours** | 92.09 / 89.12 | 88.89 / 100.00 | 94.94 / 95.02 | 94.68 / 86.20 | 92.65 / 92.59 |

BMNet remains stronger on FaceForensics++, on UADFV accuracy, and on DFDC AUC.

---

## Ablation studies

### Temporal modeling (controlled)

All four configurations share training data, initialization, and the model-selection criterion (validation video AUROC).

| Configuration | Val. AUROC | Pooled AUROC | Macro AUROC | PR-AUC | EER | F1 |
|---|---|---|---|---|---|---|
| (i) Spatial-only, end-to-end (mean pooling) | 0.9953 | 0.9946 | 0.9487 | 0.9847 | 0.0169 | 0.9747 |
| (ii) Spatial + Temporal Transformer (full model) | 0.9922 | 0.9924 | 0.9480 | 0.9759 | 0.0198 | 0.9721 |
| (iii) Frozen backbone of (i) + new linear head | 0.9985 | 0.9972 | 0.9683 | 0.9937 | 0.0169 | 0.9774 |
| (iv) Frozen backbone of (ii) + new linear head | 0.9985 | 0.9979 | 0.9834 | 0.9962 | 0.0152 | 0.9797 |

- (i) vs (ii): practically indistinguishable, and the differences are far smaller than the bootstrap CIs. **The Transformer adds no measurable discriminative information at about 5 FPS.**
- (iii)/(iv): linear probes (30 epochs, AdamW, lr 5e-4, weight decay 1e-2) on mean-pooled frozen features. Configuration (iv) needs no Transformer at inference, reducing parameters from 42.01 M to 27.82 M.
- Each configuration was trained **once**. The advantage of (iv) over (iii) should be confirmed over multiple seeds.

### Clip-to-video aggregation (full model)

| Aggregation | Pooled AUROC | Macro AUROC | PR-AUC | EER | F1 |
|---|---|---|---|---|---|
| Max | 0.9868 | 0.8936 | 0.9579 | 0.0212 | 0.9659 |
| Median | 0.9898 | 0.9138 | 0.9674 | 0.0215 | 0.9721 |
| **Mean** | **0.9924** | **0.9480** | **0.9759** | **0.0198** | 0.9721 |

Max pooling lets a single over-confident clip decide the label of a real video; median discards informative high-confidence clips.

---

## Robustness to post-processing

Perturbations are applied at inference time to the aligned 224 x 224 test faces. The model is **not** retrained and no degradation augmentation was used. Accuracy is at tau = 0.7507.

| Perturbation | Parameter | Pooled AUROC | Macro AUROC | PR-AUC | Acc. |
|---|---|---|---|---|---|
| None (clean) | | 0.9924 | 0.9480 | 0.9759 | 0.9782 |
| JPEG | Q = 75 | 0.9818 | 0.9212 | 0.9443 | 0.9605 |
| JPEG | Q = 50 | 0.9746 | 0.9476 | 0.9351 | 0.9313 |
| JPEG | Q = 30 | 0.9379 | 0.9256 | 0.8668 | 0.8721 |
| FFT low-pass | 40% retained | 0.9769 | 0.8754 | 0.9284 | 0.9602 |
| FFT low-pass | 20% retained | 0.8770 | 0.6937 | 0.7092 | 0.7886 |
| Gaussian blur | k = 3, sigma = 1.0 | 0.8865 | 0.6773 | 0.7234 | 0.8034 |
| Gaussian blur | k = 5, sigma = 1.5 | 0.6313 | 0.5679 | 0.4453 | 0.4655 |
| Gaussian blur | k = 7, sigma = 2.0 | 0.5430 | 0.5191 | 0.3927 | 0.3992 |
| Downscale | 112 px | 0.6696 | 0.5770 | 0.4727 | 0.5007 |
| Downscale | 56 px | 0.5258 | 0.5125 | 0.3839 | 0.3887 |

The detector is mostly robust to JPEG compression but vulnerable to blur and downscaling, suggesting heavy reliance on high-frequency facial detail. Degradation-aware training is a clear next step.

---

## Explainability

Grad-CAM is computed on the output of the last ConvNeXt stage for all 16 frames of a clip, w.r.t. the clip logit, with maps normalized jointly over the clip. The last-layer `[CLS]` attention weight per frame is reported alongside.

Observations (Section 5.7 of the paper):

1. **Sparse in time:** only one to three of the displayed frames carry substantial activation.
2. **Mixed spatial focus:** some peaks fall on facial regions (nose, mid-face, eyes), others on the hairline/forehead or on the black alignment padding. Reliance on padding is a potential shortcut, since its shape depends on head pose and may differ between sources.
3. **Temporal attention does not consistently track spatial evidence:** it is nearly uniform for many clips, and where peaked, the peak does not coincide with the strongest Grad-CAM frame.

These observations are consistent with the ablation: decisions rely mainly on per-frame appearance cues rather than temporal relations.

```bash
python3 visualize_explainability.py \
    --checkpoint ./checkpoints_temporal/best_checkpoint.pth \
    --test-dir ./data/shards/videos/test
```

---

## Computational cost

Profiled on one RTX 3080 Ti, batch size 1, 16-frame clips, network only.

| Metric | Value |
|---|---|
| Total parameters | 42.01 M |
| ConvNeXt-Tiny backbone | 27.82 M |
| Temporal Transformer + head | 14.19 M |
| GFLOPs per 16-frame clip / per frame | 71.44 / 4.46 |
| Latency per clip | 15.45 ms |
| Network throughput | about 1,035 frames/s |

These figures **exclude** video decoding, face detection, and tracking, which dominate end-to-end latency in deployment. The Transformer adds 14.19 M parameters but negligible compute; replacing it with mean pooling does not change latency significantly.

---

## Repository structure

```
.
├── build_manifest.py              # build the video/image manifest and splits
├── preprocess_pipeline_v3.py      # detection, tracking, quality filter, alignment, clipping
├── tracker.py                     # ByteTrack-style identity association
├── build_production_shards_v4.py  # serialize clips into memory-mapped shards
├── verify_shards.py               # shard sanity checks
├── verify_production_shards.py    # production-shard checks (e.g., split disjointness)
├── dataset_streaming.py           # streaming dataset / dataloader
├── models/                        # ConvNeXt-Tiny encoder + Temporal Transformer
├── train_temporal.py              # training (progressive unfreezing)
├── evaluate.py, evaluate1.py      # video/clip evaluation, bootstrap CIs, thresholds
├── ablation_phase3.py             # spatial-only vs temporal ablation
├── perturbation_test.py           # robustness to JPEG / blur / downscale / low-pass
├── visualize_explainability.py    # Grad-CAM + [CLS] attention
├── profile_efficiency.py          # parameters, GFLOPs, latency
├── infer_single_video.py          # run the detector on one video
├── checkpoints_spatial/           # spatial-only model checkpoints
├── checkpoints_temporal/          # full model checkpoints
├── ablation_results/
├── evaluation_results/
├── robustness_results/
├── LICENSE
└── README.md
```

## Installation

```bash
git clone https://github.com/aashishsaini101/Deepfake_detection.git
cd Deepfake_detection
pip install -r requirements.txt
```

Tested with Python 3.9+, PyTorch 2.5.1, CUDA 12.1.

## Usage

The following steps follow the order of the pipeline. Run any script with `--help` to see its arguments.

```bash
# 1. Preprocess videos -> aligned, identity-consistent 16-frame clips
python3 preprocess_pipeline_v3.py --help

# 2. Build shards and verify them
python3 build_production_shards_v4.py --help
python3 verify_production_shards.py --help

# 3. Train the full model (Stage 1 frozen backbone, Stage 2 joint)
python3 train_temporal.py --help

# 4. Evaluate (video-level, macro-source, bootstrap CIs, frozen thresholds)
python3 evaluate.py --help

# 5. Spatial-only vs temporal ablation
python3 ablation_phase3.py \
    --checkpoint ./checkpoints_temporal/best_checkpoint.pth \
    --data-path ./data/shards/videos/test \
    --frames 16

# 6. Robustness to post-processing
python3 perturbation_test.py \
    --checkpoint ./checkpoints_temporal/best_checkpoint.pth

# 7. Explainability
python3 visualize_explainability.py \
    --checkpoint ./checkpoints_temporal/best_checkpoint.pth

# 8. Single-video inference
python3 infer_single_video.py --help
```

## Limitations

- **Evaluation scope:** source-stratified, video-disjoint split only. No leave-one-dataset-out evaluation, so no claim is made about unseen generation methods.
- **Source imbalance:** DeeperForensics is 56.5% of test videos and is separated perfectly, inflating pooled metrics. Prefer the macro-source AUROC.
- **Small sources:** DFDC (11 real), Celeb-DF (22 real), and UADFV (4 real) give wide confidence intervals.
- **Single runs:** each ablation configuration was trained once; multi-seed confirmation is needed, especially for the frozen-backbone result.
- **Temporal modeling:** at about 5 FPS the Transformer gives no measurable benefit over frame averaging. Dense consecutive-frame clips and feature-difference inputs are untested. Only 16-frame clips were studied.
- **Degradation sensitivity:** large drops under blur and downscaling; no degradation augmentation was used.
- **Shortcut cues:** Grad-CAM shows activation on alignment padding and hair.
- **Comparison fairness:** baseline numbers are taken from the literature, not retrained on identical splits.

## Citation

The paper is under double-blind review. Please update this entry once it is published.

```bibtex
@article{anonymous2026identityaware,
  title   = {Identity-Aware Deepfake Video Detection with Decoupled Spatial and Temporal Modeling: A Controlled Analysis Under Source-Aware Evaluation},
  author  = {Anonymous},
  journal = {Under review at Transactions on Machine Learning Research},
  year    = {2026}
}
```

## Acknowledgements

Built on PyTorch, ConvNeXt, InsightFace, ByteTrack, Transformers, and Grad-CAM. Datasets: DeeperForensics-1.0, HiDF, FaceForensics++, Celeb-DF, DFDC, UADFV, and DFD.

## License

Released under the MIT License (see `LICENSE`).
