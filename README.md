# Event-Based Sign-Language Recognition

Spiking neural networks for recognising sign language and gestures from **event-camera**
data, with a controlled study of how much of the visual field the task actually needs.

Master's dissertation project. Two public DVS128 datasets, no new recording.

---

## What this project asks

An event camera reports only *changes* in brightness, so it is already a sparse sensor.
This project asks whether the signal can be reduced further **without losing accuracy**:

| Arm | Input | Question |
|---|---|---|
| **A** | Full event frame | What is the accuracy ceiling? |
| **B** | Hand region only | Does the hand alone carry the signal? |
| **C** | Full frame, much smaller network | How few parameters are enough? |

Arms A and B use an **identical network** and differ only in input, so any gap measures
the cost of discarding non-hand data rather than a modelling difference.

---

## Datasets

| | SL-Animals-DVS | DVS128 Gesture |
|---|---|---|
| Task | 19 sign-language animal signs | 11 hand/arm gestures |
| Samples | 1,121 | 1,341 |
| Subjects | 59 (4 recording sessions) | 29 |
| Sensor | DVS128, 128×128, events only | DVS128, 128×128 |
| Split | **Subject-independent**, 41/9/9 users | Official train/test, val carved from train |
| Split sizes | 779 / 171 / 171 | 915 / 162 / 264 |
| Chance | 5.3% | 9.1% |

Splits were audited for leakage: **zero subject overlap** and zero sample-file overlap
between train, validation and test.

Neither dataset is redistributed here. Download instructions are below.

---

## Results

All figures are **means over multiple seeds** with standard deviations, selected on
validation and evaluated once on test. Best-of-N single runs are not reported as headlines.

### Main results — `ghostsew12` (~240k parameters, T=16)

| Dataset | A — full frame | B — hand only | Gap |
|---|---|---|---|
| **SL-Animals** | **90.2% ± 0.9** (n=4) | 77.8% ± 4.1 (n=4) | 12.4 |
| **DVSGesture** | **91.9% ± 0.2** (n=3) | 87.1% ± 2.6 (n=4) | 4.8 |

A DVS-tuned variant (`dvsnet`, 495k params) reaches **92.8%** on DVSGesture (n=1).

### Hand-region reduction is a trade-off curve, not a single point

The mask keeps the densest-scoring pixels holding `keep_frac` of the event mass:

| `keep_frac` | Events kept | Reduction | SL-Animals accuracy | Gap to full |
|---|---|---|---|---|
| 0.5 | 38.4% | 2.6× | 77.8% ± 4.1 | 12.4 |
| 0.7 | 56.5% | 1.8× | **82.7% ± 2.9** | 7.5 |

The cost of hand-only reduction depends on **task granularity**: it is far cheaper for
coarse arm gestures (DVSGesture, 4.8 points) than for fine-grained sign language
(SL-Animals, 12.4 points), where the sign is defined by hand *configuration*.

### Architecture comparison (SL-Animals, full frame)

| Model | Params | Test |
|---|---|---|
| **ghostsew12** | 247k | **90.2% ± 0.9** |
| deep_ghost_drop | 51k | 81.3% |
| deep_ghost | 51k | 80.1% |
| deep5_ghost | 108k | 79.5% ± 2.0 |
| deep5 | 174k | 76.6% |
| deep | 80k | 69.0% |
| ghost | 32k | 66.7% |
| standard | 335k | 64.3% |
| base | 43k | 61.4% |
| temporal attention | 43k | 49.7% |
| PLIF (learnable τ) | 43k | 49.1% |

### Comparison with published work

**SL-Animals-DVS**

| Method | Accuracy |
|---|---|
| SLAYER (original paper, full set) | 60.9 ± 4.6% |
| SLAYER (optimised reimplementation) | 66.4 ± 5.8% |
| SNN-HDC | 74.1% |
| STBP | ~77% |
| SLAYER (paper, *reduced* set) | 78.0 ± 3.1% |
| **This work** | **90.2 ± 0.9%** (full set) |

**DVS128 Gesture**

| Method | Accuracy | Params |
|---|---|---|
| MSVIT | 98.8% | 1.67M |
| Spikformer | 98.2% | ~2.6M |
| SEW-ResNet | 97.9% | — |
| **This work** | **92.8%** | **495k** |

> Published SL-Animals results typically use 4-fold cross-validation; this work uses a
> single fixed subject-independent split, so the comparison is not strictly like-for-like.
> Test sets are small (171 and 264 samples), giving ±2.3% and ±1.7% binomial standard error.

---

## Architecture

`ghostsew12` — a spiking residual network combining two ideas:

```
Input [B, T=16, 2, 128, 128]
  │
STEM   Conv(2→16) → BatchNorm → LIF → MaxPool → MaxPool        128 → 32
  │    (early downsampling is required: a spiking net stores activations
  │     for all T timesteps, so 128×128 blocks exhaust a 6 GB GPU)
  │
12 × Ghost-SEW residual block:
         ┌── GhostConv → BN → LIF → GhostConv → BN → LIF ──┐
     x ──┤                                                  (+) ──→ out
         └──────────────── shortcut ───────────────────────┘
     (channels double at blocks 4 and 8; pooling 32 → 16 → 8)
  │
AdaptiveAvgPool(4) → Flatten → Linear(→ classes)
  │
Mean firing rate over T timesteps → class scores
```

| Component | Role |
|---|---|
| **LIF neurons** | Accumulate charge, emit a spike, reset — the spiking mechanism |
| **Ghost convolutions** | Half the feature maps from a real conv, half from a cheap depthwise conv (~2× fewer parameters per block) |
| **SEW residual** | Shortcut added to the **output spikes** (element-wise ADD), which removes the ~5-layer depth ceiling of plain spiking networks |

Depth has a measured optimum at 12 blocks on this data scale:

| Blocks | 5 | 8 | **12** | 18 |
|---|---|---|---|---|
| DVSGesture test | 88.0% | 89.4% | **92.0%** | 86.7% |

---

## Training recipe

| Setting | Value |
|---|---|
| Optimiser | AdamW, weight decay 5e-4 |
| Learning rate | 5e-4, cosine annealing |
| Label smoothing | 0.1 |
| Early-stopping patience | 20 |
| Max epochs | 80 |
| Batch size | 8 |
| Timesteps `T` | 16 |
| Model selection | Best **validation** epoch, evaluated once on test |

### Augmentation (training only)

| Transform | SL-Animals | DVSGesture |
|---|---|---|
| Horizontal flip | 50% | **0%** |
| Spatial shift ±8 px | 70% | 70% |
| Rotation ±12° | 30% | **0%** |
| Temporal shift ±2 steps | 50% | 50% |
| Random erase | 25% | 25% |
| CutMix | 30% of batches | 30% |

**Flip and rotation are disabled for DVSGesture.** Six of its eleven classes are
directional (*right hand wave* vs *left hand wave*, *arm clockwise* vs
*counter-clockwise*), so mirroring produces a genuinely different class while keeping the
original label. Fixing this gained **+4.9 to +8.9 points** depending on architecture.

---

## Reproducing the results

### 1. Environment

```bash
conda create -n dissertation python=3.11
conda activate dissertation
pip install "numpy<2" torch --index-url https://download.pytorch.org/whl/cu121
pip install spikingjelly tonic scipy matplotlib imageio
```

Developed on an RTX 4050 (6 GB). Peak usage ~2 GB at batch 8, T=16.

### 2. Data

Place the datasets as:

```
data/
├── SL-Animals-DVS/     # .aedat recordings + tag files
└── DVSGesture/         # downloaded automatically by tonic on first run
```

SL-Animals-DVS: request from the authors (Vasudevan et al.).
DVS128 Gesture: fetched by `tonic` automatically.

### 3. Build splits

```bash
cd code
python make_splits.py
```

Writes `data/splits/sl_animals.json` and `data/splits/dvsgesture.json`.
Splits are subject-independent for SL-Animals; DVSGesture uses the official train/test.

### 4. Train

Single runs (each writes a JSON with the full per-epoch curve to `data/results/`):

```bash
cd routing
python A1_sl_full.py --net ghostsew12 --T 16 --epochs 80 --seed 0
```

```bash
python B1_sl_hand.py --net ghostsew12 --T 16 --keep_frac 0.7 --epochs 80 --seed 0
```

```bash
python A2_dvs_full.py --net dvsnet --T 16 --epochs 80 --seed 0
```

Add `--quick` to any entry point for a fast smoke run.

### 5. Reproduce a full result with error bars

The queue scripts run whole experiment sets and skip anything already completed:

```bash
python explore_final.py
```

```bash
python explore_final.py --summary
```

The summary prints mean ± std per configuration and the A/B significance test.

### 6. Data-layer checks

```bash
python dataset.py --check
```

```bash
python dataset.py --qc-mask --cls 3
```

`--check` reports event retention for each mode; `--qc-mask` renders a full-frame versus
hand-region comparison for visual inspection.

---

## Repository layout

```
code/
  load_datasets.py      read SL AEDAT 2.0 and DVSGesture via tonic; dataset statistics
  make_splits.py        subject-independent train/val/test splits
  make_boxes.py         auto-annotated activity boxes + tracking (exploratory track)
  representations.py    events → [T, 2, 128, 128] tensor
  event_visualizer.py   render events to frames / GIF

routing/
  mark_model.py         all architectures, RecognitionDataset, train_and_eval
  dataset.py            data layer, hand-region mask, crop64 mode, QC tools
  augment.py            augmentation stack + per-dataset presets
  ensemble.py           checkpoint ensembling and test-time augmentation
  A1_sl_full.py         entry point — SL-Animals, full frame
  A2_dvs_full.py        entry point — DVSGesture, full frame
  B1_sl_hand.py         entry point — SL-Animals, hand region
  B2_dvs_hand.py        entry point — DVSGesture, hand region
  C1_sl_small.py        entry point — SL-Animals, small network
  C2_dvs_small.py       entry point — DVSGesture, small network
  explore*.py           experiment queues, one per investigation phase
  run_all.py            the original full grid
```

Each `explore*.py` documents one phase of the investigation and its reasoning in the
module docstring; together they form the methodology trail.

---

## Hand-region extraction

The hand mask ranks pixels by **rarity rather than density**. The torso produces the most
events but fires in nearly every frame, while hands sweep through few frames — so a
density-based method selects the body. The method applies an occupancy weight
`exp(-occupancy / τ)`, a temporal-continuity prior toward the previous frame's position,
relative-mass blob selection (so a weaker second hand survives), and dilation.

Validated on 12,280 frames across 7 subjects and all 4 sessions with zero failures.

The mask was originally tuned by visual inspection alone; tuning `keep_frac` against
downstream accuracy later gained **+5 points**, which is why the trade-off curve above is
reported rather than a single operating point.

---

## Limitations

- Single fixed split rather than 4-fold cross-validation, unlike most published work.
- Small test sets (171 / 264 samples).
- The hand mask is a heuristic, not human ground truth; on side-on recordings some
  forearm is included, so it is more precisely a *hand / limb-extremity* region.
- The activity boxes produced by `make_boxes.py` are **pseudo-labels** from an
  event-density method, not human annotation.
- This is not a new dataset — it is an auto-annotated extension of existing public ones.

---

## Licence

Academic use. Datasets remain under their original licences and are not redistributed.
