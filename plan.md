# Court Calibration Plan (Basketball, CVPR 2020)

## Objective
Reproduce the Sha et al. end-to-end pipeline and expose a CLI that takes a video and outputs a calibrated overlay video.

Target command:
`python main.py infer-video --input in.mp4 --output out.mp4 --ckpt checkpoints/e2e.pt --sport basketball`

## Paper Features to Implement
1. Area-based semantic segmentation (4 court regions)
2. Camera pose initialization using template dictionary + siamese retrieval
3. Homography refinement using STN (8 projective params)
4. Staged training, then end-to-end joint training
5. Evaluation with `IoU_entire` and `IoU_part` (mean + median)

## Paper-Locked Parameters (Must be explicit in configs)
1. Loss schedule and weights
- `L = alpha * Lce + beta * Lcon + (1 - alpha - beta) * Lwarp`
- Warmup and staged schedule:
  - seg warmup: 20 epochs
  - then `alpha=0.1`, `beta=0.9` (siamese on)
  - after 10 more epochs: `alpha=0.05`, `beta=0.05` (STN on, joint training)

2. Warp-loss blend (`delta`)
- Basketball default: `delta=0.8` (paper setting)
- Keep `delta` as config value, not hardcoded.

3. GMM dictionary controls
- Fixed covariance scale per camera params (paper basketball defaults):
  - pan std: `5 deg`
  - tilt std: `5 deg`
  - focal std: `1000 px`
  - camera xyz std: `15 ft`
- Off-diagonal covariance terms set to zero.
- Stopping threshold: `0.6`.
- Persist learned `K` (number of components), component means, and covariance.

4. PTZ assumptions (must be documented in code + config)
- Principal point at image center.
- Square pixels.
- No lens distortion.
- Camera roll neglected / near zero.

5. Evaluation protocol lock
- Report both `IoU_entire` and `IoU_part`, each with mean and median.
- Runtime reported as average per-frame inference time with method documented:
  - warmup frames dropped
  - hardware recorded
  - input resolution and FPS recorded

6. Reproducibility lock
- Fixed random seeds for data split, training, and template generation.
- Save exact train/val/test split manifest to disk.
- Save full config snapshot with each checkpoint.

## Step-by-Step Plan

### Step 1: CLI + Project Skeleton
- Replace `main.py` with argparse subcommands:
  - `train-seg`, `train-pose`, `train-stn`, `train-e2e`, `eval`, `infer-video`
- Create base folders: `src/`, `configs/`, `data/`, `checkpoints/`, `outputs/`
- Add config loader + logging + seed utilities.

Acceptance:
- `python main.py --help` and each subcommand help page works.

### Step 2: Court Model + Annotation Schema
- Define canonical top-view basketball court geometry.
- Define 4 area classes exactly for model output channels.
- Define annotation format per frame: image path, GT homography, split metadata.

Acceptance:
- Data validation script confirms every sample has valid homography + generated mask.

### Step 3: Semantic Label Generation
- Generate segmentation GT by warping top-view court model with GT homography.
- Add horizontal flip augmentation and metadata update logic.

Acceptance:
- Visual sanity script exports frame/mask overlays for random samples.

### Step 4: Segmentation Module (U-Net)
- Model: U-Net style encoder-decoder.
- Input: RGB frame.
- Output: 4-channel court area probabilities.
- Loss: cross-entropy (paper Eq. 3).

Acceptance:
- Segmentation predictions are stable on validation set.

### Step 5: Pose Dictionary Generation
- Estimate PTZ-like params from GT homography using paper assumptions:
  - centered principal point
  - square pixels
  - no lens distortion
  - negligible roll
- Build dictionary using GMM clustering for basketball.
- Produce templates `T_k` and homographies `H_k*`.

Acceptance:
- Template bank is saved, loadable, and visually diverse.

### Step 6: Siamese Retrieval
- Train siamese encoder with contrastive loss (paper Eq. 6).
- Retrieve nearest template via latent L2 distance.

Acceptance:
- Top-1 retrieval beats random baseline and improves through training.

### Step 7: STN Homography Refinement
- Input to STN: concatenation of selected template + predicted semantic map.
- STN predicts 8 relative homography params.
- Initialize final layer to identity bias (as in paper).
- Compose final homography from selected template homography + STN output.

Acceptance:
- Refinement improves alignment vs template-only initialization.

### Step 8: Losses and Joint Objective
- Implement Dice-based warp losses in camera and top-view spaces (Eq. 7, Eq. 8).
- Implement weighted total loss (Eq. 9).

Acceptance:
- All loss terms are finite and jointly decrease.

### Step 9: Training Schedule
- Warmup segmentation for 20 epochs.
- Enable siamese branch.
- After 10 additional epochs, enable STN branch.
- Continue full joint training to convergence.

Acceptance:
- End-to-end checkpoint outperforms modular training on IoU metrics.

### Step 10: Evaluation
- Implement:
  - `IoU_entire`
  - `IoU_part`
- Report mean + median.
- Benchmark average inference time per frame.

Acceptance:
- Script outputs paper-style result table.

### Step 11: Video Inference Pipeline
Per frame:
1. frame -> segmentation
2. segmentation -> template retrieval
3. STN -> refined homography
4. project court geometry to image
5. draw overlays (OpenCV)

Output:
- MP4 with court lines + optional region fill + confidence text.

Acceptance:
- `infer-video` command produces a playable, stable overlay video.

### Step 12: Robustness Layer
- Temporal smoothing of homography.
- Confidence gating + hold-last-good on bad frames.
- Re-acquire calibration when confidence recovers.

Acceptance:
- Reduced jitter and fewer catastrophic overlay failures.

## Recommended Repo Layout
```text
.
├── main.py
├── plan.md
├── requirements.txt
├── configs/
├── checkpoints/
├── outputs/
├── data/
└── src/
    ├── camera/
    ├── data/
    ├── eval/
    ├── infer/
    ├── losses/
    ├── models/
    ├── train/
    └── utils/
```

## CLI Contract
- `python main.py train-seg --config configs/train_basketball.yaml`
- `python main.py train-pose --config configs/train_basketball.yaml`
- `python main.py train-stn --config configs/train_basketball.yaml`
- `python main.py train-e2e --config configs/train_basketball.yaml`
- `python main.py eval --config configs/train_basketball.yaml --ckpt checkpoints/e2e.pt`
- `python main.py infer-video --input in.mp4 --output out.mp4 --ckpt checkpoints/e2e.pt --sport basketball`

## Risks
1. Domain shift in segmentation (lighting, custom floor paint, occlusions)
2. Template dictionary misses rare camera poses
3. STN instability on extreme transforms
4. Data volume may be insufficient for full generalization

## Definition of Done
1. Full train/eval/infer workflow runs from terminal.
2. End-to-end model calibrates unseen basketball videos.
3. Output MP4 shows accurate projected court geometry over time.
4. Metrics report includes `IoU_entire`, `IoU_part`, and runtime.
