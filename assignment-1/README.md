# AI401 Assignment 1: soil grain-size distribution from photos

Kaggle: [Predicting Soil Grain Size Distributions from Images](https://www.kaggle.com/competitions/soil-grain-size-from-photos).
From smartphone photos of a soil sample, predict the cumulative percentage passing 11 sieves (0.002-200 mm).
The same problem is solved with CNNs, with Vision Transformers and with self-supervised encoders, on one tiling
pipeline and one set of folds.

<!-- RESULTS -->

## Part 0. Data, metric, validation

### EDA (`notebooks/assignment1.ipynb`, `results/eda/`)

| | train | test |
|---|---|---|
| physical samples | 24 | 10 (public LB = 3 fixed samples) |
| photos | 127 (3-8 per sample) | 35 |
| cameras | Motorola Edge (69), Samsung A52 (55), Motorola Edge 60 fusion (3) | iPhone 14 (14), iPhone 16 (21) |
| stored resolution | 1600 x 720 and 1599 x 1200 (messenger-compressed), 4096 x 2304 | 4032 x 3024, 5712 x 4284 (native) |
| pixels per mm | 4.60, 4.55, 12.5 | 13.9, 19.5 |
| field of view | about 350 x 160-260 mm | about 290 x 220 mm |

- **Scale.** `ppm_updated.csv` gives ppm at each camera's native resolution. Most training photos are stored 2.5x
  (Motorola) and 5.8x (Samsung) smaller, so the effective ppm is the table value times the size ratio. A
  pipeline that takes the table value at face value is off by a factor of 2.5-5.8, i.e. 40-76 EMD from scale alone.
  The host says every photo is taken from 21 cm. As a check of the Samsung scale, `scripts/calibrate.py` estimates
  it without labels: the radially averaged texture spectrum (in cycles per mm) of the Samsung photos is matched to
  the Motorola photos of the same 21 soils. It gives 4.8 px/mm against the table's 4.55, the same +5% bias that
  the method shows on a Motorola-vs-Motorola self-check (`results/calibration/samsung_ppm.png`).
- **Domain shift.** The training photos are low resolution and recompressed. The test photos come from two
  phones never seen in training, at 3-4x the pixel density.
- **Curves are bimodal.** 9 fine soils (D50 0.02-0.1 mm, 37-90% finer than 0.063 mm), 11 gravels (D50 2.5-6 mm, 52-77%
  coarser than 2 mm), 4 sands in between. At the common density of 0.21 mm per pixel, clay, silt and fine sand
  are below one pixel, so the left half of every curve has to be inferred from colour and surface appearance.
- **Backgrounds.** Samsung photos show the whole tray with its rim and an LED strip; Motorola and iPhone photos show
  a tray edge on one side. Tiles are taken from the central 70% of the frame only.

### Metric

EMD = sum_{i=1..10} |F_i - F^_i| (log10 x_{i+1} - log10 x_i), in percent x decades, range [0, 500]
(`src/soilgs/metric.py`). It is the Wasserstein-1 distance between the two grain-size distributions on a
log-diameter axis, i.e. the area between the two grading curves on the usual semi-log plot.

- **Strengths:** physically meaningful, every decade of grain size counts the same, linear in the error (robust), and
  usable directly as a training loss for a cumulative target.
- **Weaknesses / edge cases:** (1) a scale error moves the whole curve along log d; a factor-2 error alone costs
  100 x log10(2) = **30.1 EMD** (checked numerically in the notebook), more than the gap between a good and a naive
  model. (2) The clay/silt part (0.002-0.063 mm) is 1.5 of the 5 decades, but it is invisible in a photo.
  (3) The grid is coarse (about 0.5 decade per interval), so shifts inside one interval are invisible. (4) The 200 mm
  point is fixed at 100, and for fine soils the whole gravel range adds almost nothing.
- **Complementary metric:** errors of D10 / D30 / D50 / D60 in decades (0.30 = factor 2), reported for every run
  (`D50_err_dec`). Engineers read these numbers (Cu, Cc), and they separate "right shape, wrong position" from
  "wrong shape".
- **Alternative target:** the 11 bin masses (fraction between consecutive sieves) instead of the 10 cumulative
  values. A softmax over the bins followed by a cumulative sum always gives a valid curve (non-negative increments,
  ends at 100), and the EMD becomes a differentiable loss. All neural models here use this head. Ridge/k-NN
  predictions on cumulative values are repaired with clipping plus isotonic regression (PAV), which is the
  least-squares projection onto valid curves (closer to the raw prediction than a running maximum).

### Validation and input pipeline

- **Folds:** 6 group folds over physical samples (`results/data/folds.csv`), stratified by D50 (each block of 6
  samples sorted by D50 is dealt to the 6 folds), so every fold holds fine and coarse soils. All photos of a sample
  are on the same side. Predictions are averaged tile -> photo -> sample before scoring. CV = mean ± std of the
  per-fold mean EMD. Every experiment uses the same folds.
- **Tiles:** each photo is resampled with its ppm to 4.67 px/mm, the density of the training photos. The training
  photos are therefore never upsampled, and the test photos are downsampled to the same density. 48 x 48 mm tiles
  (224 px, 50% overlap, central 70% of the frame, at most 48 per photo) are cached once as uint8 arrays and kept on
  the GPU. 48 mm is about 10 medium-gravel grains across and still resolves coarse sand (2 mm = 9 px). 24 mm and
  96 mm caches are kept for the ablation. Extra caches for the robustness study rebuild the training photos with a
  wrong ppm (x0.5 ... x2) or at reduced pixel density with the correct ppm.
- **Augmentation is scale-safe:** dihedral flips/rotations and mild colour jitter only. Random-resized crops or zoom
  change mm-per-pixel, which changes the label.

## Repository

| path | what |
|---|---|
| `config.yaml` | data/tiling settings and every reported experiment (defaults + overrides) |
| `src/soilgs/metric.py` | competition EMD, D-value errors, curve repair (clip + isotonic PAV) |
| `src/soilgs/tiles.py`, `data.py` | photo index and ppm, calibrated tiling, tile caches, group folds |
| `src/soilgs/models.py` | timm backbones, from-scratch ViT, softmax-over-bins curve head, EMD loss, LoRA, BN/GN handling, scale-safe GPU augmentation |
| `src/soilgs/train.py` | CV training of one experiment: OOF + test predictions, controls, scale robustness, CKA, compute log |
| `src/soilgs/frozen.py` | frozen SSL features: ridge (5 feature sets), weighted k-NN, linear probe |
| `src/soilgs/dino.py` | continued DINO self-distillation (domain adaptation) with scale-safe multi-crop |
| `scripts/prepare.py` | index, EDA, folds, all tile caches (`results/data/folds.csv` is the fold file) |
| `scripts/calibrate.py` | label-free check of a camera's ppm (texture spectra of the same soils) |
| `scripts/run.py` | runs experiments from the config (`--family cnn|vit|ssl`, `--only a,b`) |
| `scripts/baselines.py` | image-free baselines and shortcut controls |
| `scripts/interpret.py` | Grad-CAM / attention rollout maps for the best model of each family |
| `scripts/ensemble.py`, `report.py`, `submit.py` | OOF error correlation and ensembles; tables and figures; Kaggle submission |
| `scripts/predict.py` | inference: photo(s) + pixels-per-mm -> valid curve |
| `notebooks/assignment1.ipynb` | EDA, metric analysis and all result tables/figures (reads `results/` only) |
| `HYPOTHESES.md` | hypotheses written before each block of experiments, outcomes after |
| `results/` | per-experiment config, OOF, test predictions, metrics, compute (`results/exps/<name>/`), tables, figures |
| `remote/` | RunPod helpers (`runpod_ctl.py`, `pod.sh`, `setup.sh`, `run_all.sh`) |

Reproduce (one GPU, about 2 GPU-hours):

```bash
uv sync --frozen
uv run kaggle competitions download -c soil-grain-size-from-photos -p data/raw   # then unzip
bash remote/run_all.sh          # prepare -> baselines -> cnn -> vit -> ssl -> interpret -> ensemble -> report
uv run python scripts/predict.py --ppm 13.942 photo1.jpg photo2.jpg --exps <experiment>
```

Software: Python 3.12, torch 2.14.1+cu126, timm 1.0.30 (all pinned in `uv.lock`), bf16 autocast. Pretrained weights
(timm tags): `resnet18.tv_in1k`, `resnet50.a1_in1k`, `efficientnet_b0.ra_in1k`, `convnext_tiny.fb_in22k_ft_in1k`,
`deit_small_patch16_224.fb_in1k`, `swin_tiny_patch4_window7_224.ms_in22k_ft_in1k`, `maxvit_tiny_tf_224.in1k`,
`vit_small_patch14_dinov2.lvd142m`, `vit_small_patch16_dinov3.lvd1689m`. No external labelled soil data, no manual
labels. Only training photos are used for self-supervised adaptation (test photos are never used for training).
Nothing depends on test file names or ids except reading them.
