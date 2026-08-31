# DoseRAD2026

Reference code for our final **proton CT and proton MRI** entries to the
[DoseRAD2026 Grand Challenge](https://doserad2026.grand-challenge.org/). The
method predicts one three-dimensional dose contribution per proton beamlet in
a beam's-eye-view (BEV) frame and maps it back to the native patient grid.

This is a compact scientific release: it contains the final dose architecture,
preprocessing geometry, loss, training loop, inference runtime, Docker build
context, and paper sources. It intentionally excludes challenge images,
caches, experiment logs, abandoned variants, private medical data, Docker
archives, and weight binaries.

> **Scope.** This repository describes the final ProtonFiLM-Lite route. Photon
> experiments and the abandoned three-checkpoint ensembles are not presented
> as the submitted method.

## Method overview

For each ray, CT anatomy is interpolated with a cubic B-spline into a
`288 x 64 x 64` `(depth, lateral, superior)` BEV grid at `2 x 1 x 1 mm`. The
first depth plane is 320 mm upstream from the ray target. Intensities are
clipped to physical HU bounds and mapped to `[0,1]`; the beamlet energy is
mapped from 31.729--200.7966 MeV to `[0,1]`.

The dose model is a six-level, base-width-6 residual 3-D U-Net with GroupNorm,
SiLU, Softplus output, and restrained deep FiLM conditioning. Its deployed
form has 3,094,511 trainable parameters. Two decoder heads provide deep
supervision during training only. The loss is the equal-weight sum of the
challenge-aligned masked beam MAE and integrated depth-dose (IDD) RMSE.

For MRI, an sCT is produced once per source volume before the same dose route.
The public reproducibility path uses the 2.5-D UNet++/ResNet-34
`MR_CBCT/CV_1.pt` checkpoint from
[IMPACTSynth](https://huggingface.co/VBoussot/ImpactSynth), executed with
[KonfAI](https://github.com/vboussot/KonfAI). This public checkpoint replaces
our unavailable private DoseRAD-specific sCT fine-tuning, so public MRI output
is method-reproducible but not bit-identical to the challenge submission.

The full method diagram is in [the paper](paper/paper.pdf).

## Repository structure

```text
configs/                 final proton geometry and checkpoint provenance
doserad2026/             network, geometry, cache, loss and runtime modules
docker/                  complete Grand Challenge invoke-API build context
models/                  model placement and download instructions (no weights)
paper/                   clean LNCS manuscript sources and figures
scripts/                 cache, model download/export and inference utilities
tests/                   numerical and smoke tests
training/                standalone final proton trainer
```

## Installation

Python 3.10 was used for the recorded training. For a CUDA 12.6 workstation:

```bash
git clone https://github.com/HEMONCedric/DoseRAD2026.git
cd DoseRAD2026
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install --index-url https://download.pytorch.org/whl/cu126 torch==2.7.1
python -m pip install -e ".[dev]"
```

Install the additional public sCT runtime only when reproducing proton MRI:

```bash
python -m pip install -e ".[sct,dev]"
```

The dependency list is deliberately small and pinned. The Docker uses the
official PyTorch 2.9.1/CUDA 12.6 runtime image, matching the documented
submission environment more closely than the training environment.

## Dataset

Challenge data are not distributed here. The raw training converter expects:

```text
/path/to/proton/training/
├── beam_parameters.json
├── 1ABB001/
│   ├── 1ABB001.json
│   ├── image/
│   │   ├── ct.mha
│   │   └── mr.mha
│   └── dose/
│       ├── Dose_B0_R0_L0.mha
│       └── ...
└── ...
```

Patient identifiers in `configs/proton_final.json` are anonymized challenge
case IDs and define the six-patient model-selection split. Generate the final
cache with:

```bash
python scripts/prepare_proton_cache.py \
  --raw-root /path/to/proton/training \
  --output /path/to/proton_bev_cache/v1_288x64x64_s2x1x1_e31p729_200p7966
```

The result contains `train/<patient>/*.npz` and
`validation/<patient>/*.npz`. CT and dose are float16 after normalization and
global dose scaling; normalized energy remains float32. Use
`--limit-patients 1 --limit-samples-per-patient 2` for a smoke test.

The default command creates the **final-phase** 69/6 split. To reproduce the
preceding 65/10 base phase, prepare a second cache by passing the six IDs from
`configs/proton_final.json` plus `1ABB135 1ABB149 1THB037 1THB122` to
`--validation-patients`. The four additional cases move from validation to
training only for the final continuation.

## Training

The original 400-epoch phase uses AdamW, weights-only decay, effective batch
four, BF16, clipping at 1.0, and cosine decay:

```bash
CUDA_VISIBLE_DEVICES=0 python training/train_proton.py \
  --cache-dir /path/to/proton_bev_cache/base_65train_10validation \
  --out-dir results/proton_base \
  --device cuda:0 \
  --epochs 400 \
  --batch-size 2 \
  --accumulation-steps 2 \
  --lr 1e-3 \
  --min-lr 2.5e-5 \
  --weight-decay 5e-4 \
  --grad-clip 1.0 \
  --amp bf16 \
  --checkpoint-every 3
```

The final 80-epoch continuation retains the AdamW moments, scaler, and RNG
state but starts a fresh cosine schedule. It adds the paired lateral flip:

```bash
CUDA_VISIBLE_DEVICES=0 python training/train_proton.py \
  --cache-dir /path/to/proton_bev_cache/v1_288x64x64_s2x1x1_e31p729_200p7966 \
  --out-dir results/proton_final_cont80 \
  --device cuda:0 \
  --continue-phase results/proton_base/last.pt \
  --epochs 80 \
  --batch-size 2 \
  --accumulation-steps 2 \
  --lr 1e-4 \
  --min-lr 4e-5 \
  --weight-decay 5e-4 \
  --lateral-flip-prob 0.25 \
  --grad-clip 1.0 \
  --amp bf16 \
  --checkpoint-every 3
```

`best.pt` is updated on every validation improvement, `last.pt` after every
epoch, and `epoch_003.pt`, `epoch_006.pt`, etc. every three epochs. TensorBoard
events are written below `<out-dir>/tensorboard`.

## Checkpoints

Git ignores all weight formats. See [models/README.md](models/README.md) for
exact paths, provenance, and export commands.

The final dose source checkpoint is phase epoch 79 (stored epoch 78), SHA-256
`eb0c5451a9eacd373fe8f54fd05a9cc33098891f2ee9083d2761232319d65e0b`.
Export its inference-only state to:

```text
models/proton/proton_film_lite_b6_epoch79.pt
```

The expected inference export SHA-256 is
`e027b133b625b0bb89cb109a44e74f987a45da9646680a05bd8ee089550fe902`.

For public sCT, download only the pinned IMPACTSynth CV-1 configuration and
weights plus the public body-mask dependency:

```bash
python scripts/download_public_sct.py
```

The downloader creates an offline-compatible Hugging Face cache and never
requires a token for these public repositories. It pins IMPACTSynth revision
`d00d991cd6a84683bf83f7cfe53630d7bdbb73b6` and ImpactSeg revision
`e34d956e27347d6621b27a087d5bfe0c87c89f5c`; the selected public CV-1 weight
has SHA-256
`02c4e57011fad5890c23b1707040454c363883ccabcfffba19ec2480503c0905`.

## Inference outside Docker

Run the dose network on one cached BEV sample:

```bash
python scripts/predict_proton_bev.py \
  /path/to/sample.npz \
  models/proton/proton_film_lite_b6_epoch79.pt \
  outputs/prediction_gy.npy \
  --device cuda:0
```

Generate a public sCT from one MR volume:

```bash
python scripts/predict_sct_public.py \
  input_mr.mha output_sct.mha \
  --model-cache models/impact_synth \
  --gpu 0
```

Native patient-space challenge inference is exposed by
`doserad2026.runtime.ProtonRuntime` and is the same entry point used by the
container.

## Docker / Grand Challenge

No pre-built image or Docker archive is published. Build it locally:

```bash
./docker/build.sh
```

Run a proton-CT job with model resources mounted read-only:

```bash
mkdir -p outputs/proton-ct
./docker/run.sh \
  proton-ct \
  /absolute/path/to/grand-challenge-input \
  "$PWD/outputs/proton-ct" \
  "$PWD/models"
```

The container implements `GET /health` and `POST /invoke` on port 4743,
reads `/input`, and writes ten
`/output/images/stacked-radiation-dose-map-{1..10}/output.mha` slots. MRI uses
the same command with `proton-mri` after `download_public_sct.py`. Hugging Face
resolution is forced into offline mode at runtime; all model resources must
already be mounted below `/opt/ml/model`. The local run helper leaves Docker's
port networking enabled so the invoke API remains reachable and allocates 2
GiB of shared memory for the KonfAI MRI stage.

Detailed model layout, build behavior, and `curl` examples are in
[docker/README.md](docker/README.md). Input/output naming follows the
[official DoseRAD2026 submission instructions](https://doserad2026.grand-challenge.org/submission-instructions/).

## Reproducibility and recorded results

- Seeds: model/data seed 2026; the six-patient final validation split is fixed.
- Final dose hardware: one NVIDIA RTX 6000 Ada (48 GiB), Intel Xeon W5-3425,
  502 GiB host RAM.
- Final cached-BEV selection loss: 0.0108966 (masked MAE 0.0063354, IDD
  0.0045612) on 6,480 beamlets from six patients.
- These are model-selection surrogates, not an independent patient-space test
  of the final checkpoint. The paper labels older native-space numbers as
  historical rather than silently mixing protocols.
- The checked-in runtime is a transparent, numerically aligned reference that
  uses SciPy cubic transforms and uncompressed streaming MetaImage output. The
  paper's reported runtime was measured with the submission build's CuPy
  transforms and compressed writer; do not attribute those timings to this
  portable public implementation without a new benchmark.

Run the local checks with:

```bash
pytest -q
ruff check doserad2026 scripts training tests
```

## Paper and citation

The LNCS manuscript and its figures are under `paper/`. Until final bibliographic
metadata are assigned, cite this repository via `CITATION.cff`. IMPACTSynth and
KonfAI must also be cited when using the public MRI route.

## License and data policy

Code in this release is MIT licensed; inherited attribution is retained in
`LICENSE` and `THIRD_PARTY_NOTICES.md`. IMPACTSynth, ImpactSeg, and KonfAI
retain their own licenses and terms. The DoseRAD2026 data remain subject to
the challenge terms and are not included. Do not commit raw images, dose maps,
caches, model weights, predictions, or Docker exports; `.gitignore` enforces
these exclusions.
