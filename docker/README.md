# Proton container

The repository publishes the complete build context, not a pre-built image.
The image implements Grand Challenge's long-lived `invoke` API and supports
`TASK=proton-ct` and `TASK=proton-mri`.

## Resources mounted at runtime

```text
models/
├── proton/
│   └── proton_film_lite_b6_epoch79.pt
└── impact_synth/                    # required only by proton-mri
    └── hub/                          # produced by download_public_sct.py
```

The build deliberately excludes `models/`. This mirrors Grand Challenge model
resources and prevents weight files from entering either Git or an image
layer.

## Build

From the repository root:

```bash
python scripts/download_public_sct.py       # proton-mri only
./docker/build.sh
```

## Run locally

```bash
mkdir -p outputs/proton-ct
./docker/run.sh proton-ct /absolute/path/to/input outputs/proton-ct "$PWD/models"
```

The server listens on port 4743. In a second terminal:

```bash
curl --fail http://127.0.0.1:4743/health
curl --fail --request POST http://127.0.0.1:4743/invoke
```

The helper forces Hugging Face into offline mode, but retains the default
Docker network so that port 4743 is reachable. No model is downloaded during
container execution. It allocates 2 GiB of shared memory because KonfAI's
PyTorch loading path can otherwise exceed Docker's 64 MiB default and exit
with `SIGBUS`.

Input and output slugs follow the official DoseRAD2026 example. The input must
contain `stacked-proton-beam-level-metadata.json` and one or more source image
directories under `images/`. Ten output slots are always created.

For MRI, the public CV-1 IMPACTSynth model creates one sCT per source image.
The wrapper first executes the pinned public ImpactSeg body checkpoint through
KonfAI, then performs the public mask resampling and dilation before synthesis.
This public checkpoint substitutes for the unavailable private DoseRAD-specific
sCT fine-tuning. Therefore the public MRI route is method-reproducible but is
not bit-identical to the challenge submission described in the paper.

The public reference runtime intentionally favors auditability and portability:
it uses SciPy cubic transforms and writes uncompressed streaming MetaImages.
The paper's runtime table was measured from the optimized submission build
(CuPy transforms and compressed output) and must not be treated as a benchmark
of this public container without remeasurement.
