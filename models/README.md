# Model resources

No weight file is tracked by Git. The Docker and command-line tools read model
resources from this directory (or from a path supplied explicitly).

## Proton dose checkpoint

Place the inference export at:

```text
models/proton/proton_film_lite_b6_epoch79.pt
```

The source training checkpoint is the final continuation `best.pt`, phase
epoch 79 (zero-based epoch 78). Its source SHA-256 is recorded in
`configs/proton_final.json`. Export it with:

```bash
python scripts/export_proton_weights.py \
  /path/to/best.pt \
  models/proton/proton_film_lite_b6_epoch79.pt
```

The export removes the two training-only deep-supervision heads. Never add the
resulting `.pt` file to Git; distribute it through a release asset, an approved
model store, or directly to an authorized user. The expected inference export
SHA-256 is
`e027b133b625b0bb89cb109a44e74f987a45da9646680a05bd8ee089550fe902`.

## Public sCT resources

The reproducibility route uses `MR_CBCT/CV_1.pt` from
[VBoussot/ImpactSynth](https://huggingface.co/VBoussot/ImpactSynth), plus its
public ImpactSeg body-mask dependency. Download both pinned revisions with:

```bash
python scripts/download_public_sct.py
```

This creates an offline-compatible Hugging Face cache below
`models/impact_synth/hub/`. These public resources are intentionally ignored
by Git. The downloader verifies:

- IMPACTSynth revision `d00d991cd6a84683bf83f7cfe53630d7bdbb73b6`,
  `MR_CBCT/CV_1.pt` SHA-256
  `02c4e57011fad5890c23b1707040454c363883ccabcfffba19ec2480503c0905`;
- ImpactSeg revision `e34d956e27347d6621b27a087d5bfe0c87c89f5c`,
  `body/Body.pt` SHA-256
  `ba1c39f84a6128e6813e17f0f193330719fdb91a4d1aa77dce0c920cff78b4d2`.

The wrapper runs the public body model directly through KonfAI, then applies
the public nearest-neighbor 1 x 1 x 3 mm resampling and radius-5 box dilation
before CV-1 synthesis. This avoids KonfAI's optional nested-app package while
preserving the documented mask operations. The public CV-1 model is a
reproducibility substitute; it is not the private DoseRAD-specific sCT
fine-tuning checkpoint described in the paper.
