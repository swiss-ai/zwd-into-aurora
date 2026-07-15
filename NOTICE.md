# NOTICE

This repository (`zwd-into-aurora`) is a **derivative work** released under the MIT License (see
[`LICENSE.txt`](LICENSE.txt)). It integrates GNSS-derived Zenith Wet Delay (ZWD) and six-hour
accumulated precipitation into the Aurora weather foundation model.

## Lineage / attribution

This code descends from two upstream projects and would not exist without them:

1. **microsoft/aurora** — <https://github.com/microsoft/aurora> (MIT License,
   Copyright (c) Microsoft Corporation).
   The [`aurora/`](aurora/) package and substantial parts of the surrounding pipeline are derived
   from Aurora. Original copyright notices and per-file license headers are retained.
   This repository is, by way of swiss-ai/ESFM (below), a fork of Aurora at commit
   [`04a5ca0`](https://github.com/microsoft/aurora/tree/04a5ca0793069ca19ff139f54a5c4f9ab29ba592).

2. **swiss-ai/ESFM** (Earth System Foundation Model) — <https://github.com/swiss-ai/ESFM>.
   The ESFM team took the Aurora commit above and developed the machinery to train / fine-tune it at
   scale. The training entrypoint (`train_fsdp.py`), the dataset and loss configuration, and much of
   the data-loading / distributed-training pipeline in this repository originate from that work.

## What is original to this repository

The contribution of this work (the ZWD + precipitation paper) is layered on top of the above:

- Integration of GNSS-derived ZWD as a new surface variable via Aurora's variable-embedding
  mechanism.
- Fine-tuning for six-hour accumulated precipitation, with and without ZWD, and the associated loss
  weighting (`λ_ZWD`, see [`loss_config.yaml`](loss_config.yaml)).
- ZWDX / MSWEP preprocessing (`create_zarr_zwdx.py`, `scripts/preprocess/`) and the precipitation
  evaluation (`utils/metrics.py`, `inference_direct.py`, `scripts/postprocess/`).

## Important disclaimer

The commit history of **this** repository does **not** reproduce the original per-commit authorship
of the upstream Aurora or ESFM code. The upstream repositories linked above are the authoritative
record of that history and authorship. The Microsoft Aurora contributors and the swiss-ai/ESFM
contributors are **not** responsible for, and do not endorse, the contents of this repository.
