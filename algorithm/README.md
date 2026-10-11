# Algorithm research sources

This directory contains actual, byte-identical project source rather than an
empty public stub. It is a **reference/prototype implementation**, not a claim
that the manuscript's complete SPECTRA-YOLO SLF experiments can be reproduced
from this release. No trained parameters, incomplete training checkpoints,
dataset images, measured FPGA results, or fabricated result logs are included.

## What is implemented

| Source | Implemented scope |
| --- | --- |
| `tools/spectra_a_prototype/model.py` | Untrained Conv-BN/RepBlock backbone, top-down FPN and optional P2 head; branch/BN fusion and decoding. |
| `tools/spectra_a_prototype/numerics.py` | 4/8-bit scalar-exponent quantization, channel-group weight quantization, integer rounding/saturation and CPU convolution references. |
| `tools/spectra_a_prototype/resource_model.py`, `tile_planner.py` | Arithmetic/liveness accounting and a DDR-backed tile-capacity plan, not measured hardware latency. |
| `tools/spectra_rewrite_training/losses.py` | Global and object-balanced feature reconstruction losses as standalone functions. |
| `tools/spectra_rewrite_training/detection.py` | FP32 detection assignment/loss with explicitly documented implementation choices. |
| `tools/spectra_experiment` | Verified-image/XML data loading, deterministic FP32 training components, CPU NMS and category-aware `pycocotools` bbox evaluation. |
| `tools/spectra_formal_experiment` | Explicit frozen-split FP32 lifecycle, code/data/checkpoint identity checks and prediction archive validation. |

The graph and model adapter expose B0/B1/F32/R32/R64 variants. They are **not** a
certified byte-equivalent YOLOv8/SLF implementation. Standalone quantization and
reconstruction losses do not constitute an integrated QAT/SLF training loop.
No SAM3 supervision, whole-network integer deployment, VisDrone evaluation
protocol, or FPGA timing/power measurement is certified by these Python files.
The evaluator explicitly rejects unsupported ignore/crowd protocols; accepting
ten class labels alone must not be described as official VisDrone evaluation.

## Environment and safe checks

Python 3.12 is the locally tested interpreter. Direct third-party imports are
PyTorch, NumPy, Pillow and pycocotools; standard-library imports need no package.
`requirements.txt` is a dependency list, not an exact environment lock. See
`VALIDATION.json` for the environment and checks actually performed for this
release. No dependencies are vendored or silently installed by the scripts.
For a fresh environment, select a platform-appropriate PyTorch build before
installing the remaining dependency list.

Run from this directory (the `algorithm` directory of the repository):

```sh
python -m compileall -q tools verify_synthetic.py
python verify_synthetic.py --suite quick
python -m tools.spectra_formal_experiment.cli --help
```

`python verify_synthetic.py --suite all` additionally selects every released
synthetic test module and may take longer. These checks use generated fixtures,
not NEU/VisDrone images. Their correctness, optimizer roundtrip, or arithmetic
checks are **not** detector accuracy, convergence, device throughput or power
evidence.

## External inputs for the frozen FP32 lifecycle

The formal CLI takes explicit `--manifest`, `--raw-root`, `--split-csv` and
`--output-dir` paths; it has no hard-coded local drive dependency. The initial
default configuration is R32, six classes, FP32, 640 pixels, batch 16, 300 epochs,
SGD (learning rate 0.01, momentum 0.937, weight decay 0.0005), seed 17,
three warmup epochs and cosine decay. These are **code defaults**, not a claim
that a completed run or the manuscript's headline numbers are bundled here.

The code-pinned NEU contract is `neu_main_v1`, with 1,260/270/270
train/tune/evaluation images and `legacy_raw_xyxy + legacy_include` annotation
policies:

- Manifest SHA256: `9d5c3cf582990e543bd2d3668a02ebd39f5b6b3ba42aa7d87fee93bf602a73c7`
- Split CSV SHA256: `f1b5c0fefbce849f91f9e296bbba1185fafc94438ea544298c5e4495f61adfa9`
- CSV columns (in order): `sample_id,source_group,split_role`; source groups are decoded-RGB SHA256 values.

Raw images/XML and the paired audited manifest are intentionally external.
The manifest contains original source-path provenance; data portability is
handled by the explicit `--raw-root` argument. Do not sanitize a frozen manifest
and then pass its old SHA256, modify the pinned hash to claim a different split
is the same experiment, or treat synthetic-fixture contracts as the formal run.
Acquire the source data through its authorized provider, and obtain the exact
paired manifest/split from the project authors before running this frozen CLI.
If those inputs are unavailable, only the synthetic checks are runnable; this
release is not full paper-result reproduction.

Inspection without optimization, after supplying the exact inputs:

```sh
python -m tools.spectra_formal_experiment.cli dry-run --manifest /path/to/manifest.json --raw-root /path/to/NEU --split-csv /path/to/split.csv --split-sha256 f1b5c0fefbce849f91f9e296bbba1185fafc94438ea544298c5e4495f61adfa9 --output-dir /path/to/new-dry-run
python -m tools.spectra_formal_experiment.cli verify-all --manifest /path/to/manifest.json --raw-root /path/to/NEU --split-csv /path/to/split.csv --split-sha256 f1b5c0fefbce849f91f9e296bbba1185fafc94438ea544298c5e4495f61adfa9 --output-dir /path/to/new-input-verification
```

Training must be explicitly requested through the CLI; no release script starts
it. Use `train --help` for its arguments. The formal schedule is 300 epochs and
every output directory must be new. Resume/evaluation require a verified paired
checkpoint and successful `run.final.json`; failed/probe/incomplete sources are
rejected. This release includes no such final trained source and makes no claim
that its paths regenerate a published AP value.

## Provenance and redistribution

`SOURCE_PROVENANCE.json` records project-relative source names, sizes and both
source/released SHA256 values. The copied source bytes have not been changed.
The wrapper and this documentation are release additions, not historical
experimental evidence. No existing license file was found in the selected
source directories. This release does not invent an author license grant;
public visibility alone does not grant unrestricted reuse. Third-party runtime
packages retain their own licenses and are not redistributed here.
