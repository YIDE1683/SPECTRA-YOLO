# SPECTRA-YOLO engineering reference package

Source and verification utilities for the SPECTRA-YOLO project, accompanying the manuscript **SPECTRA-YOLO: Software-Hardware Co-Designed YOLOv8 for FPGA Edge Inspection**. This release replaces the initial public scaffold with actual local research and hardware reference sources. It is not a release of trained final-detector weights or a complete reproduction of all manuscript experiments.

## Contents

| Directory | Included | Verification boundary |
| --- | --- | --- |
| `algorithm/` | SpectraA prototype, graph/numeric/planning utilities, detection loss and data adapters, FP32 training/evaluation and formal-run lifecycle, synthetic tests | The existing research prototype is not the trained final SLF-YOLOv8 model. Synthetic tests do not establish NEU-DET or VisDrone accuracy. |
| `hardware/` | HLS kernels, processing-system C sources, Vivado templates, separate implementation scripts | Reference integration sources; no board programming or end-to-end deployment is validated by this release. |
| `hardware/INDEPENDENT_IMPLEMENTATION_SUMMARY.json` | Derived summary and SHA256 provenance for existing implementation reports | Separate 177.778 MHz implementation, 179 DSPs, 3.551 W estimated on-chip power; not measured board power or full-network throughput. |
| `docs/` | Availability routes, rights notice, scope and validation guide | Explicit inclusion/exclusion and evidence mapping. |

Start with [algorithm/README.md](algorithm/README.md) and [hardware/README.md](hardware/README.md). Both describe existing limitations and commands. No hardware command is required for the synthetic algorithm checks.

## Safe local verification

With the dependencies in `algorithm/requirements.txt` already available:

```text
cd algorithm
python -B verify_synthetic.py
```

See `algorithm/VALIDATION.json` and `hardware/RELEASE_CHECKS.json` for the checks actually performed. See `RELEASE_MANIFEST.json` for file SHA256 values. Do not interpret a passing source or contract test as a completed training run, accuracy validation, timing closure for another source configuration, or physical acceptance.

## Data and experiment records

Dataset images, trained weights, unfinished checkpoints, raw training and board-measurement logs, generated vendor IP, SDK headers, bitstreams, XSA/DCP files and proprietary licence material are excluded. NEU-DET and VisDrone must be obtained through their official providers. There is no archived DOI assigned to this code release. Pin the Git commit for the exact version used.

The available code is reference/prototype support. Complete final-model checkpoints, matching seed-level evaluation records, calibration provenance and full board timing/power records are not included; this repository therefore does not independently validate every numerical claim in the manuscript.

## Rights

No new blanket open-source licence is granted in this release. Copyright and third-party rights are retained; see [docs/RIGHTS_NOTICE.md](docs/RIGHTS_NOTICE.md). External Python packages and AMD tools remain subject to their own licences and are not bundled. Publishing source for inspection does not by itself grant unrestricted redistribution or use.
