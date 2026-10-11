# Data and code availability routes

| Material | Access route in this release | Status |
| --- | --- | --- |
| Research-reference algorithms and tests | `algorithm/` | Included, with source hashes and synthetic validation. Not final trained SLF checkpoints. |
| HLS/PS/Vivado integration sources | `hardware/` | Included, with source hashes and integration limitations. |
| Existing independent implementation report values | `hardware/INDEPENDENT_IMPLEMENTATION_SUMMARY.json` | Derived non-identifying summary; raw report hashes retained. |
| NEU-DET | Original provider and dataset reference cited in the manuscript | Not redistributed. |
| VisDrone-DET | https://github.com/VisDrone/VisDrone-Dataset and https://aiskyeye.com/ | Not redistributed. |
| Final model weights and complete matched evaluation/calibration records | No public files in this release | Not included. No claim of complete result reproduction. |
| Full board timing/power measurements | No public files in this release | Not included. On-chip power estimation is not board power. |

The initial public scaffold remains in Git history. The current source package uses the new `algorithm/` and `hardware/` directories. Legacy root-level `hls/`, `ps/` and `vivado/` files from the first commit are placeholders only, not the current implementation or build entry points.
