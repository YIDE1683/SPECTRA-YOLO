# Reproducibility scope

The first public commit contained placeholders only. The present release adds
actual reference sources under `algorithm/` and `hardware/`. Use their README
files and validation records rather than the legacy root stub paths.

The released tests verify synthetic software contracts. Historical Vivado
values are separately identified, with report hashes. Final SLF checkpoints,
complete NEU/VisDrone evaluation records, whole-network quantized deployment,
and board-level throughput/power evidence are not supplied or newly verified.

The package supports inspection of implementation ideas and selected software
behaviours, not independent reproduction of every paper result. Dataset files,
licensed vendor dependencies and raw private logs are excluded. Rights and
access routes are in `RIGHTS_NOTICE.md` and `DATA_AND_CODE_SCOPE.md`.
