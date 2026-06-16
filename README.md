# SPECTRA-YOLO Public Engineering Package

This repository provides the SPECTRA-YOLO engineering package for the manuscript:

`SPECTRA-YOLO: Software-Hardware Co-Designed YOLOv8 for FPGA Edge Inspection`

The package is intentionally limited for public citation and editorial inspection. It provides the expected project layout, tool categories, and a minimal host-side build check. Detailed kernels, scheduling logic, calibration files, generated hardware projects, datasets, model weights, board images, and measurement logs are not included in this public package.

## Contents

```text
docs/
  REPRODUCIBILITY_SCOPE.md

hls/
  public_accelerator_stub.h
  public_accelerator_stub.cpp

ps/
  main.c
  Makefile

vivado/
  constraints/zcu102_public_stub.xdc
  scripts/create_project_stub.tcl
```

## Build Check

The public host scaffold can be compiled with:

```bash
cd ps
make
```

For a cross compiler:

```bash
cd ps
make CROSS=aarch64-linux-gnu-
```

This build only verifies the public scaffold. It is not the complete board implementation.

## Access Boundary

The complete engineering archive is retained by the authors. Additional materials can be provided to editors and reviewers through the corresponding author when required by the submission process.
