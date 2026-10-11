# Hardware reference sources

This subset contains the project's actual HLS convolution/line-buffer/INT4
packing sources, Linux PS preprocessing/NMS/pipeline/UIO sources, and Vivado
integration scripts. It replaces neither the complete detector deployment nor
board-level performance evidence. Historical comments that quote manuscript
values are retained for source fidelity; the boundaries below govern this
release.

## Target and dependencies

- Target part: `xczu9eg-ffvb1156-2-e`, ZCU102 / Cortex-A53 Linux.
- HLS: AMD/Xilinx `ap_int.h`, `hls_stream.h`, Vitis HLS and installed vendor IP.
- The independent implementation records were produced with Vivado 2025.2
  (win64, build 6299465); this release has not rerun that toolchain.
- PS: AArch64 GCC/sysroot, POSIX threads, libm, libjpeg and libpng development
  headers/libraries. The present decoder implementation uses JPEG; a PNG or
  live-camera input path is not established by the source comments alone.
- No generated IP, encrypted vendor library, BSP, device tree, weights, physical
  DMA allocation, bitstream, XSA, or runtime Linux image is redistributed.
  Obtain vendor tools/IP through their own licences.

## What the code currently implements

`hls/` includes the 3-by-3 streaming convolution top, line buffer, packed INT4
three-product helper and exponent configuration. It is a kernel implementation,
not an executable representation of the complete YOLO graph. In particular,
the `use_int4 == 0` weight loader still narrows values to `Weight4`, and the
activation feeder uses `Act4`. Do not use the flag as evidence of an independent
8-bit arithmetic path. Calibration-table comments do not replace reproducible
calibration records or prove a checkpoint's accuracy.

`ps/` contains useful preprocessing, NMS and scheduling/UIO reference code.
Before device execution, it needs a matched exported model graph and tensor
layout, weight/bias loading, allocated DMA buffers and cache coherency,
generated HLS register-map verification, bounded waits, and reviewed pipeline
synchronisation. The PS layer list is not a complete graph exporter. The
published hand-written register offsets must not be assumed compatible with
an arbitrary HLS export. No board execution is claimed here.

## Distinct clock / implementation scopes

1. `vivado/` is the original integration template. Its XDC/comments request a
   200 MHz compute clock, but its Tcl connects accelerator `ap_clk` to
   `pl_clk2` (nominally 100 MHz). The multiple clock constraints and broad
   false paths need design-specific validation. This template is not a
   timing-closed 200 MHz board result and is not the recommended build entry.
2. `independent_flow/scripts/` preserves the subsequently checked single-clock
   flow, with only paths made portable. Both its HLS constraint and BD request
   a 200 MHz target. Historical **routed** reports instead show **177.778 MHz
   (5.625 ns)**, **179 DSP blocks**, and **3.551 W estimated on-chip power**
   (2.822 W dynamic + 0.729 W static). Request and achieved clock are different
   quantities. These reports do not establish detector FPS, total board power,
   HLS latency/II, or an 8-bit execution result.

`INDEPENDENT_IMPLEMENTATION_SUMMARY.json` records those extracted values and
SHA-256 digests of the original reports, without redistributing machine paths
or private logs. `SOURCE_PROVENANCE.json` maps each released file to its
project-relative source and records source/release digests and adaptations.

## Build commands (not executed for this release)

From `hardware/`, with the matching vendor installation and licences:

```sh
vitis_hls -f independent_flow/scripts/run_hls.tcl
vivado -mode batch -source independent_flow/scripts/run_vivado_impl.tcl -tclargs 8
```

The scripts regenerate build products under `independent_flow/build` and
`independent_flow/exports`; the implementation script can create a bitstream
but does not program a device. The HLS flow uses reset/overwrite semantics
only in its generated project directory. Preserve previous generated work
elsewhere before running it. Vendor/IP-version differences can change results.

For PS cross-compilation, from `hardware/ps/`:

```sh
make CROSS=aarch64-linux-gnu-
```

The released Makefile removes AArch32-only `-mfpu`/`-mfloat-abi` flags from the
AArch64 target. On a non-ARM Linux host, a compile-only diagnostic can override
the target flags with `make CFLAGS='-O2 -Wall -Wextra'`; it is not a hardware
inference test. Deployment and SSH targets require explicit destinations via
`DEPLOY_DIR` or `ZCU102_HOST=user@host`. Do not invoke them or the PS inference
binary without a separately authorised, fully specified device operation.

## Release checks

All 24 selected source files have tracked hashes; local quoted includes resolve.
See `RELEASE_CHECKS.json`. No compiler was available on the Windows PATH, and
the existing WSL setup could not start because its configured custom kernel
was missing (`WSL_E_CUSTOM_KERNEL_NOT_FOUND`). No system repair, dependency
installation, HLS synthesis, Vivado implementation, FPGA programming, board
measurement, or physical-device access was performed for this release.

These are genuine engineering sources with explicit integration boundaries,
not a claim that all manuscript accuracy or throughput numbers are independently
reproduced by this subset.
