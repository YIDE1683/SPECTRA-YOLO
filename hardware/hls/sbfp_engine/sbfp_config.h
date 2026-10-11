/*=============================================================================
 * sbfp_config.h  –  SBFP Compile-Time Shared Exponent Table
 *
 * Paper: Section 4.2, Theorem 2, Corollary 1
 *
 * These constants are determined OFFLINE by KL-divergence minimisation
 * over 1,000 calibration images (Eq. 20-21) and stored here as compile-time
 * constants baked into the FPGA configuration bitstream.
 *
 * Corollary 1: Because E_shared is fixed at compile time, the three hardware
 * modules required by dynamic BFP are ELIMINATED:
 *   (a) per-block max-finding adder tree
 *   (b) normalisation barrel-shifter
 *   (c) exponent register update logic
 * → LUT reduction: 227,145 → 119,732  (-47.3%)  [Table 5]
 *
 * Format (Eq. 22): [sign(1b) | mantissa(n-1 bits) | E_shared=CONST]
 * Scale factor:    2^E_shared  (applied at dequantisation output only)
 *
 * How to re-calibrate:
 *   Run:  python SPECTRALproject/experiments/calibrate_sbfp.py \
 *              --model runs/slf_neu_det/weights/best.pt \
 *              --calib-data datasets/NEU-DET --n-bits 8 \
 *              --out zynq/hls/sbfp_engine/sbfp_config.h
 *=============================================================================*/
#ifndef SBFP_CONFIG_H
#define SBFP_CONFIG_H

/* Total number of quantised layers in SLF-YOLOv8 PL implementation         */
#define SBFP_N_LAYERS   23

/*
 * Optimal shared exponent E* per layer (Theorem 2, Eq. 20).
 * Index = layer index (0 = first CBS, 22 = P5 C2f head).
 * Values below are calibrated on NEU-DET with 1,000 calibration images.
 * Larger exponent → larger representable range (coarser quantisation).
 */
static const int SBFP_EXPONENTS_8B[SBFP_N_LAYERS] = {
    /* Layer  0 – CBS Stem  (P1)         */ -6,
    /* Layer  1 – CBS P2               */  -5,
    /* Layer  2 – C2f P2               */  -5,
    /* Layer  3 – CBAM P2              */  -5,
    /* Layer  4 – CBS P3               */  -4,
    /* Layer  5 – C2f P3 (×6)          */  -4,
    /* Layer  6 – CBAM P3              */  -4,
    /* Layer  7 – CBS P4               */  -3,
    /* Layer  8 – C2f P4 (×6)          */  -3,
    /* Layer  9 – CBAM P4              */  -3,
    /* Layer 10 – CBS P5               */  -2,
    /* Layer 11 – C2f P5 (×3)          */  -2,
    /* Layer 12 – CBAM P5              */  -2,
    /* Layer 13 – SPPF                 */  -3,
    /* Layer 14 – BiFPN node (P4↑P5)   */  -3,
    /* Layer 15 – BiFPN node (P3↑P4)   */  -4,
    /* Layer 16 – BiFPN node (P2↑P3)   */  -5,
    /* Layer 17 – BiFPN node (P3↓P2)   */  -4,
    /* Layer 18 – BiFPN node (P4↓P3)   */  -3,
    /* Layer 19 – BiFPN node (P5↓P4)   */  -2,
    /* Layer 20 – Detect head P2       */  -4,
    /* Layer 21 – Detect head P3       */  -3,
    /* Layer 22 – Detect head P5       */  -2,
};

/* INT4 mode exponents (SBFP-4b, §4.3) – wider range per layer              */
static const int SBFP_EXPONENTS_4B[SBFP_N_LAYERS] = {
    -4, -4, -3, -3, -3, -3, -3, -2, -2, -2,
    -1, -1, -1, -2, -2, -3, -4, -3, -2, -1,
    -3, -2, -1,
};

/*
 * Scale factors: scale[l] = 2^E*[l]
 * Pre-computed as fixed-point reciprocals for dequantisation.
 * At inference: output_fp32 = mantissa_int × (1 << E*)
 * On FPGA: implemented as arithmetic left-shift by |E*| (barrel-shifter NOT needed
 * because shift amount is a compile-time constant → synthesiser emits a wire remap).
 */
#define SBFP_SCALE_8B(layer_idx)   (1 << (-(SBFP_EXPONENTS_8B[layer_idx])))
#define SBFP_SCALE_4B(layer_idx)   (1 << (-(SBFP_EXPONENTS_4B[layer_idx])))

#endif /* SBFP_CONFIG_H */
