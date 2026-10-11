/*=============================================================================
 * sbfp_engine.h  –  Static Block Floating-Point Quantisation Engine
 *
 * Paper: Section 4.2, Theorem 2, Eq. 18-22, Corollary 1
 *
 * SBFP bit representation (Eq. 18, 22):
 *   x_SBFP = (-1)^s × m × 2^{E_shared}
 *   where s   = sign bit (1b)
 *         m   = unsigned mantissa (N_BITS-1 bits)
 *         E_shared = compile-time constant from sbfp_config.h
 *
 * The engine provides two operations:
 *   1. sbfp_quantise()    – FP32/INT16 → SBFP_8b/4b (used at PS→PL DMA hand-off)
 *   2. sbfp_dequantise()  – SBFP_8b/4b → scaled INT32 (used at layer output)
 *
 * Hardware cost (Corollary 1): ZERO runtime logic.
 *   The shared exponent is a constant wired into the datapath at synthesis.
 *   Quantisation = arithmetic right-shift by |E*| (wire remap, 0 LUT).
 *   Dequantisation = arithmetic left-shift by |E*| (wire remap, 0 LUT).
 *=============================================================================*/
#ifndef SBFP_ENGINE_H
#define SBFP_ENGINE_H

#include <ap_int.h>
#include <ap_fixed.h>
#include "sbfp_config.h"

/*---------------------------------------------------------------------------
 * Data types
 *---------------------------------------------------------------------------*/
typedef ap_int<8>   SBFP8;     /* signed 8-bit mantissa  (SBFP-8b mode) */
typedef ap_int<4>   SBFP4;     /* signed 4-bit mantissa  (SBFP-4b mode) */
typedef ap_int<32>  Acc32;

/*---------------------------------------------------------------------------
 * sbfp_quantise_8b()
 *
 * Convert a 16-bit integer activation to 8-bit SBFP mantissa.
 * The scale = 2^{E*} is a compile-time constant → synthesises to a
 * right-shift by |E*| bits (wire remap only, 0 LUT cost).
 *
 * Template parameter LAYER_IDX selects the per-layer exponent from
 * sbfp_config.h.  The compiler resolves SBFP_EXPONENTS_8B[LAYER_IDX]
 * at elaboration time.
 *---------------------------------------------------------------------------*/
template<int LAYER_IDX>
static inline SBFP8 sbfp_quantise_8b(ap_int<16> x)
{
#pragma HLS INLINE
    /*
     * Shift amount = -E* (E* is negative for activations in (0,1) range).
     * Constant shift → synthesiser emits wire remap, not a barrel shifter.
     */
    const int shift = -SBFP_EXPONENTS_8B[LAYER_IDX];
    ap_int<16> shifted = (shift >= 0) ? (x >> shift) : (x << (-shift));
    /* Saturate to 8-bit signed range [-127, 127] */
    if      (shifted >  127) return (SBFP8) 127;
    else if (shifted < -127) return (SBFP8)-127;
    else                     return (SBFP8)(shifted);
}

/*---------------------------------------------------------------------------
 * sbfp_quantise_4b()  – 4-bit mantissa variant (SBFP-4b + INT4 packing)
 *---------------------------------------------------------------------------*/
template<int LAYER_IDX>
static inline SBFP4 sbfp_quantise_4b(ap_int<16> x)
{
#pragma HLS INLINE
    const int shift = -SBFP_EXPONENTS_4B[LAYER_IDX];
    ap_int<16> shifted = (shift >= 0) ? (x >> shift) : (x << (-shift));
    if      (shifted >   7) return (SBFP4) 7;
    else if (shifted <  -7) return (SBFP4)-7;
    else                    return (SBFP4)(shifted);
}

/*---------------------------------------------------------------------------
 * sbfp_dequantise_8b()
 *
 * Convert 8-bit mantissa back to full-precision accumulator after MAC.
 * Multiply by 2^{E*} = left-shift by |E*| (wire remap, 0 LUT).
 * Result fed to bias-add + Hard-Swish in Stage 3 of PE pipeline (Eq.17).
 *---------------------------------------------------------------------------*/
template<int LAYER_IDX>
static inline Acc32 sbfp_dequantise_8b(SBFP8 m)
{
#pragma HLS INLINE
    const int shift = -SBFP_EXPONENTS_8B[LAYER_IDX];  /* positive */
    return ((Acc32)m) << shift;
}

/*---------------------------------------------------------------------------
 * sbfp_bn_fold()
 *
 * Batch-normalisation parameters are folded into the weight tensor offline
 * (Section 4.2, before SBFP calibration), eliminating runtime BN computation.
 * This function is a documentation stub showing the folding formula:
 *
 *   w_fold = w × (γ / √(σ²+ε))
 *   b_fold = β - μ × (γ / √(σ²+ε))
 *
 * The folded weights are stored in sbfp_config.h weight tables.
 * No hardware logic is generated from this function.
 *---------------------------------------------------------------------------*/
static inline void sbfp_bn_fold(
    float *weights_out, float *bias_out,
    const float *weights_in, const float *bias_in,
    const float *gamma, const float *beta,
    const float *mean, const float *var,
    int n_filters, float eps = 1e-5f)
{
    for (int i = 0; i < n_filters; i++) {
        float scale = gamma[i] / __builtin_sqrtf(var[i] + eps);
        weights_out[i] = weights_in[i] * scale;
        bias_out[i]    = beta[i] - mean[i] * scale + (bias_in ? bias_in[i] : 0.0f);
    }
}

#endif /* SBFP_ENGINE_H */
