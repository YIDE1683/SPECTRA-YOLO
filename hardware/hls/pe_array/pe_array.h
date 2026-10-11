/*=============================================================================
 * pe_array.h  –  Processing Element Array for SPECTRA-YOLO on ZCU102
 *
 * Paper: Section 4.1 / 4.3   (DSP48E2 INT4 Activation-Broadcast Packing)
 *
 * Hardware specs (ZCU102, Table 5):
 *   Grid:       32 × 16 PEs  (T_out=32, T_in=16)
 *   Kernel:     9 MACs/PE    (3×3 conv)
 *   INT8 peak:  32×16×9×200MHz×2 = 1.84 TOPS  (Eq. 13)
 *   INT4 peak:  3× packing   = 5.53 TOPS       (Eq. 14)
 *   DSP48E2:    512 → 172    (⌈512/3⌉+1, §4.3)
 *
 * INT4 DSP48E2 activation-broadcast layout (§4.3, guard band g=3 bits):
 *   A-port [25:0]:  [ |w2| |g| |w1| |g| |w0| ]  three 4b magnitudes @ 0/11/22
 *   B-port [ 3:0]:  [ |a| ]                       one broadcast 4b magnitude
 *   Product:  |a|·A places |w_i|·|a| at bit 0/11/22 (8b product + 3b guard = 11b)
 *   Signs:    s_i = sign(w_i) XOR sign(a) restored after extraction (§4.3)
 *
 * Pipeline stages per PE (Eq. 17):
 *   Stage 1: INT8/INT4 multiply via DSP48E2
 *   Stage 2: 32-bit accumulate with overflow protection
 *   Stage 3: bias addition + Hard-Swish activation
 *=============================================================================*/
#ifndef PE_ARRAY_H
#define PE_ARRAY_H

#include <ap_int.h>
#include <hls_stream.h>

/*---------------------------------------------------------------------------
 * Compile-time parameters – adjust to match synthesis target
 *---------------------------------------------------------------------------*/
#define T_OUT       32          /* output-channel tile (PE rows)            */
#define T_IN        16          /* input-channel  tile (PE columns)         */
#define K           3           /* convolution kernel size                  */
#define MAX_IC_TILES 64         /* max 1024 input channels / T_IN           */
#define INT4_PACK   3           /* 3-product activation-broadcast (§4.3)    */
#define GUARD_BITS  3           /* guard band width g=3 (§4.3)              */
#define PROD_BITS   8           /* |w|·|a| product width (4b × 4b)          */
#define SLOT_WIDTH  (PROD_BITS + GUARD_BITS)  /* 11-bit product window      */
#define N_OUT_GROUPS ((T_OUT + INT4_PACK - 1) / INT4_PACK)  /* 11 groups    */

/*---------------------------------------------------------------------------
 * Data types
 *---------------------------------------------------------------------------*/
typedef ap_int<4>   Weight4;    /* signed 4-bit weight                      */
typedef ap_int<4>   Act4;       /* signed 4-bit activation                  */
typedef ap_int<8>   Weight8;    /* signed 8-bit weight  (SBFP-8b mode)      */
typedef ap_int<8>   Act8;       /* signed 8-bit activation                  */
typedef ap_uint<32> PackedW;    /* A-port: three |w| @ 0/11/22 + signs[28:26]*/
typedef ap_uint<8>  PackedA;    /* B-port: broadcast |a|[3:0] + sign[4]      */
typedef ap_int<32>  Acc32;      /* 32-bit accumulator with overflow protect */
typedef ap_int<16>  Out16;      /* output after quantisation                */

/*---------------------------------------------------------------------------
 * INT4 DSP48E2 activation-broadcast packing helpers (§4.3)
 *---------------------------------------------------------------------------*/
/*
 * pack_weights_int4() – pack three signed INT4 weights onto the A-port (§4.3)
 *   A[ 3: 0] = |w0|,  A[14:11] = |w1|,  A[25:22] = |w2|   (magnitudes @ 0/11/22)
 *   signs    : [26]=s0, [27]=s1, [28]=s2                  (restored post-multiply)
 *   guard bits [10:4],[21:15] stay zero – absorb |w|·|a| product carry
 */
static inline PackedW pack_weights_int4(Weight4 w0, Weight4 w1, Weight4 w2)
{
#pragma HLS INLINE
    PackedW packed = 0;
    packed.range( 3,  0) = (w0 < 0) ? (ap_uint<4>)(-w0) : (ap_uint<4>)w0;
    packed.range(14, 11) = (w1 < 0) ? (ap_uint<4>)(-w1) : (ap_uint<4>)w1;
    packed.range(25, 22) = (w2 < 0) ? (ap_uint<4>)(-w2) : (ap_uint<4>)w2;
    packed[26] = (w0 < 0);
    packed[27] = (w1 < 0);
    packed[28] = (w2 < 0);
    return packed;
}

/*
 * pack_acts_int4() – pack one signed INT4 activation broadcast onto B-port (§4.3)
 *   B[3:0] = |a|,  B[4] = sign(a)   (single magnitude reused by all three w_i)
 */
static inline PackedA pack_acts_int4(Act4 a0)
{
#pragma HLS INLINE
    PackedA packed = 0;
    packed.range(3, 0) = (a0 < 0) ? (ap_uint<4>)(-a0) : (ap_uint<4>)a0;
    packed[4] = (a0 < 0);
    return packed;
}

/*---------------------------------------------------------------------------
 * Hard-Swish activation (Eq. 11): HS(x) = x · ReLU6(x+3) / 6
 * Implemented with integer arithmetic only (no exponential / LUT).
 *---------------------------------------------------------------------------*/
static inline Acc32 hard_swish(Acc32 x)
{
#pragma HLS INLINE
    Acc32 x3   = x + 3;
    Acc32 relu6 = (x3 < 0) ? (Acc32)0 : ((x3 > 6) ? (Acc32)6 : x3);
    return x * relu6 / 6;   /* synthesiser maps div-6 to reciprocal multiply */
}

/*---------------------------------------------------------------------------
 * Top-level PE array function (called from spectra_yolo_top via DATAFLOW)
 *---------------------------------------------------------------------------*/
void pe_array_top(
    hls::stream<PackedW>  &weight_stream,  /* packed INT4 weights in          */
    hls::stream<PackedA>  &act_stream,     /* packed INT4 activations in      */
    hls::stream<Acc32>    &bias_stream,    /* INT32 biases in                 */
    hls::stream<Out16>    &out_stream,     /* output activations (post-HS)    */
    int                    oc_tiles,       /* number of output-channel tiles   */
    int                    ic_tiles,       /* number of input-channel tiles    */
    int                    spatial_len    /* H × W output positions           */
);

#endif /* PE_ARRAY_H */
