/*=============================================================================
 * pe_array.cpp  –  SPECTRA-YOLO PE Array Implementation (HLS)
 *
 * Synthesise with Vitis HLS 2022.2 targeting xczu9eg-ffvb1156-2-e
 * Target clock: 5 ns (200 MHz PL fabric)
 *
 * Key pragmas:
 *   #pragma HLS PIPELINE II=1    – inner MAC loop, 1 new MAC per cycle
 *   #pragma HLS ARRAY_PARTITION  – weight/act arrays across DSP48E2 ports
 *   #pragma HLS DATAFLOW         – overlap MAC with bias/activation stages
 *
 * INT4 packing (§4.3):  one DSP48E2 computes three 4-bit MACs per cycle
 *   via activation-broadcast (one |a| on B-port, three |w| on A-port)
 *   → peak 1.84 TOPS × 3 = 5.53 TOPS                  (Eq. 14)
 *   → DSP count: ⌈(32×16)/3⌉ + 1 = 172 DSP48E2 slices  (§4.3)
 *=============================================================================*/
#include "pe_array.h"

/*---------------------------------------------------------------------------
 * systolic_mac_unit()
 *
 * Models one DSP48E2 in INT4 activation-broadcast mode (§4.3).
 * One activation magnitude |a| (B-port) is broadcast to three weight
 * magnitudes |w0|,|w1|,|w2| (A-port).  The three |w_i|·|a| products land in
 * guard-band-isolated windows at bit 0/11/22; signs are restored by XOR.
 *---------------------------------------------------------------------------*/
static void systolic_mac_unit(
    PackedW  pw,        /* three |w| magnitudes + signs (A-port [28:0]) */
    PackedA  pa,        /* one broadcast |a| magnitude + sign (B-port)  */
    Acc32   &acc0,      /* accumulator – output-channel lane 0          */
    Acc32   &acc1,      /* accumulator – output-channel lane 1          */
    Acc32   &acc2       /* accumulator – output-channel lane 2          */
)
{
#pragma HLS INLINE
#pragma HLS PIPELINE II=1

    /* DSP48E2 unsigned multiply: 27b A-port × 4b B-port → 31b P-register */
    ap_uint<27> magA = pw.range(25, 0);
    ap_uint<4>  magB = pa.range(3, 0);
    ap_uint<31> dsp_result = magA * magB;

    /* Extract three |w_i|·|a| products from guard-band-isolated windows
     * (slots @ bit 0/11/22, each 8b product + g=3 guard = 11b).          */
    ap_uint<8> p0 = dsp_result.range( 7,  0);
    ap_uint<8> p1 = dsp_result.range(18, 11);
    ap_uint<8> p2 = dsp_result.range(29, 22);

    /* Sign restoration: s_i = sign(w_i) XOR sign(a)  (§4.3) */
    bool sa = pa[4];
    acc0 += (pw[26] ^ sa) ? (Acc32)(-(int)p0) : (Acc32)p0;
    acc1 += (pw[27] ^ sa) ? (Acc32)(-(int)p1) : (Acc32)p1;
    acc2 += (pw[28] ^ sa) ? (Acc32)(-(int)p2) : (Acc32)p2;
}

/*---------------------------------------------------------------------------
 * pe_group()  –  three output channels packed into one DSP lane (§4.3)
 *
 * Each input tap activation is broadcast to the three output-channel weights
 * packed in the same DSP48E2 slice, producing three partial sums per cycle.
 *---------------------------------------------------------------------------*/
static void pe_group(
    PackedW  weights[T_IN * K * K],  /* per tap: three out-channel |w| packed  */
    PackedA  acts[T_IN * K * K],     /* per tap: one broadcast |a|             */
    Acc32    bias0, Acc32 bias1, Acc32 bias2,
    Out16   &out0, Out16 &out1, Out16 &out2
)
{
#pragma HLS INLINE
    Acc32 acc0 = 0, acc1 = 0, acc2 = 0;

    MAC_LOOP:
    for (int i = 0; i < T_IN * K * K; i++) {
#pragma HLS PIPELINE II=1
        systolic_mac_unit(weights[i], acts[i], acc0, acc1, acc2);
    }

    /* Stage 3: bias add + Hard-Swish (Eq. 17, Stage 3) */
    out0 = (Out16)(hard_swish(acc0 + bias0).range(15, 0));
    out1 = (Out16)(hard_swish(acc1 + bias1).range(15, 0));
    out2 = (Out16)(hard_swish(acc2 + bias2).range(15, 0));
}

static void pe_group_accumulate(
    PackedW  weights[T_IN * K * K],
    PackedA  acts[T_IN * K * K],
    Acc32   &acc0,
    Acc32   &acc1,
    Acc32   &acc2)
{
#pragma HLS INLINE
    MAC_ACC_LOOP:
    for (int i = 0; i < T_IN * K * K; i++) {
#pragma HLS PIPELINE II=1
        systolic_mac_unit(weights[i], acts[i], acc0, acc1, acc2);
    }
}

/*---------------------------------------------------------------------------
 * pe_array_top()  –  32×16 systolic grid (outer loop over output tiles)
 *
 * Streams:
 *   weight_stream  → N_OUT_GROUPS × T_IN × K² PackedW per output tile
 *   act_stream     → T_IN × K² PackedA per spatial position (broadcast)
 *   bias_stream    → T_OUT biases per output tile
 *   out_stream     → T_OUT Out16 per spatial position
 *---------------------------------------------------------------------------*/
void pe_array_top(
    hls::stream<PackedW>  &weight_stream,
    hls::stream<PackedA>  &act_stream,
    hls::stream<Acc32>    &bias_stream,
    hls::stream<Out16>    &out_stream,
    int                    oc_tiles,
    int                    ic_tiles,
    int                    spatial_len
)
{
#pragma HLS INTERFACE axis port=weight_stream
#pragma HLS INTERFACE axis port=act_stream
#pragma HLS INTERFACE axis port=bias_stream
#pragma HLS INTERFACE axis port=out_stream
#pragma HLS INTERFACE s_axilite port=oc_tiles   bundle=ctrl
#pragma HLS INTERFACE s_axilite port=ic_tiles   bundle=ctrl
#pragma HLS INTERFACE s_axilite port=spatial_len bundle=ctrl
#pragma HLS INTERFACE s_axilite port=return     bundle=ctrl

    int safe_ic_tiles = (ic_tiles > MAX_IC_TILES) ? MAX_IC_TILES : ic_tiles;

    /* Weight buffer: output groups × input-channel tiles × 3×3 taps. */
    PackedW w_buf[N_OUT_GROUPS][MAX_IC_TILES][T_IN * K * K];
#pragma HLS ARRAY_PARTITION variable=w_buf dim=1 complete
#pragma HLS RESOURCE variable=w_buf core=RAM_2P_BRAM

    Acc32 bias_buf[T_OUT];
#pragma HLS ARRAY_PARTITION variable=bias_buf dim=1 complete

    TILE_LOOP:
    for (int ot = 0; ot < oc_tiles; ot++) {

        WEIGHT_LOAD:
        for (int it = 0; it < safe_ic_tiles; it++)
            for (int g = 0; g < N_OUT_GROUPS; g++)
                for (int k = 0; k < T_IN * K * K; k++) {
#pragma HLS PIPELINE II=1
                    w_buf[g][it][k] = weight_stream.read();
                }

        BIAS_LOAD:
        for (int row = 0; row < T_OUT; row++) {
#pragma HLS PIPELINE II=1
            bias_buf[row] = bias_stream.read();
        }

        SPATIAL_LOOP:
        for (int sp = 0; sp < spatial_len; sp++) {
            Acc32 acc0[N_OUT_GROUPS], acc1[N_OUT_GROUPS], acc2[N_OUT_GROUPS];
#pragma HLS ARRAY_PARTITION variable=acc0 complete
#pragma HLS ARRAY_PARTITION variable=acc1 complete
#pragma HLS ARRAY_PARTITION variable=acc2 complete

            INIT_ACC:
            for (int g = 0; g < N_OUT_GROUPS; g++) {
#pragma HLS UNROLL
                int oc0 = g * INT4_PACK;
                acc0[g] = bias_buf[oc0];
                acc1[g] = (oc0 + 1 < T_OUT) ? bias_buf[oc0 + 1] : (Acc32)0;
                acc2[g] = (oc0 + 2 < T_OUT) ? bias_buf[oc0 + 2] : (Acc32)0;
            }

            ICT_LOOP:
            for (int it = 0; it < safe_ic_tiles; it++) {
                PackedA act_win[T_IN * K * K];
#pragma HLS ARRAY_PARTITION variable=act_win complete

                ACT_LOAD:
                for (int k = 0; k < T_IN * K * K; k++) {
#pragma HLS PIPELINE II=1
                    act_win[k] = act_stream.read();
                }

                PE_GROUP_LOOP:
                for (int g = 0; g < N_OUT_GROUPS; g++) {
#pragma HLS UNROLL
                    pe_group_accumulate(w_buf[g][it], act_win, acc0[g], acc1[g], acc2[g]);
                }
            }

            WRITE_GROUP_LOOP:
            for (int g = 0; g < N_OUT_GROUPS; g++) {
#pragma HLS UNROLL
                int oc0 = g * INT4_PACK;
                Out16 r0 = (Out16)(hard_swish(acc0[g]).range(15, 0));
                Out16 r1 = (Out16)(hard_swish(acc1[g]).range(15, 0));
                Out16 r2 = (Out16)(hard_swish(acc2[g]).range(15, 0));
                out_stream.write(r0);
                if (oc0 + 1 < T_OUT) out_stream.write(r1);
                if (oc0 + 2 < T_OUT) out_stream.write(r2);
            }
        }
    }
}
