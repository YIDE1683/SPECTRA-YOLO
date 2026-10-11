/*=============================================================================
 * spectra_yolo_top.cpp  –  SPECTRA-YOLO PL Accelerator Top (HLS DATAFLOW)
 *
 * Dataflow graph (Fig. 3, Section 4):
 *
 *   DDR4 ──HP1──► line_buffer_fill ──► act_stream ──►┐
 *   DDR4 ──HP0──► weight_load      ──► wgt_stream ──►│
 *   DDR4 ──HP3──► bias_load        ──► bias_stream──►├──► pe_array_top ──► out_stream ──HP2──► DDR4
 *                                                     │
 *   SBFP exponent applied as compile-time wire remap ─┘
 *
 * HLS DATAFLOW enables the three sub-functions to execute concurrently,
 * overlapping memory access with computation (Section 5.1).
 *
 * Pragma summary (Section 5.3):
 *   #pragma HLS PIPELINE II=1    → innermost MAC loop
 *   #pragma HLS DATAFLOW         → top-level concurrency
 *   #pragma HLS INTERFACE m_axi  → four AXI-HP master ports
 *   #pragma HLS INTERFACE s_axilite → control/status registers
 *=============================================================================*/
#include "spectra_yolo_top.h"

/*---------------------------------------------------------------------------
 * weight_load()
 *
 * Streams weights from DDR4 (HP0) into the PE array weight buffer and packs
 * them into INT4 activation-broadcast groups (§4.3): three output-channel
 * weights per DSP A-port (29-bit PackedW = magnitudes @ 0/11/22 + 3 signs).
 * For INT4 mode: three 4-bit weights per group (12 raw bits).
 * For INT8 mode: three 8-bit weights truncated to INT4 per group (24 raw bits).
 *---------------------------------------------------------------------------*/
static void weight_load(
    ap_uint<AXI_HP_WIDTH>  *weight_mem,
    hls::stream<PackedW>   &wgt_stream,
    int n_packed_w, int use_int4)
{
#pragma HLS INLINE
    int group_bits = use_int4 ? (INT4_PACK * 4) : (INT4_PACK * 8);
    int groups     = AXI_HP_WIDTH / group_bits;
    int emit_groups = (groups > 4) ? 4 : groups;
    int n_wgt_words = (n_packed_w + emit_groups - 1) / emit_groups;
    int produced = 0;

    WLOAD_LOOP:
    for (int i = 0; i < n_wgt_words; i++) {
#pragma HLS PIPELINE II=1
        ap_uint<AXI_HP_WIDTH> axi_word = weight_mem[i];
        /* Each 128-bit word feeds several 3-weight broadcast groups (§4.3). */
        for (int s = 0; s < 4 && s < groups && produced < n_packed_w; s++, produced++) {
#pragma HLS UNROLL
            Weight4 w0, w1, w2;
            if (use_int4) {
                w0 = (Weight4)((ap_int<4>)(axi_word >> (s*12)));
                w1 = (Weight4)((ap_int<4>)(axi_word >> (s*12 + 4)));
                w2 = (Weight4)((ap_int<4>)(axi_word >> (s*12 + 8)));
            } else {
                /* INT8 source: keep low INT4 nibble for the broadcast group */
                w0 = (Weight4)((ap_int<8>)(axi_word >> (s*24)));
                w1 = (Weight4)((ap_int<8>)(axi_word >> (s*24 + 8)));
                w2 = (Weight4)((ap_int<8>)(axi_word >> (s*24 + 16)));
            }
            wgt_stream.write(pack_weights_int4(w0, w1, w2));
        }
    }
}

/*---------------------------------------------------------------------------
 * bias_load()
 *
 * Streams per-output-channel biases (INT32) from DDR4 via HP3.
 *---------------------------------------------------------------------------*/
static void bias_load(
    ap_uint<AXI_HP_WIDTH>  *bias_mem,
    hls::stream<Acc32>     &bias_stream,
    int n_out_ch, int oc_tiles)
{
#pragma HLS INLINE off
    int total_bias = oc_tiles * T_OUT;
    int words = (n_out_ch * 4 + AXI_HP_WIDTH/8 - 1) / (AXI_HP_WIDTH/8);
    int ch = 0;
    BLOAD_LOOP:
    for (int w = 0; w < words; w++) {
#pragma HLS PIPELINE II=1
        ap_uint<AXI_HP_WIDTH> axi_word = bias_mem[w];
        for (int b = 0; b < AXI_HP_WIDTH/32 && ch < n_out_ch; b++, ch++) {
#pragma HLS UNROLL
            Acc32 bias = (Acc32)(axi_word >> (b * 32));
            bias_stream.write(bias);
        }
    }
    BPAD_LOOP:
    for (; ch < total_bias; ch++) {
#pragma HLS PIPELINE II=1
        bias_stream.write((Acc32)0);
    }
}

/*---------------------------------------------------------------------------
 * activation_load()
 *
 * Reads the input activation tensor from HP1 and feeds the line-buffer stream.
 * The activation pass is repeated once per output-channel tile so that the
 * current streaming PE array can process each output tile without requiring a
 * full-frame activation replay buffer inside PL.
 *---------------------------------------------------------------------------*/
static void activation_load(
    ap_uint<AXI_HP_WIDTH>  *act_mem,
    hls::stream<AxiWord>   &axi_act_stream,
    int img_w, int img_h, int in_ch, int oc_tiles)
{
#pragma HLS INLINE off
    int words_per_pixel = (in_ch + (AXI_HP_WIDTH/8) - 1) / (AXI_HP_WIDTH/8);
    int words_per_pass  = img_w * img_h * words_per_pixel;

    ACT_TILE_LOOP:
    for (int ot = 0; ot < oc_tiles; ot++) {
        ACT_LOAD_LOOP:
        for (int i = 0; i < words_per_pass; i++) {
#pragma HLS PIPELINE II=1
            axi_act_stream.write(act_mem[i]);
        }
    }
}

static void line_buffer_fill_tiles(
    hls::stream<AxiWord>   &axi_act_stream,
    hls::stream<Window3x3>  &win_stream,
    int img_w, int img_h, int in_ch, int oc_tiles)
{
#pragma HLS INLINE off
    LBUF_TILE_LOOP:
    for (int ot = 0; ot < oc_tiles; ot++) {
        line_buffer_fill(axi_act_stream, win_stream, img_w, img_h, in_ch);
    }
}

static void window_to_act_stream(
    hls::stream<Window3x3>  &win_stream,
    hls::stream<PackedA>    &act_stream,
    int img_w, int img_h, int in_ch, int oc_tiles)
{
#pragma HLS INLINE off
    int valid_w = (img_w >= K) ? (img_w - K + 1) : 0;
    int valid_h = (img_h >= K) ? (img_h - K + 1) : 0;
    int spatial_valid = valid_w * valid_h;
    int ic_tiles = (in_ch + T_IN - 1) / T_IN;
    int safe_ic_tiles = (ic_tiles > MAX_IC_TILES) ? MAX_IC_TILES : ic_tiles;

    WIN_TILE_LOOP:
    for (int ot = 0; ot < oc_tiles; ot++) {
        WIN_SPATIAL_LOOP:
        for (int sp = 0; sp < spatial_valid; sp++) {
            WIN_ICT_LOOP:
            for (int it = 0; it < safe_ic_tiles; it++) {
                WIN_LANE_LOOP:
                for (int lane = 0; lane < T_IN; lane++) {
                    int ch = it * T_IN + lane;
                    Window3x3 win;
                    bool valid_ch = (ch < in_ch);
                    if (valid_ch) {
                        win = win_stream.read();
                    }
                    WIN_TAP_ROW:
                    for (int r = 0; r < K; r++) {
                        WIN_TAP_COL:
                        for (int c = 0; c < K; c++) {
#pragma HLS PIPELINE II=1
                            Act4 a = valid_ch ? (Act4)win.pix[r][c] : (Act4)0;
                            act_stream.write(pack_acts_int4(a));
                        }
                    }
                }
            }
        }
    }
}

/*---------------------------------------------------------------------------
 * output_store()
 *
 * Writes PE output stream back to DDR4 via HP2.
 * SBFP dequantisation (compile-time left-shift) applied here.
 *---------------------------------------------------------------------------*/
static void output_store(
    hls::stream<Out16>     &out_stream,
    ap_uint<AXI_HP_WIDTH>  *out_mem,
    int n_out_elems, int layer_idx)
{
#pragma HLS INLINE off
    (void)layer_idx;
    int words = (n_out_elems * 2 + AXI_HP_WIDTH/8 - 1) / (AXI_HP_WIDTH/8);
    ap_uint<AXI_HP_WIDTH> axi_word = 0;
    int elem = 0;
    OSTORE_LOOP:
    for (int w = 0; w < words; w++) {
#pragma HLS PIPELINE II=1
        axi_word = 0;
        for (int b = 0; b < AXI_HP_WIDTH/16 && elem < n_out_elems; b++, elem++) {
            Out16 val = out_stream.read();
            axi_word |= ((ap_uint<AXI_HP_WIDTH>)(ap_uint<16>)val) << (b * 16);
        }
        out_mem[w] = axi_word;
    }
}

/*---------------------------------------------------------------------------
 * spectra_yolo_top()  –  top-level DATAFLOW function
 *---------------------------------------------------------------------------*/
void spectra_yolo_top(
    ap_uint<AXI_HP_WIDTH>  *weight_mem,
    ap_uint<AXI_HP_WIDTH>  *act_mem,
    ap_uint<AXI_HP_WIDTH>  *out_mem,
    ap_uint<AXI_HP_WIDTH>  *bias_mem,
    int layer_idx, int img_w, int img_h,
    int in_ch, int out_ch, int use_int4,
    volatile int *done_tiles)
{
/* AXI4 Master interfaces – one per HP port (burst, 64-beat max burst) */
#pragma HLS INTERFACE m_axi port=weight_mem offset=slave bundle=HP0 depth=65536 max_read_burst_length=64
#pragma HLS INTERFACE m_axi port=act_mem    offset=slave bundle=HP1 depth=65536 max_read_burst_length=64
#pragma HLS INTERFACE m_axi port=out_mem    offset=slave bundle=HP2 depth=65536 max_write_burst_length=64
#pragma HLS INTERFACE m_axi port=bias_mem   offset=slave bundle=HP3 depth=4096  max_read_burst_length=16
/* AXI4-Lite Slave – all scalar args + return share 'ctrl' bundle */
#pragma HLS INTERFACE s_axilite port=layer_idx   bundle=ctrl
#pragma HLS INTERFACE s_axilite port=img_w       bundle=ctrl
#pragma HLS INTERFACE s_axilite port=img_h       bundle=ctrl
#pragma HLS INTERFACE s_axilite port=in_ch       bundle=ctrl
#pragma HLS INTERFACE s_axilite port=out_ch      bundle=ctrl
#pragma HLS INTERFACE s_axilite port=use_int4    bundle=ctrl
#pragma HLS INTERFACE s_axilite port=done_tiles  bundle=ctrl
#pragma HLS INTERFACE s_axilite port=return      bundle=ctrl

    int oc_tiles   = (out_ch + T_OUT - 1) / T_OUT;
    int ic_tiles   = (in_ch  + T_IN  - 1) / T_IN;
    int safe_ic_tiles = (ic_tiles > MAX_IC_TILES) ? MAX_IC_TILES : ic_tiles;
    int valid_w = (img_w >= K) ? (img_w - K + 1) : 0;
    int valid_h = (img_h >= K) ? (img_h - K + 1) : 0;
    int spatial = valid_w * valid_h;
    int n_packed_w = oc_tiles * safe_ic_tiles * N_OUT_GROUPS * T_IN * K * K;
    int n_out_elems = oc_tiles * T_OUT * spatial;

    *done_tiles = oc_tiles * safe_ic_tiles;

    hls::stream<PackedW>    wgt_stream ("wgt_stream");
    hls::stream<PackedA>    act_stream ("act_stream");
    hls::stream<Acc32>      bias_stream("bias_stream");
    hls::stream<Out16>      out_stream ("out_stream");
    hls::stream<AxiWord>    axi_act_stream("axi_act_stream");
    hls::stream<Window3x3>  win_stream ("win_stream");
#pragma HLS STREAM variable=wgt_stream  depth=512
#pragma HLS STREAM variable=act_stream  depth=512
#pragma HLS STREAM variable=bias_stream depth=64
#pragma HLS STREAM variable=out_stream  depth=512
#pragma HLS STREAM variable=axi_act_stream depth=256
#pragma HLS STREAM variable=win_stream  depth=256

#pragma HLS DATAFLOW

    weight_load(weight_mem, wgt_stream, n_packed_w, use_int4);
    bias_load(bias_mem, bias_stream, out_ch, oc_tiles);
    activation_load(act_mem, axi_act_stream, img_w, img_h, in_ch, oc_tiles);
    line_buffer_fill_tiles(axi_act_stream, win_stream, img_w, img_h, in_ch, oc_tiles);
    window_to_act_stream(win_stream, act_stream, img_w, img_h, in_ch, oc_tiles);
    pe_array_top(wgt_stream, act_stream, bias_stream, out_stream,
                 oc_tiles, safe_ic_tiles, spatial);
    output_store(out_stream, out_mem, n_out_elems, layer_idx);
}
