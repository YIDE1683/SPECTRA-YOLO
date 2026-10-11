/*=============================================================================
 * line_buffer.cpp  –  Line Buffer + Ping-Pong DMA Controller (HLS)
 *
 * Theorem 4: For K=3, data reuse = 9×, DDR reads = W×C (not K²×W×C)
 * Eq. 26:    R = K² = 9  for 3×3 convolution
 *=============================================================================*/
#include "line_buffer.h"

/*---------------------------------------------------------------------------
 * line_buffer_fill()
 *
 * Architecture:
 *   ┌─────────────────────────────────────────────────────────────────┐
 *   │  DDR4 (AXI-HP)  ──►  FIFO row 1  ──►  Window[0][*]            │
 *   │                 ──►  FIFO row 0  ──►  Window[1][*]            │
 *   │                 ──►  Shift reg   ──►  Window[2][*]  ──► PE    │
 *   └─────────────────────────────────────────────────────────────────┘
 *   Each input pixel read once from FIFO; reused K²=9 times in window.
 *---------------------------------------------------------------------------*/
void line_buffer_fill(
    hls::stream<AxiWord>   &in_stream,
    hls::stream<Window3x3>  &win_stream,
    int img_w, int img_h, int n_channels)
{
#pragma HLS INTERFACE axis     port=in_stream
#pragma HLS INTERFACE axis     port=win_stream
#pragma HLS INTERFACE s_axilite port=img_w      bundle=ctrl
#pragma HLS INTERFACE s_axilite port=img_h      bundle=ctrl
#pragma HLS INTERFACE s_axilite port=n_channels bundle=ctrl
#pragma HLS INTERFACE s_axilite port=return     bundle=ctrl

    /* K-1 = 2 line FIFO queues (one per previous row) */
    static Act8 line_fifo[K-1][MAX_IMG_W * MAX_CHANNELS];
#pragma HLS ARRAY_PARTITION variable=line_fifo dim=1 complete
#pragma HLS RESOURCE variable=line_fifo core=RAM_2P_BRAM

    /* 3×3 sliding window shift register (per channel) */
    Act8 win_reg[K][K];
#pragma HLS ARRAY_PARTITION variable=win_reg complete

    int fifo_ptr = 0;
    int fifo_len = img_w * n_channels;

    ROW_LOOP:
    for (int row = 0; row < img_h; row++) {
        COL_LOOP:
        for (int col = 0; col < img_w; col++) {
            CH_LOOP:
            for (int ch = 0; ch < n_channels; ch += AXI_HP_WIDTH/8) {
#pragma HLS PIPELINE II=1

                /* Read one AXI word (16 bytes = 16 INT8 channels) */
                AxiWord axi_in = in_stream.read();

                CHAN_UNPACK:
                for (int b = 0; b < AXI_HP_WIDTH/8 && ch+b < n_channels; b++) {
#pragma HLS UNROLL
                    Act8 pixel = (Act8)(axi_in >> (b*8));

                    /* Shift window columns left */
                    win_reg[0][0] = win_reg[0][1];
                    win_reg[0][1] = win_reg[0][2];
                    win_reg[1][0] = win_reg[1][1];
                    win_reg[1][1] = win_reg[1][2];
                    win_reg[2][0] = win_reg[2][1];
                    win_reg[2][1] = win_reg[2][2];

                    /* Shift FIFO rows into window */
                    int ptr = (row * img_w + col) * n_channels + ch + b;
                    ptr %= fifo_len;

                    win_reg[0][2] = (row >= 2) ? line_fifo[0][ptr] : (Act8)0;
                    win_reg[1][2] = (row >= 1) ? line_fifo[1][ptr] : (Act8)0;
                    win_reg[2][2] = pixel;

                    /* Update FIFO queues (rotate: oldest evicted, new pixel in) */
                    line_fifo[0][ptr] = line_fifo[1][ptr];  /* promote row 1→0 */
                    line_fifo[1][ptr] = pixel;               /* insert new row  */

                    /* Emit 3×3 window when both border conditions satisfied */
                    if (col >= K-1 && row >= K-1) {
                        Window3x3 w;
                        for (int r = 0; r < K; r++)
                            for (int c = 0; c < K; c++)
                                w.pix[r][c] = win_reg[r][c];
                        win_stream.write(w);
                    }
                }
            }
        }
    }
}

/*---------------------------------------------------------------------------
 * pingpong_dma_ctrl()  –  Ping-pong double buffer management
 *
 * State machine:
 *   IDLE  → START → FILL_BUF1 → SWAP → FILL_BUF0 → SWAP → ...
 *   While PE reads BufA (active), DMA fills BufB (shadow).
 *   SWAP is atomic: only at tile boundary (when PE signals tile done).
 *---------------------------------------------------------------------------*/
void pingpong_dma_ctrl(
    hls::stream<AxiWord>  &dma_rd_stream,
    hls::stream<AxiWord>  &dma_wr_stream,
    volatile ap_uint<32>  *ctrl_reg,
    volatile ap_uint<32>  *status_reg,
    ap_uint<64>            buf0_addr,
    ap_uint<64>            buf1_addr,
    int                    tile_bytes,
    int                    n_tiles)
{
#pragma HLS INTERFACE axis      port=dma_rd_stream
#pragma HLS INTERFACE axis      port=dma_wr_stream
#pragma HLS INTERFACE s_axilite port=ctrl_reg    bundle=ctrl  offset=0x00
#pragma HLS INTERFACE s_axilite port=status_reg  bundle=ctrl  offset=0x04
#pragma HLS INTERFACE s_axilite port=buf0_addr   bundle=ctrl  offset=0x08
#pragma HLS INTERFACE s_axilite port=buf1_addr   bundle=ctrl  offset=0x10
#pragma HLS INTERFACE s_axilite port=tile_bytes  bundle=ctrl  offset=0x18
#pragma HLS INTERFACE s_axilite port=n_tiles     bundle=ctrl  offset=0x1C
#pragma HLS INTERFACE s_axilite port=return      bundle=ctrl

    int  active_buf = 0;   /* 0=buf0 active (PE reads), 1=buf1 active */
    *status_reg = 0;

    TILE_LOOP:
    for (int tile = 0; tile < n_tiles; tile++) {
        int shadow_buf = 1 - active_buf;
        int words = tile_bytes / (AXI_HP_WIDTH / 8);

        /* DMA: pre-fetch next tile into shadow buffer */
        DMA_FETCH:
        for (int w = 0; w < words; w++) {
#pragma HLS PIPELINE II=1
            AxiWord data = dma_rd_stream.read();
            dma_wr_stream.write(data);   /* → BRAM shadow buffer via HP port */
        }

        /* Wait for PE to finish current tile (poll ctrl bit1) */
        WAIT_PE:
        ap_uint<32> ctrl_value = *ctrl_reg;
        while ((ctrl_value & (ap_uint<32>)0x2) == 0) {
#pragma HLS PIPELINE off
            ctrl_value = *ctrl_reg;
        }

        /* Atomic swap: shadow becomes active */
        active_buf = shadow_buf;
        *status_reg = ((ap_uint<32>)(active_buf & 0x1)) | (ap_uint<32>)0x2; /* done bit */
    }
    ap_uint<32> status_value = *status_reg;
    *status_reg = status_value | (ap_uint<32>)0x1;   /* overall done */
}
