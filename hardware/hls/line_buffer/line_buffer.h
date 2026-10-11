/*=============================================================================
 * line_buffer.h  –  Sliding-Window Line Buffer + Ping-Pong Memory Subsystem
 *
 * Paper: Section 4.4 / 4.5, Theorem 4, Eq. 26
 *
 * Theorem 4 (Line Buffer Data Reuse):
 *   For a K×K convolution on feature map (W, C), the line buffer reduces
 *   DDR read bandwidth by K², from K²·W·C reads/output row to W·C reads/row.
 *   Data reuse factor R = K²  (Eq. 26)
 *
 * Implementation:
 *   K-1 = 2 FIFO line queues (for K=3 conv), depth = W × C / T_IN
 *   Sliding window: shift register of K×K = 9 elements per T_IN group
 *   Runtime-configurable depth via AXI-Lite register write (Section 4.4)
 *
 * Memory subsystem (Section 4.5):
 *   4 × AXI-HP ports, 128-bit width, 300 MHz → 19.2 GB/s peak (Eq. 15)
 *   Ping-pong double buffer: Buffer_A feeds PE array while Buffer_B
 *   pre-fetches next tile via scatter-gather DMA (Eq. 27)
 *   Buffer size: PINGPONG_KB KB per input channel (configurable)
 *=============================================================================*/
#ifndef LINE_BUFFER_H
#define LINE_BUFFER_H

#include <ap_int.h>
#include <hls_stream.h>

#include "../pe_array/pe_array.h"

/*---------------------------------------------------------------------------
 * Constants – match synthesis target
 *---------------------------------------------------------------------------*/
#define MAX_IMG_W       640     /* max feature map width (pixels)           */
#define MAX_IMG_H       640     /* max feature map height                   */
#define MAX_CHANNELS    1024    /* max channels (P5 C2f, Table 5)           */
#define PINGPONG_KB     512     /* on-chip ping-pong buffer per channel (KB)*/

/* AXI-HP port data width (bits) – ZCU102 HP ports are 128-bit             */
#define AXI_HP_WIDTH    128
typedef ap_uint<AXI_HP_WIDTH> AxiWord;

/*---------------------------------------------------------------------------
 * Sliding-window line buffer (K=3 only; extend for K=1 padding)
 *---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------
 * Window register: K×K elements per input channel group
 *---------------------------------------------------------------------------*/
struct Window3x3 {
    Act8 pix[K][K];
};

/*---------------------------------------------------------------------------
 * Ping-pong buffer descriptor (managed by PS scatter-gather DMA)
 *---------------------------------------------------------------------------*/
struct PingPongDesc {
    volatile Act8  *buf[2];     /* two on-chip BRAM banks                  */
    volatile int    active;     /* which bank PE array currently reads      */
    volatile int    ready;      /* which bank DMA has finished filling       */
};

/*---------------------------------------------------------------------------
 * Function declarations
 *---------------------------------------------------------------------------*/

/*
 * line_buffer_fill()
 *
 * Fills the sliding-window buffer from the AXI-HP input stream and
 * emits 3×3 windows to the PE array stream (one window per output pixel).
 *
 * Data reuse: each input pixel is read once from DDR, reused K²=9 times
 * in the window register before being evicted from the FIFO queue.
 * → DDR bandwidth reduction: 9× vs naive approach (Theorem 4).
 *
 * Args:
 *   in_stream   – AXI-HP input (row-major, C-major layout)
 *   win_stream  – 3×3 window output stream to PE array
 *   img_w       – feature map width  (runtime, ≤ MAX_IMG_W)
 *   img_h       – feature map height (runtime, ≤ MAX_IMG_H)
 *   n_channels  – number of input channels (runtime, ≤ MAX_CHANNELS)
 */
void line_buffer_fill(
    hls::stream<AxiWord>  &in_stream,
    hls::stream<Window3x3> &win_stream,
    int                    img_w,
    int                    img_h,
    int                    n_channels
);

/*
 * pingpong_dma_ctrl()
 *
 * Controls ping-pong double buffering (Section 4.5).
 * While Buffer_A feeds the PE array (read phase), Buffer_B is
 * pre-fetched from DDR4 via scatter-gather DMA (write phase).
 * After PE completes a tile, buffers swap atomically.
 *
 * AXI-Lite register map (8 registers × 4 bytes):
 *   0x00 – ctrl:    bit0=start, bit1=reset
 *   0x04 – status:  bit0=done, bit1=active_buf
 *   0x08 – buf0_lo: base address of buffer 0 (low 32b)
 *   0x0C – buf0_hi: base address of buffer 0 (high 32b)
 *   0x10 – buf1_lo: base address of buffer 1 (low 32b)
 *   0x14 – buf1_hi: base address of buffer 1 (high 32b)
 *   0x18 – tile_sz: tile size in bytes
 *   0x1C – n_tiles: total number of tiles to transfer
 */
void pingpong_dma_ctrl(
    hls::stream<AxiWord>  &dma_rd_stream,  /* AXI-HP read port               */
    hls::stream<AxiWord>  &dma_wr_stream,  /* AXI-HP write port              */
    volatile ap_uint<32>  *ctrl_reg,       /* AXI-Lite ctrl                  */
    volatile ap_uint<32>  *status_reg,     /* AXI-Lite status                */
    ap_uint<64>            buf0_addr,      /* physical address buffer 0      */
    ap_uint<64>            buf1_addr,      /* physical address buffer 1      */
    int                    tile_bytes,     /* bytes per tile                  */
    int                    n_tiles        /* total tiles to DMA              */
);

#endif /* LINE_BUFFER_H */
