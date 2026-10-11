/*=============================================================================
 * spectra_yolo_top.h  –  SPECTRA-YOLO Accelerator Top-Level Interface
 *
 * Paper: Section 4 (complete), Section 5.1, Fig. 3
 *
 * This is the top-level HLS function exported as an AXI4-Lite + AXI4-HP IP.
 * Vivado IP Integrator wraps this into the Block Design (create_bd.tcl).
 *
 * AXI4-Lite control register map (ctrl bundle, base = 0x00):
 *   0x00 – ap_ctrl:    bit0=ap_start, bit1=ap_done, bit2=ap_idle, bit3=ap_ready
 *   0x04 – layer_idx:  current layer index (0-22)
 *   0x08 – img_w:      feature map width
 *   0x0C – img_h:      feature map height
 *   0x10 – in_ch:      number of input channels
 *   0x14 – out_ch:     number of output channels
 *   0x18 – use_int4:   1 = INT4 DSP packing mode, 0 = INT8 mode
 *   0x1C – done_tiles: number of completed spatial tiles (read-only)
 *
 * AXI4 Master ports (M_AXI_DATA, 4 HP ports merged via AXI SmartConnect):
 *   m_axi_data_0 – weight read port   (HP0, 128-bit)
 *   m_axi_data_1 – activation read    (HP1, 128-bit)
 *   m_axi_data_2 – output write       (HP2, 128-bit)
 *   m_axi_data_3 – bias/config read   (HP3, 128-bit)
 *
 * Aggregate bandwidth: 4 × 128-bit × 300 MHz = 19.2 GB/s  (Eq. 15, 27)
 *=============================================================================*/
#ifndef SPECTRA_YOLO_TOP_H
#define SPECTRA_YOLO_TOP_H

#include <ap_int.h>
#include <ap_axi_sdata.h>
#include <hls_stream.h>

#include "../pe_array/pe_array.h"
#include "../sbfp_engine/sbfp_engine.h"
#include "../sbfp_engine/sbfp_config.h"
#include "../line_buffer/line_buffer.h"

/*---------------------------------------------------------------------------
 * AXI-Stream side-channel type for HP port transactions
 *---------------------------------------------------------------------------*/
typedef ap_axiu<AXI_HP_WIDTH, 1, 1, 1>  AxiHpPkt;

/*---------------------------------------------------------------------------
 * Top-level function declaration
 *
 * Synthesise with:
 *   vitis_hls -f run_hls.tcl
 * Target device: xczu9eg-ffvb1156-2-e (ZCU102)
 * Target clock:  5 ns (200 MHz)
 *---------------------------------------------------------------------------*/
void spectra_yolo_top(
    /* AXI4 Master – weight memory (HP0) */
    ap_uint<AXI_HP_WIDTH>  *weight_mem,
    /* AXI4 Master – input activation memory (HP1) */
    ap_uint<AXI_HP_WIDTH>  *act_mem,
    /* AXI4 Master – output memory (HP2) */
    ap_uint<AXI_HP_WIDTH>  *out_mem,
    /* AXI4 Master – bias / layer config memory (HP3) */
    ap_uint<AXI_HP_WIDTH>  *bias_mem,
    /* AXI4-Lite Slave – control registers */
    int                     layer_idx,
    int                     img_w,
    int                     img_h,
    int                     in_ch,
    int                     out_ch,
    int                     use_int4,
    volatile int            *done_tiles
);

#endif /* SPECTRA_YOLO_TOP_H */
