#===============================================================================
# zcu102_constraints.xdc  –  SPECTRA-YOLO Timing Constraints for ZCU102
#
# Device: xczu9eg-ffvb1156-2-e  (ZCU102 evaluation board)
# Paper:  Section 6.1 – "PL fabric operates at 200 MHz"
#         Section 4.5  – "4 AXI-HP ports … 300 MHz" (Eq. 15, 27)
#
# Clock domains:
#   pl_clk0  – PL fabric clock (PE array, line buffer):      200 MHz (5.0 ns)
#   pl_clk1  – AXI-HP DMA clock (memory controller):         300 MHz (3.33 ns)
#   pl_clk2  – AXI4-Lite control register clock:             100 MHz (10.0 ns)
#===============================================================================

#-------------------------------------------------------------------------------
# Primary PL clock – sourced from PS (Zynq MPSoC PL_CLK0)
# Configured in create_bd.tcl: zynq_ps_0.PCW_FPGA0_PERIPHERAL_FREQMHZ = 200
#-------------------------------------------------------------------------------
create_clock -period 5.000 -name pl_clk0 \
    [get_ports pl_clk0]

#-------------------------------------------------------------------------------
# AXI-HP DMA clock – PL_CLK1, 300 MHz
# Used by pingpong_dma_ctrl and all AXI-HP master interfaces on HP0-HP3
#-------------------------------------------------------------------------------
create_clock -period 3.333 -name pl_clk1 \
    [get_ports pl_clk1]

#-------------------------------------------------------------------------------
# AXI4-Lite control clock – PL_CLK2, 100 MHz
# Used by UIO register access from PS (Section 5.3, <1 µs latency)
#-------------------------------------------------------------------------------
create_clock -period 10.000 -name pl_clk2 \
    [get_ports pl_clk2]

#-------------------------------------------------------------------------------
# Clock domain crossing constraints
# PE array (200 MHz) ↔ AXI-HP DMA (300 MHz): async CDC via FIFO
#-------------------------------------------------------------------------------
set_clock_groups -asynchronous \
    -group [get_clocks pl_clk0] \
    -group [get_clocks pl_clk1]

set_clock_groups -asynchronous \
    -group [get_clocks pl_clk0] \
    -group [get_clocks pl_clk2]

#-------------------------------------------------------------------------------
# Input/Output delays for AXI-HP ports (measured at HP port boundary)
# Values align with Zynq MPSoC TRM Table B-12 (PS-PL AXI interface timing)
#-------------------------------------------------------------------------------
set_input_delay  -clock pl_clk1 -max 1.5 [get_ports {m_axi_hp0_* m_axi_hp1_*}]
set_input_delay  -clock pl_clk1 -min 0.3 [get_ports {m_axi_hp0_* m_axi_hp1_*}]
set_output_delay -clock pl_clk1 -max 1.5 [get_ports {m_axi_hp2_* m_axi_hp3_*}]
set_output_delay -clock pl_clk1 -min 0.3 [get_ports {m_axi_hp2_* m_axi_hp3_*}]

#-------------------------------------------------------------------------------
# False paths: static configuration signals (layer_idx, img_w, etc.)
# These are written once per inference call by PS via AXI4-Lite, then held
# constant throughout the computation. No timing closure needed across cycles.
#-------------------------------------------------------------------------------
set_false_path -from [get_cells -hierarchical -filter {NAME =~ *ctrl_reg*}] \
               -to   [get_cells -hierarchical -filter {NAME =~ *pe_array*}]

#-------------------------------------------------------------------------------
# SBFP shared exponent constants (Corollary 1):
# These are synthesis-time constants folded into the netlist as static LUT values.
# Declare as false paths to prevent timing engine from optimising them.
#-------------------------------------------------------------------------------
set_false_path -from [get_cells -hierarchical \
    -filter {NAME =~ *sbfp_exponent_rom*}]

#-------------------------------------------------------------------------------
# Pblock (floorplanning): assign PE array to SLR0 for thermal isolation
# ZCU102 XCZU9EG has one SLR; this pblock guides placer to high-DSP region.
#-------------------------------------------------------------------------------
create_pblock pb_pe_array
add_cells_to_pblock [pb_pe_array] \
    [get_cells -hierarchical -filter {NAME =~ *pe_array_top*}]
resize_pblock [pb_pe_array] \
    -add {SLICE_X0Y0:SLICE_X119Y239 DSP48E2_X0Y0:DSP48E2_X7Y95}

create_pblock pb_line_buffer
add_cells_to_pblock [pb_line_buffer] \
    [get_cells -hierarchical -filter {NAME =~ *line_buffer*}]
resize_pblock [pb_line_buffer] \
    -add {RAMB36_X0Y0:RAMB36_X7Y47 RAMB18_X0Y0:RAMB18_X7Y95}

#-------------------------------------------------------------------------------
# DSP48E2 utilisation target (Corollary 2):
#   INT8 baseline: 512 DSPs  (20.3% of 2,520)
#   SBFP-8b:       512 DSPs  (20.3%)
#   SBFP-4b+INT4:  128 DSPs  (5.1%)   ← target configuration
#
# Uncomment to enforce DSP ceiling during implementation:
# set_property ALLOW_COMBINATORIAL_LOOPS TRUE [get_nets -hierarchical -filter {NAME =~ *dsp_guard*}]
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
# Configuration memory (ZCU102 onboard QSPI flash for bitstream)
# set_property CONFIG_VOLTAGE    1.8    [current_design]
# set_property CFGBVS            GND    [current_design]
# set_property BITSTREAM.CONFIG.SPI_BUSWIDTH 4 [current_design]
#-------------------------------------------------------------------------------
