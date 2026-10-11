#===============================================================================
# create_bd.tcl  –  Vivado Block Design for SPECTRA-YOLO on ZCU102
#
# Paper: Section 5.1 (Heterogeneous SoC Architecture), Fig. 6(a)
#
# Block design topology:
#   Zynq UltraScale+ MPSoC (PS)
#     └─ HP0-HP3: 4 × 128-bit AXI-HP slave ports @ 300 MHz
#          ↕ AXI SmartConnect (4-to-1 arbitration)
#     └─ spectra_yolo_top IP (HLS-generated, 200 MHz PL)
#          ├─ AXI4-Lite ctrl  ← PS GP0 (100 MHz)
#          └─ IRQ → PS IRQ[0] (tile-done interrupt for H3S Level 1)
#
# Usage:
#   vivado -mode batch -source create_bd.tcl
#   (or source in Vivado Tcl Console after opening a project)
#===============================================================================

# ---- Project / device settings ----
set part_name "xczu9eg-ffvb1156-2-e"
set bd_name   "spectra_yolo_bd"

set_part $part_name

# Create block design
create_bd_design $bd_name

#===============================================================================
# 1. Add Zynq UltraScale+ MPSoC
#===============================================================================
set zynq_ps [create_bd_cell -type ip -vlnv xilinx.com:ip:zynq_ultra_ps_e:3.5 zynq_ps_0]

# Apply ZCU102 preset (board preset enables DDR4, UART, USB, Ethernet)
set_property -dict [list \
    CONFIG.PSU_BOARD_PARAMS              {} \
    CONFIG.PCW_FPGA0_PERIPHERAL_FREQMHZ {200} \
    CONFIG.PCW_FPGA1_PERIPHERAL_FREQMHZ {300} \
    CONFIG.PCW_FPGA2_PERIPHERAL_FREQMHZ {100} \
    CONFIG.PSU__USE__M_AXI_GP0          {1}  \
    CONFIG.PSU__USE__S_AXI_GP0          {1}  \
    CONFIG.PSU__USE__S_AXI_GP1          {1}  \
    CONFIG.PSU__USE__S_AXI_GP2          {1}  \
    CONFIG.PSU__USE__S_AXI_GP3          {1}  \
    CONFIG.PSU__USE__IRQ0               {1}  \
    CONFIG.PSU__HIGH_ADDRESS__ENABLE    {1}  \
] $zynq_ps

apply_bd_automation -rule xilinx.com:bd_rule:zynq_ultra_ps_e \
    -config {apply_board_preset 1} $zynq_ps

#===============================================================================
# 2. Add SPECTRA-YOLO Accelerator IP (HLS-generated)
#    Run HLS synthesis first:  vitis_hls -f run_hls.tcl
#    Then add generated IP repo to Vivado:
#    set_property ip_repo_paths {../hls/conv_top/hls_prj/solution1/impl/ip} [current_project]
#===============================================================================
set accel [create_bd_cell -type ip \
    -vlnv xilinx.com:hls:spectra_yolo_top:1.0 spectra_yolo_top_0]

#===============================================================================
# 3. AXI4-Lite control connection: PS GP0 → accelerator ctrl
#    Clock: PL_CLK2 (100 MHz)
#===============================================================================
set axi_lite_ic [create_bd_cell -type ip \
    -vlnv xilinx.com:ip:axi_interconnect:2.1 axi_lite_ic]
set_property CONFIG.NUM_SI {1} $axi_lite_ic
set_property CONFIG.NUM_MI {1} $axi_lite_ic

connect_bd_intf_net \
    [get_bd_intf_pins $zynq_ps/M_AXI_HPM0_FPD] \
    [get_bd_intf_pins $axi_lite_ic/S00_AXI]

connect_bd_intf_net \
    [get_bd_intf_pins $axi_lite_ic/M00_AXI] \
    [get_bd_intf_pins $accel/s_axi_ctrl]

connect_bd_net \
    [get_bd_pins $zynq_ps/pl_clk2] \
    [get_bd_pins $axi_lite_ic/ACLK] \
    [get_bd_pins $axi_lite_ic/S00_ACLK] \
    [get_bd_pins $axi_lite_ic/M00_ACLK] \
    [get_bd_pins $accel/ap_clk]

connect_bd_net \
    [get_bd_pins $zynq_ps/pl_resetn0] \
    [get_bd_pins $axi_lite_ic/ARESETN] \
    [get_bd_pins $axi_lite_ic/S00_ARESETN] \
    [get_bd_pins $axi_lite_ic/M00_ARESETN] \
    [get_bd_pins $accel/ap_rst_n]

#===============================================================================
# 4. AXI4 HP connections: accelerator ↔ PS HP0-HP3
#    Using AXI SmartConnect for width conversion + arbitration
#    Clock: PL_CLK1 (300 MHz) – matches HP port maximum frequency
#===============================================================================
foreach {hp_idx bundle_name} {0 HP0 1 HP1 2 HP2 3 HP3} {
    set sc [create_bd_cell -type ip \
        -vlnv xilinx.com:ip:smartconnect:1.0 smartconnect_hp${hp_idx}]
    set_property CONFIG.NUM_SI {1} $sc
    set_property CONFIG.NUM_MI {1} $sc

    connect_bd_intf_net \
        [get_bd_intf_pins $accel/m_axi_${bundle_name}] \
        [get_bd_intf_pins $sc/S00_AXI]

    connect_bd_intf_net \
        [get_bd_intf_pins $sc/M00_AXI] \
        [get_bd_intf_pins $zynq_ps/S_AXI_HP${hp_idx}_FPD]

    connect_bd_net \
        [get_bd_pins $zynq_ps/pl_clk1] \
        [get_bd_pins $sc/aclk]

    connect_bd_net \
        [get_bd_pins $zynq_ps/pl_resetn0] \
        [get_bd_pins $sc/aresetn]
}

#===============================================================================
# 5. Interrupt: tile-done IRQ → PS IRQ[0]
#    Used by H3S Level 1 scheduler to detect PL inference completion (Section 5.2)
#===============================================================================
set xlconcat [create_bd_cell -type ip \
    -vlnv xilinx.com:ip:xlconcat:2.1 xlconcat_irq]
set_property CONFIG.NUM_PORTS {1} $xlconcat

connect_bd_net \
    [get_bd_pins $accel/interrupt] \
    [get_bd_pins $xlconcat/In0]

connect_bd_net \
    [get_bd_pins $xlconcat/dout] \
    [get_bd_pins $zynq_ps/pl_ps_irq0]

#===============================================================================
# 6. Address mapping (Zynq MPSoC 64-bit address space)
#    Accelerator ctrl registers: 0xA000_0000 (64 KB)
#    HP port base addresses set by PS at runtime (via H3S scheduler)
#===============================================================================
assign_bd_address
set_property offset 0xA0000000 \
    [get_bd_addr_segs {$accel/s_axi_ctrl/Reg}]
set_property range 64K \
    [get_bd_addr_segs {$accel/s_axi_ctrl/Reg}]

#===============================================================================
# 7. Validate and save
#===============================================================================
validate_bd_design
save_bd_design

# Generate output products and wrapper
make_wrapper -files [get_files ${bd_name}.bd] -top
add_files -norecurse [glob ./${bd_name}_wrapper.v]
set_property top ${bd_name}_wrapper [current_fileset]

puts "Block design '${bd_name}' created successfully."
puts "Next: Run Synthesis → Implementation → Generate Bitstream"
puts "      or: launch_runs impl_1 -to_step write_bitstream -jobs 8"
