# Checked Vivado Block Design for SPECTRA-YOLO on ZCU102.
#
# This script is derived from ../../vivado/scripts/create_bd.tcl but fixes
# the clocking mismatch identified in reports/source_audit.md: the HLS
# accelerator and its AXI fabric are clocked from pl_clk0 at 200 MHz so that a
# successful implementation report can support the manuscript's PL-fabric claim.
#
# The original manuscript also discusses a 300 MHz AXI-HP bandwidth context.
# A separate multi-clock design would need explicit AXI clock converters because
# the HLS IP exposes one ap_clk. This checked batch design therefore prioritizes
# a timing-consistent 200 MHz accelerator implementation for report generation.

set part_name "xczu9eg-ffvb1156-2-e"
set bd_name   "spectra_yolo_bd"

if {[catch {current_project} current_project_name] || $current_project_name eq ""} {
    set_part $part_name
}

create_bd_design $bd_name

proc configure_zynq_ps_ports {ps_cell} {
    set_property -dict [list \
        CONFIG.PSU__CRL_APB__PL0_REF_CTRL__FREQMHZ {200} \
        CONFIG.PSU__PL_CLK0_BUF                   {TRUE} \
        CONFIG.PSU__USE__M_AXI_GP0                {1} \
        CONFIG.PSU__USE__M_AXI_GP1                {0} \
        CONFIG.PSU__USE__M_AXI_GP2                {0} \
        CONFIG.PSU__USE__S_AXI_GP2                {1} \
        CONFIG.PSU__USE__S_AXI_GP3                {1} \
        CONFIG.PSU__USE__S_AXI_GP4                {1} \
        CONFIG.PSU__USE__S_AXI_GP5                {1} \
        CONFIG.PSU__USE__IRQ0                     {1} \
        CONFIG.PSU__HIGH_ADDRESS__ENABLE          {1} \
    ] $ps_cell
}

set zynq_ps [create_bd_cell -type ip -vlnv xilinx.com:ip:zynq_ultra_ps_e:3.5 zynq_ps_0]

configure_zynq_ps_ports $zynq_ps

apply_bd_automation -rule xilinx.com:bd_rule:zynq_ultra_ps_e \
    -config {apply_board_preset 1} $zynq_ps

configure_zynq_ps_ports $zynq_ps

set accel [create_bd_cell -type ip \
    -vlnv xilinx.com:hls:spectra_yolo_top:1.0 spectra_yolo_top_0]

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

# Clock all HLS-facing interfaces at 200 MHz to match the report target.
connect_bd_net \
    [get_bd_pins $zynq_ps/pl_clk0] \
    [get_bd_pins $axi_lite_ic/ACLK] \
    [get_bd_pins $axi_lite_ic/S00_ACLK] \
    [get_bd_pins $axi_lite_ic/M00_ACLK] \
    [get_bd_pins $accel/ap_clk]

set reset_sync [create_bd_cell -type ip \
    -vlnv xilinx.com:ip:proc_sys_reset:5.0 proc_sys_reset_0]
puts "proc_sys_reset C_EXT_RESET_HIGH=[get_property CONFIG.C_EXT_RESET_HIGH $reset_sync]"

set reset_inv [create_bd_cell -type ip \
    -vlnv xilinx.com:ip:util_vector_logic:2.0 resetn_to_reset]
set_property -dict [list \
    CONFIG.C_OPERATION {not} \
    CONFIG.C_SIZE {1} \
] $reset_inv

set reset_locked [create_bd_cell -type ip \
    -vlnv xilinx.com:ip:xlconstant:1.1 reset_locked_const]
set_property CONFIG.CONST_WIDTH {1} $reset_locked
set_property CONFIG.CONST_VAL {1} $reset_locked

connect_bd_net \
    [get_bd_pins $zynq_ps/pl_resetn0] \
    [get_bd_pins $reset_inv/Op1]

connect_bd_net \
    [get_bd_pins $reset_inv/Res] \
    [get_bd_pins $reset_sync/ext_reset_in]

connect_bd_net \
    [get_bd_pins $zynq_ps/pl_clk0] \
    [get_bd_pins $reset_sync/slowest_sync_clk]

connect_bd_net \
    [get_bd_pins $reset_locked/dout] \
    [get_bd_pins $reset_sync/dcm_locked]

connect_bd_net \
    [get_bd_pins $reset_sync/peripheral_aresetn] \
    [get_bd_pins $axi_lite_ic/ARESETN] \
    [get_bd_pins $axi_lite_ic/S00_ARESETN] \
    [get_bd_pins $axi_lite_ic/M00_ARESETN] \
    [get_bd_pins $accel/ap_rst_n]

foreach ps_clk_pin {
    maxihpm0_fpd_aclk
    maxihpm0_lpd_aclk
    saxihp0_fpd_aclk
    saxihp1_fpd_aclk
    saxihp2_fpd_aclk
    saxihp3_fpd_aclk
} {
    set clk_pin [get_bd_pins -quiet $zynq_ps/$ps_clk_pin]
    if {[llength $clk_pin]} {
        connect_bd_net [get_bd_pins $zynq_ps/pl_clk0] $clk_pin
    }
}

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
        [get_bd_pins $zynq_ps/pl_clk0] \
        [get_bd_pins $sc/aclk]

    connect_bd_net \
        [get_bd_pins $reset_sync/peripheral_aresetn] \
        [get_bd_pins $sc/aresetn]
}

set irq_pins [get_bd_pins -quiet $accel/interrupt]
if {[llength $irq_pins]} {
    set xlconcat [create_bd_cell -type ip \
        -vlnv xilinx.com:ip:xlconcat:2.1 xlconcat_irq]
    set_property CONFIG.NUM_PORTS {1} $xlconcat

    connect_bd_net \
        [get_bd_pins $accel/interrupt] \
        [get_bd_pins $xlconcat/In0]

    connect_bd_net \
        [get_bd_pins $xlconcat/dout] \
        [get_bd_pins $zynq_ps/pl_ps_irq0]
}

assign_bd_address
set ctrl_seg [get_bd_addr_segs -quiet $accel/s_axi_ctrl/Reg]
if {[llength $ctrl_seg]} {
    if {![catch {get_property offset $ctrl_seg} ctrl_offset] &&
        ![catch {get_property range $ctrl_seg} ctrl_range]} {
        puts "Vivado-assigned control segment: offset=$ctrl_offset range=$ctrl_range"
    }
}

validate_bd_design
save_bd_design

generate_target all [get_files ${bd_name}.bd]
set wrapper_files [make_wrapper -files [get_files ${bd_name}.bd] -top]
if {[llength $wrapper_files]} {
    add_files -norecurse $wrapper_files
    set_property top ${bd_name}_wrapper [current_fileset]
}

puts "Checked block design '${bd_name}' created for 200 MHz accelerator implementation."
