# Vivado implementation flow for SPECTRA-YOLO on ZCU102.
# Requires successful HLS IP export from scripts/run_hls.tcl.

set package_root [file normalize [file join [file dirname [info script]] ".."]]
set zynq_root    [file normalize [file join $package_root ".."]]
set build_root   [file join $package_root "build"]
set export_root  [file join $package_root "exports"]
set report_root  [file join $export_root "vivado_reports"]
set jobs 8
if {[info exists argv] && [llength $argv] > 0} {
  set jobs [lindex $argv 0]
}

file mkdir $build_root
file mkdir $export_root
file mkdir $report_root

set part_name "xczu9eg-ffvb1156-2-e"
set project_name "spectra_yolo_zcu102"
set project_dir [file join $build_root $project_name]
set hls_ip_repo [file join $build_root "hls_spectra_yolo_top/solution1/impl/ip"]

create_project -force $project_name $project_dir -part $part_name

if {[file exists $hls_ip_repo]} {
  set_property ip_repo_paths [list $hls_ip_repo] [current_project]
  update_ip_catalog
} else {
  error "HLS IP repo not found: $hls_ip_repo. Run scripts/run_hls.tcl first."
}

source [file join $package_root "scripts/create_bd_checked.tcl"]
add_files -fileset constrs_1 -norecurse [file join $package_root "scripts/zcu102_constraints_checked.xdc"]
update_compile_order -fileset sources_1
update_compile_order -fileset sim_1

launch_runs synth_1 -jobs $jobs
wait_on_run synth_1
open_run synth_1
report_utilization -hierarchical -file [file join $report_root "post_synth_utilization_hier.rpt"]
report_timing_summary -file [file join $report_root "post_synth_timing_summary.rpt"]

launch_runs impl_1 -to_step write_bitstream -jobs $jobs
wait_on_run impl_1
open_run impl_1

report_utilization -hierarchical -file [file join $report_root "post_route_utilization_hier.rpt"]
report_timing_summary -file [file join $report_root "post_route_timing_summary.rpt"]
report_power -file [file join $report_root "post_route_power.rpt"]
report_clock_utilization -file [file join $report_root "post_route_clock_utilization.rpt"]
report_route_status -file [file join $report_root "post_route_status.rpt"]

write_checkpoint -force [file join $export_root "spectra_yolo_post_route.dcp"]
write_hw_platform -fixed -include_bit -force -file [file join $export_root "spectra_yolo_zcu102.xsa"]

proc collect_bit_files {dir} {
  set files [glob -nocomplain -directory $dir -types f *.bit]
  foreach child_dir [glob -nocomplain -directory $dir -types d *] {
    foreach bit_file [collect_bit_files $child_dir] {
      lappend files $bit_file
    }
  }
  return $files
}

set bit_candidates [collect_bit_files $project_dir]
foreach bit $bit_candidates {
  file copy -force $bit [file join $export_root [file tail $bit]]
}

puts "Vivado implementation completed."
puts "Vivado reports copied to: $report_root"
