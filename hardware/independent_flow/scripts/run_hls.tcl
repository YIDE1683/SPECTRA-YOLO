# Vitis HLS batch flow for SPECTRA-YOLO.
# Sources are resolved relative to this release; generated products stay
# under independent_flow/build and independent_flow/exports.

set package_root [file normalize [file join [file dirname [info script]] ".."]]
set zynq_root    [file normalize [file join $package_root ".."]]
set build_root   [file join $package_root "build"]
set export_root  [file join $package_root "exports"]
set log_root     [file join $package_root "logs"]

file mkdir $build_root
file mkdir $export_root
file mkdir $log_root

set project_dir [file join $build_root "hls_spectra_yolo_top"]
set top_name "spectra_yolo_top"

open_project -reset $project_dir
set_top $top_name

add_files [file join $zynq_root "hls/conv_top/spectra_yolo_top.cpp"] \
  -cflags "-std=c++11 -I[file join $zynq_root hls/conv_top] -I[file join $zynq_root hls/pe_array] -I[file join $zynq_root hls/line_buffer] -I[file join $zynq_root hls/sbfp_engine]"
add_files [file join $zynq_root "hls/pe_array/pe_array.cpp"] \
  -cflags "-std=c++11 -I[file join $zynq_root hls/conv_top] -I[file join $zynq_root hls/pe_array] -I[file join $zynq_root hls/line_buffer] -I[file join $zynq_root hls/sbfp_engine]"
add_files [file join $zynq_root "hls/line_buffer/line_buffer.cpp"] \
  -cflags "-std=c++11 -I[file join $zynq_root hls/conv_top] -I[file join $zynq_root hls/pe_array] -I[file join $zynq_root hls/line_buffer] -I[file join $zynq_root hls/sbfp_engine]"

open_solution -reset "solution1"
set_part {xczu9eg-ffvb1156-2-e}
create_clock -period 5.000 -name default

csynth_design
export_design -format ip_catalog -rtl verilog

set rpt_src [file join $project_dir "solution1/syn/report"]
set rpt_dst [file join $export_root "hls_reports"]
file mkdir $rpt_dst
foreach rpt [glob -nocomplain -directory $rpt_src *.rpt *.xml] {
  file copy -force $rpt $rpt_dst
}

puts "HLS synthesis completed."
puts "HLS reports copied to: $rpt_dst"
puts "HLS IP should be under: [file join $project_dir solution1 impl ip]"
