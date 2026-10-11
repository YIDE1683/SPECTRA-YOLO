# Checked constraint boundary for the SPECTRA-YOLO ZCU102 BD flow.
#
# The implementation top is an IP Integrator wrapper driven by the Zynq
# UltraScale+ PS. Clock constraints for pl_clk0 and PS-related interfaces are
# emitted by the generated PS/IP XDC files under the BD output tree.
#
# Earlier guarded Tcl-style constraints were intentionally removed because
# Vivado parses XDC with a restricted command set and reports `if` blocks as
# critical warnings. Keep this file as a documented no-op unless a future flow
# exposes additional top-level PL pins that need explicit package/timing
# constraints.
#
# Documentation boundary:
# SBFP-4b + INT4 activation broadcast uses the corrected 3-product mapping:
# 512 logical lanes / 3 products per DSP -> ceil(512/3) = 172 DSP48E2.
# The older 128-DSP 4-way packing comment is not used in this checked flow.
