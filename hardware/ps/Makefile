#===============================================================================
# Makefile  –  SPECTRA-YOLO PS ARM Application
#
# Target: ARM Cortex-A53 (Xilinx ZCU102, aarch64-linux-gnu)
# Paper:  Section 5.3, H3S scheduler + UIO driver + preprocessing + NMS
#
# Usage (cross-compile on x86 host):
#   make CROSS=aarch64-linux-gnu-
#
# Usage (native compile on ZCU102 running PetaLinux):
#   make
#
# Output:
#   spectra_yolo_ps   –  main inference binary
#   libspectra_ps.a   –  static library (uio_ctrl + h3s + preprocess + nms)
#===============================================================================

CROSS      ?=
CC         := $(CROSS)gcc
AR         := $(CROSS)ar

# Optimisation: -O2 + NEON SIMD for Cortex-A53
CFLAGS     := -O2 -Wall -Wextra \
              -march=armv8-a -mtune=cortex-a53 \
              -DHAVE_NEON=1

LDFLAGS    := -lpthread -lm -ljpeg -lpng

SRCS       := uio_ctrl.c \
              h3s_scheduler.c \
              preprocess.c \
              nms.c

OBJS       := $(SRCS:.c=.o)

LIB        := libspectra_ps.a
BIN        := spectra_yolo_ps

.PHONY: all clean lib

all: $(BIN)

lib: $(LIB)

$(LIB): $(OBJS)
	$(AR) rcs $@ $^

$(BIN): main.o $(LIB)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f $(OBJS) main.o $(LIB) $(BIN)

#===============================================================================
# Vitis / PetaLinux integration targets
#===============================================================================

# Package binaries for deployment on ZCU102 SD card
DEPLOY_DIR ?=

deploy: $(BIN)
	@test -n "$(DEPLOY_DIR)" || { echo "Set DEPLOY_DIR explicitly"; exit 1; }
	cp $(BIN) $(DEPLOY_DIR)/spectra_yolo_ps
	@echo "Deployed to $(DEPLOY_DIR)"

# Opt-in device operation: set ZCU102_HOST=user@host explicitly
ZCU102_HOST ?=
REMOTE_DIR ?= .
run-remote: $(BIN)
	@test -n "$(ZCU102_HOST)" || { echo "Set ZCU102_HOST=user@host explicitly"; exit 1; }
	scp $(BIN) $(ZCU102_HOST):$(REMOTE_DIR)/
	ssh $(ZCU102_HOST) "$(REMOTE_DIR)/spectra_yolo_ps --n-frames 100 --input /dev/video0"
