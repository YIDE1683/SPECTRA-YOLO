/*=============================================================================
 * uio_ctrl.h  –  UIO-based PL Register Access (<1 µs latency)
 *
 * Paper: Section 5.3 – "PS-PL register handshake via /dev/uio0"
 *        Theorem 5, H3S Level 1: ps_pre → pl_infer signal triggers <1 µs
 *
 * Linux UIO driver maps the AXI4-Lite register block at 0xA000_0000
 * (assigned in create_bd.tcl) into userspace as /dev/uio0.
 *
 * Register map (matches spectra_yolo_top AXI4-Lite ctrl bundle):
 *   0x00  ap_ctrl     : bit0=ap_start, bit1=ap_done, bit2=ap_idle
 *   0x04  layer_idx   : layer index (0-22)
 *   0x08  img_w       : feature map width
 *   0x0C  img_h       : feature map height
 *   0x10  in_ch       : input channels
 *   0x14  out_ch      : output channels
 *   0x18  use_int4    : 1=INT4 mode, 0=INT8 mode
 *   0x1C  done_tiles  : completed tile count (read-only)
 *
 * Usage:
 *   UioCtx *ctx = uio_open("/dev/uio0", 0xA0000000, 64*1024);
 *   uio_write32(ctx, REG_AP_CTRL, AP_START);
 *   while (!(uio_read32(ctx, REG_AP_CTRL) & AP_DONE));
 *   uio_close(ctx);
 *=============================================================================*/
#ifndef UIO_CTRL_H
#define UIO_CTRL_H

#include <stdint.h>
#include <stddef.h>

/* Register offsets (bytes) */
#define REG_AP_CTRL     0x00
#define REG_LAYER_IDX   0x04
#define REG_IMG_W       0x08
#define REG_IMG_H       0x0C
#define REG_IN_CH       0x10
#define REG_OUT_CH      0x14
#define REG_USE_INT4    0x18
#define REG_DONE_TILES  0x1C

/* ap_ctrl bit masks */
#define AP_START        (1u << 0)
#define AP_DONE         (1u << 1)
#define AP_IDLE         (1u << 2)
#define AP_READY        (1u << 3)

/* Physical base address (must match create_bd.tcl assign_bd_address) */
#define ACCEL_PHYS_BASE  0xA0000000UL
#define ACCEL_MAP_SIZE   (64 * 1024)   /* 64 KB */

/*---------------------------------------------------------------------------
 * UioCtx: opaque handle to a UIO-mapped register block
 *---------------------------------------------------------------------------*/
typedef struct {
    int      fd;          /* /dev/uioN file descriptor                      */
    void    *base;        /* mmap base pointer                              */
    size_t   map_size;    /* mmap size (bytes)                              */
    uint32_t phys_base;   /* physical base address (for DMA address calc)   */
} UioCtx;

/*---------------------------------------------------------------------------
 * API
 *---------------------------------------------------------------------------*/

/**
 * uio_open() – open and mmap a UIO device
 * @dev_path   : e.g. "/dev/uio0"
 * @phys_base  : physical base address (used for sanity check)
 * @map_size   : mmap size in bytes
 * Returns pointer on success, NULL on failure (errno set).
 */
UioCtx *uio_open(const char *dev_path, uint32_t phys_base, size_t map_size);

/**
 * uio_close() – unmap and close the UIO device
 */
void uio_close(UioCtx *ctx);

/**
 * uio_write32() – write a 32-bit value to a register offset
 * Latency: ~300 ns (L1 cache hit, no syscall) [Section 5.3]
 */
static inline void uio_write32(UioCtx *ctx, uint32_t offset, uint32_t val)
{
    volatile uint32_t *reg = (volatile uint32_t *)
                             ((uint8_t *)ctx->base + offset);
    *reg = val;
    __asm__ volatile ("dsb sy" ::: "memory");  /* ARM data sync barrier */
}

/**
 * uio_read32() – read a 32-bit value from a register offset
 */
static inline uint32_t uio_read32(UioCtx *ctx, uint32_t offset)
{
    __asm__ volatile ("dsb sy" ::: "memory");
    volatile uint32_t *reg = (volatile uint32_t *)
                             ((uint8_t *)ctx->base + offset);
    return *reg;
}

/**
 * uio_accel_start() – configure all layer parameters and assert ap_start
 * Returns immediately; poll ap_done or wait for IRQ.
 */
void uio_accel_start(UioCtx *ctx,
                     int layer_idx, int img_w, int img_h,
                     int in_ch, int out_ch, int use_int4);

/**
 * uio_accel_wait_done() – poll ap_done bit (busy-wait, <50 µs typical)
 * For longer layers use uio_accel_wait_irq() instead.
 */
void uio_accel_wait_done(UioCtx *ctx);

/**
 * uio_accel_wait_irq() – block on UIO interrupt (zero CPU overhead during wait)
 * Linux UIO driver delivers the tile-done interrupt as a readable event on fd.
 */
void uio_accel_wait_irq(UioCtx *ctx);

#endif /* UIO_CTRL_H */
