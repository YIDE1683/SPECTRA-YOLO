/*=============================================================================
 * uio_ctrl.c  –  UIO PL Register Access Implementation
 *
 * Paper: Section 5.3, Theorem 5 (H3S Level 1 handshake)
 * Measured PS→PL register write latency: ~300 ns (L1 cache, no syscall)
 * UIO IRQ latency (interrupt path):      ~800 ns
 * Total PS-PL handshake budget:          <1 µs  [Section 5.3]
 *=============================================================================*/
#include "uio_ctrl.h"

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <errno.h>

/*---------------------------------------------------------------------------
 * uio_open()
 *---------------------------------------------------------------------------*/
UioCtx *uio_open(const char *dev_path, uint32_t phys_base, size_t map_size)
{
    UioCtx *ctx = (UioCtx *)malloc(sizeof(UioCtx));
    if (!ctx) return NULL;

    ctx->fd = open(dev_path, O_RDWR | O_SYNC);
    if (ctx->fd < 0) {
        fprintf(stderr, "[uio_ctrl] Cannot open %s: %s\n",
                dev_path, strerror(errno));
        free(ctx);
        return NULL;
    }

    ctx->base = mmap(NULL, map_size, PROT_READ | PROT_WRITE,
                     MAP_SHARED, ctx->fd, 0);
    if (ctx->base == MAP_FAILED) {
        fprintf(stderr, "[uio_ctrl] mmap failed: %s\n", strerror(errno));
        close(ctx->fd);
        free(ctx);
        return NULL;
    }

    ctx->map_size  = map_size;
    ctx->phys_base = phys_base;
    return ctx;
}

/*---------------------------------------------------------------------------
 * uio_close()
 *---------------------------------------------------------------------------*/
void uio_close(UioCtx *ctx)
{
    if (!ctx) return;
    if (ctx->base && ctx->base != MAP_FAILED)
        munmap(ctx->base, ctx->map_size);
    if (ctx->fd >= 0)
        close(ctx->fd);
    free(ctx);
}

/*---------------------------------------------------------------------------
 * uio_accel_start()
 *
 * Write layer parameters to AXI4-Lite registers then assert ap_start.
 * The accelerator latches all parameters on the rising edge of ap_start.
 * Register writes complete before ap_start due to DSB in uio_write32().
 *---------------------------------------------------------------------------*/
void uio_accel_start(UioCtx *ctx,
                     int layer_idx, int img_w, int img_h,
                     int in_ch, int out_ch, int use_int4)
{
    /* Step 1: wait for idle before issuing new command */
    uint32_t status;
    int retries = 10000;
    do {
        status = uio_read32(ctx, REG_AP_CTRL);
        if (--retries == 0) {
            fprintf(stderr, "[uio_ctrl] Timeout waiting for AP_IDLE\n");
            return;
        }
    } while (!(status & AP_IDLE));

    /* Step 2: write layer configuration */
    uio_write32(ctx, REG_LAYER_IDX, (uint32_t)layer_idx);
    uio_write32(ctx, REG_IMG_W,     (uint32_t)img_w);
    uio_write32(ctx, REG_IMG_H,     (uint32_t)img_h);
    uio_write32(ctx, REG_IN_CH,     (uint32_t)in_ch);
    uio_write32(ctx, REG_OUT_CH,    (uint32_t)out_ch);
    uio_write32(ctx, REG_USE_INT4,  (uint32_t)use_int4);

    /* Step 3: assert ap_start (clears ap_done automatically in HLS IP) */
    uio_write32(ctx, REG_AP_CTRL, AP_START);
}

/*---------------------------------------------------------------------------
 * uio_accel_wait_done()
 *
 * Busy-poll ap_done bit. Appropriate for short layers (<50 µs).
 * For layers >50 µs, use uio_accel_wait_irq() to free the CPU.
 *---------------------------------------------------------------------------*/
void uio_accel_wait_done(UioCtx *ctx)
{
    while (!(uio_read32(ctx, REG_AP_CTRL) & AP_DONE)) {
        /* tight poll – CPU is reserved for H3S Level 2 latency hiding */
        __asm__ volatile ("yield" ::: "memory");
    }
}

/*---------------------------------------------------------------------------
 * uio_accel_wait_irq()
 *
 * Block on UIO interrupt event. The HLS IP asserts interrupt when ap_done.
 * Linux UIO driver delivers this as a 4-byte read on /dev/uioN.
 * CPU is yielded to OS scheduler while waiting (H3S Level 3, Theorem 5).
 *
 * IRQ must be re-enabled after each event by writing 1 to /dev/uioN.
 *---------------------------------------------------------------------------*/
void uio_accel_wait_irq(UioCtx *ctx)
{
    uint32_t irq_count;
    /* Enable interrupt (unmask) */
    uint32_t unmask = 1;
    if (write(ctx->fd, &unmask, sizeof(unmask)) != sizeof(unmask)) {
        fprintf(stderr, "[uio_ctrl] IRQ enable write failed: %s\n",
                strerror(errno));
        /* Fallback to poll */
        uio_accel_wait_done(ctx);
        return;
    }
    /* Block until IRQ fires */
    if (read(ctx->fd, &irq_count, sizeof(irq_count)) != sizeof(irq_count)) {
        fprintf(stderr, "[uio_ctrl] IRQ read failed: %s\n", strerror(errno));
    }
    /* Clear ap_done by reading ap_ctrl (HLS auto-clear on read) */
    (void)uio_read32(ctx, REG_AP_CTRL);
}
