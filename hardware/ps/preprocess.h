/*=============================================================================
 * preprocess.h  –  PS-side Image Preprocessing
 *
 * Paper: Section 5.3 – "PS preprocessing: decode → resize → normalise"
 *        H3S Level 1 Stage A, measured T_pre = 4.2 ms (ARM NEON @ 1.2 GHz)
 *
 * Pipeline:
 *   Input: JPEG/PNG/raw MIPI frame (any resolution)
 *   Step 1: Decode (libjpeg / V4L2 MIPI capture)
 *   Step 2: Letterbox resize to 640×640 (bilinear, NEON-vectorised)
 *   Step 3: Normalize to float32 in range [0.0, 1.0]:
 *             pixel_f = pixel_u8 / 255.0f
 *   Step 4: SBFP host-side quantisation to INT8 or INT4
 *             (mirrors Theorem 2, Eq. 22 at host side before DMA transfer)
 *
 * NEON optimisation:
 *   Step 2-3 use ARM NEON 128-bit SIMD (float32x4_t / uint8x16_t) to
 *   process 16 pixels/cycle on Cortex-A53. Achieves 4.2 ms for 640×640×3.
 *=============================================================================*/
#ifndef PREPROCESS_H
#define PREPROCESS_H

#include <stdint.h>
#include <stddef.h>

/* Input image maximum dimensions */
#define MAX_SRC_W   4096
#define MAX_SRC_H   4096

/* Target inference resolution (SLF-YOLOv8 input) */
#define INF_W       640
#define INF_H       640
#define INF_C       3

/*---------------------------------------------------------------------------
 * LetterboxInfo: records padding offsets for bbox coordinate recovery
 * Used by NMS to map output coordinates back to original image space.
 *---------------------------------------------------------------------------*/
typedef struct {
    float scale;   /* uniform scale factor: inf_dim / src_dim              */
    int   pad_w;   /* horizontal padding (pixels) added per side           */
    int   pad_h;   /* vertical padding   (pixels) added per side           */
    int   src_w;   /* original source image width                          */
    int   src_h;   /* original source image height                         */
} LetterboxInfo;

/*---------------------------------------------------------------------------
 * API
 *---------------------------------------------------------------------------*/

/**
 * preprocess_image() – full preprocessing pipeline: decode → resize → normalize
 *
 * @src_path   : path to image file (JPEG/PNG). NULL = capture from V4L2.
 * @out_float  : output float32 buffer [INF_H × INF_W × INF_C], row-major, CHW
 * @out_w      : output width  (= INF_W = 640)
 * @out_h      : output height (= INF_H = 640)
 *
 * Also fills *lb_info if non-NULL, for coordinate recovery in NMS.
 * Returns 0 on success, -1 on failure.
 */
int preprocess_image(const char *src_path,
                     float *out_float, int out_w, int out_h);

/**
 * preprocess_image_ex() – same as above but also returns letterbox info
 */
int preprocess_image_ex(const char *src_path,
                         float *out_float, int out_w, int out_h,
                         LetterboxInfo *lb_info);

/**
 * preprocess_resize_letterbox() – letterbox bilinear resize
 *
 * Resizes src (uint8, HWC BGR) to dst (uint8, HWC BGR, out_w × out_h)
 * with black padding to preserve aspect ratio.
 *
 * NEON-optimised: processes 16 pixels/cycle on Cortex-A53.
 */
void preprocess_resize_letterbox(const uint8_t *src, int src_w, int src_h,
                                  uint8_t *dst, int out_w, int out_h,
                                  LetterboxInfo *lb_info);

/**
 * preprocess_normalize() – uint8 HWC BGR → float32 CHW, scale to [0,1]
 *
 * Converts HWC (height × width × 3) uint8 to CHW (3 × height × width) float32.
 * Divides by 255.0f using NEON vrecpeq_f32 reciprocal approximation.
 *
 * @src      : uint8 input [H × W × 3], BGR channel order
 * @dst      : float32 output [3 × H × W], RGB channel order (B↔R swap)
 * @n_pixels : H × W
 */
void preprocess_normalize(const uint8_t *src, float *dst, int n_pixels);

/**
 * preprocess_sbfp_quantise() – float32 → INT8 SBFP (host-side, layer 0)
 *
 * Mirrors sbfp_quantise_8b<0>() from sbfp_engine.h for the PL stem layer.
 * Applied to the pre-processed float buffer before HP1 DMA transfer.
 * Uses SBFP_EXPONENTS_8B[0] = -6 (Eq. 22, stem layer exponent).
 *
 * @in_float  : float32 input  [n_elements]
 * @out_int8  : INT8   output  [n_elements]
 * @n_elements: total element count (INF_H × INF_W × INF_C)
 */
void preprocess_sbfp_quantise(const float *in_float, int8_t *out_int8,
                               int n_elements);

#endif /* PREPROCESS_H */
