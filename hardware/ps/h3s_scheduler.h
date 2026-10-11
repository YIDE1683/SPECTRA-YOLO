/*=============================================================================
 * h3s_scheduler.h  –  Heterogeneous Hierarchical Scheduling (H3S)
 *
 * Paper: Section 5.2, Theorem 5, Eq. 29-33
 *
 * H3S implements a 3-level pipeline to fully overlap PS and PL work:
 *
 *   Level 1 (Frame-level, Theorem 5, Eq. 29):
 *     Frame N:   [PS_pre] → [PL_infer] → [PS_post]
 *     Frame N+1:           [PS_pre]    → [PL_infer] → [PS_post]
 *     Frame N+2:                          [PS_pre]  → ...
 *     Effective throughput = max(T_pre, T_infer, T_post)  NOT their sum
 *
 *   Level 2 (Layer-level, Theorem 5, Eq. 30):
 *     Overlaps weight DMA for layer L+1 with computation of layer L.
 *     Implemented by ping-pong double buffer in line_buffer.cpp.
 *     Achieved via POSIX threads: dma_thread runs concurrently with compute.
 *
 *   Level 3 (Tile-level, Eq. 31):
 *     Overlaps output DMA (HP2) write with next-tile computation.
 *     HLS DATAFLOW pragma achieves this within the PL IP.
 *
 * Speedup formula (Theorem 5, Eq. 33):
 *   S_H3S = (T_pre + T_infer + T_post) /
 *            max(T_pre, T_infer, T_post) + epsilon
 *   Measured on NEU-DET: S_H3S = 2.31×  [Table 5]
 *
 * Frame pipeline state machine:
 *   IDLE → PRE_PROC → WAIT_PL → POST_PROC → IDLE
 *   Three frames in-flight simultaneously at steady state.
 *=============================================================================*/
#ifndef H3S_SCHEDULER_H
#define H3S_SCHEDULER_H

#include <stdint.h>
#include <pthread.h>
#include "uio_ctrl.h"

/* Maximum frames in the pipeline simultaneously (H3S Level 1 depth) */
#define H3S_PIPELINE_DEPTH  3

/* Layer descriptor: defines one conv layer's dimensions for PL dispatch */
typedef struct {
    int  layer_idx;   /* SBFP layer index (0-22)       */
    int  img_w;       /* output feature map width       */
    int  img_h;       /* output feature map height      */
    int  in_ch;       /* input channels                 */
    int  out_ch;      /* output channels                */
    int  use_int4;    /* 1 = INT4 DSP packing           */
} LayerDesc;

/* Frame descriptor: holds all state for one in-flight video frame */
typedef struct {
    int      frame_id;          /* monotonic frame counter                 */
    uint8_t *raw_img;           /* pointer to raw input (MIPI/USB/file)    */
    float   *pre_buf;           /* normalized float buffer after PS_pre    */
    int8_t  *quant_buf;         /* SBFP-quantised INT8/INT4 buffer for PL  */
    float   *out_buf;           /* dequantised float output from PL        */
    void    *det_results;       /* pointer to NMS output DetBox array      */
    int      n_detections;      /* count of valid detections after NMS     */
    /* Timing (nanoseconds, for H3S profiling) */
    uint64_t t_pre_start;
    uint64_t t_pre_end;
    uint64_t t_pl_start;
    uint64_t t_pl_end;
    uint64_t t_post_start;
    uint64_t t_post_end;
} FrameDesc;

/* H3S context: shared state across all pipeline threads */
typedef struct {
    UioCtx        *uio;                         /* PL register access       */
    const LayerDesc *layers;                    /* layer sequence (23 layers)*/
    int            n_layers;                    /* = SBFP_N_LAYERS = 23     */
    int            img_w_in;                    /* input image width (px)   */
    int            img_h_in;                    /* input image height (px)  */
    int            n_classes;                   /* detection classes        */
    /* Pipeline slot ring buffer (depth = H3S_PIPELINE_DEPTH) */
    FrameDesc      slots[H3S_PIPELINE_DEPTH];
    /* Synchronisation */
    pthread_mutex_t mutex;
    pthread_cond_t  cond_pre_done;   /* pre_proc → pl_dispatch              */
    pthread_cond_t  cond_pl_done;    /* pl_dispatch → post_proc             */
    /* Statistics */
    uint64_t       frames_processed;
    double         avg_fps;
    double         idle_time_ms;     /* recovered idle time (Eq. 33)        */
} H3sCtx;

/*---------------------------------------------------------------------------
 * API
 *---------------------------------------------------------------------------*/

/**
 * h3s_init() – initialise H3S context
 * @uio       : open UIO handle for PL control
 * @layers    : array of 23 LayerDesc (SLF-YOLOv8 layer sequence)
 * @n_layers  : number of layers (23)
 * @img_w_in  : input image width  (e.g. 640)
 * @img_h_in  : input image height (e.g. 640)
 * @n_classes : number of detection classes (6 for NEU-DET, 10 for VisDrone)
 */
H3sCtx *h3s_init(UioCtx *uio,
                  const LayerDesc *layers, int n_layers,
                  int img_w_in, int img_h_in, int n_classes);

/**
 * h3s_destroy() – free all resources
 */
void h3s_destroy(H3sCtx *ctx);

/**
 * h3s_run() – run the H3S three-level pipeline for N frames
 *
 * Internally spawns three POSIX threads:
 *   t_pre:    PS preprocessing (decode → resize → SBFP quantise)
 *   t_infer:  PL inference dispatch (uio_accel_start × 23 layers)
 *   t_post:   PS post-processing (dequantise → NMS → result write)
 *
 * Thread synchronisation (Eq. 29-31):
 *   cond_pre_done: t_pre signals t_infer when quant_buf ready
 *   cond_pl_done:  t_infer signals t_post when out_buf ready
 *   Mutex protects slot ring buffer pointer swap.
 *
 * @input_paths : array of image file paths (or NULL for camera stream)
 * @n_frames    : number of frames to process
 * @output_cb   : callback invoked by t_post for each completed frame
 *                signature: void cb(FrameDesc *frame, void *user_data)
 * @user_data   : passed through to output_cb
 */
typedef void (*H3sOutputCb)(FrameDesc *frame, void *user_data);

int h3s_run(H3sCtx *ctx,
            const char **input_paths, int n_frames,
            H3sOutputCb output_cb, void *user_data);

/**
 * h3s_print_stats() – print pipeline utilisation (Table 5 / Fig. 6b format)
 * Reports: avg FPS, PS_pre / PL_infer / PS_post time breakdown,
 *          idle time recovered by H3S, effective speedup S_H3S.
 */
void h3s_print_stats(H3sCtx *ctx);

#endif /* H3S_SCHEDULER_H */
