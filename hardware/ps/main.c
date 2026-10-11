/*=============================================================================
 * main.c  –  SPECTRA-YOLO PS Application Entry Point
 *
 * Paper: Section 6 (Experiments), ZCU102 real-time inference demo
 *
 * Command-line usage:
 *   ./spectra_yolo_ps --n-frames 1000 --input /path/to/images/
 *   ./spectra_yolo_ps --n-frames 100  --input /dev/video0   (camera)
 *   ./spectra_yolo_ps --n-frames 1    --input image.jpg     (single image)
 *
 * This binary:
 *   1. Opens /dev/uio0 to access PL AXI4-Lite registers
 *   2. Initialises H3S 3-level pipeline scheduler
 *   3. Runs inference and prints FPS + detection results
 *   4. Prints Table 5 equivalent stats (FPS, latency breakdown, speedup)
 *=============================================================================*/
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <getopt.h>
#include <dirent.h>

#include "uio_ctrl.h"
#include "h3s_scheduler.h"
#include "nms.h"

/* NEU-DET class names (Table 3) */
static const char *NEU_DET_CLASSES[] = {
    "Crazing", "Inclusion", "Patches", "Pitted", "Rolled", "Scratches"
};

/* SLF-YOLOv8 23-layer sequence for ZCU102 (Table 5, 640×640 input) */
static const LayerDesc LAYER_SEQ[23] = {
    /* CBS Stem */     {0,  320,320, 3,   16,  1},
    /* CBS P2  */      {1,  320,320, 16,  32,  1},
    /* C2f P2  */      {2,  320,320, 32,  32,  1},
    /* CBAM P2 */      {3,  320,320, 32,  32,  1},
    /* CBS P3  */      {4,  160,160, 32,  64,  1},
    /* C2f P3-a*/      {5,  160,160, 64,  64,  1},
    /* C2f P3-b*/      {6,  160,160, 64,  64,  1},
    /* CBAM P3 */      {7,  160,160, 64,  64,  1},
    /* CBS P4  */      {8,   80, 80, 64, 128,  1},
    /* C2f P4-a*/      {9,   80, 80,128, 128,  1},
    /* C2f P4-b*/      {10,  80, 80,128, 128,  1},
    /* CBAM P4 */      {11,  80, 80,128, 128,  1},
    /* CBS P5  */      {12,  40, 40,128, 256,  1},
    /* C2f P5-a*/      {13,  40, 40,256, 256,  1},
    /* SPPF    */      {14,  40, 40,256, 256,  1},
    /* BiFPN 0 */      {15,  80, 80,256, 128,  1},
    /* BiFPN 1 */      {16, 160,160,128,  64,  1},
    /* BiFPN 2 */      {17, 320,320, 64,  32,  1},
    /* BiFPN 3 */      {18, 160,160, 32,  64,  1},
    /* BiFPN 4 */      {19,  80, 80, 64, 128,  1},
    /* Detect P2*/     {20, 320,320, 32,  6,   1},
    /* Detect P3*/     {21, 160,160, 64,  6,   1},
    /* Detect P5*/     {22,  40, 40,256,  6,   1},
};

/*---------------------------------------------------------------------------
 * Detection callback: print results for each completed frame
 *---------------------------------------------------------------------------*/
static void on_frame_done(FrameDesc *fd, void *user_data)
{
    int verbose = *(int *)user_data;
    double pl_ms = (fd->t_pl_end  - fd->t_pl_start)  * 1e-6;
    double pre_ms= (fd->t_pre_end - fd->t_pre_start) * 1e-6;
    double pst_ms= (fd->t_post_end- fd->t_post_start)* 1e-6;

    if (verbose) {
        printf("Frame %4d: pre=%.1fms PL=%.1fms post=%.1fms | %d dets\n",
               fd->frame_id, pre_ms, pl_ms, pst_ms, fd->n_detections);
        nms_print_results((DetBox *)fd->det_results, fd->n_detections,
                          NEU_DET_CLASSES);
    } else if (fd->frame_id % 100 == 0) {
        printf("Frame %4d: %.1f ms | %d detections\n",
               fd->frame_id, pl_ms, fd->n_detections);
    }
}

/*---------------------------------------------------------------------------
 * main()
 *---------------------------------------------------------------------------*/
int main(int argc, char *argv[])
{
    const char *input_path = NULL;
    int n_frames = 100;
    int verbose  = 0;
    int n_classes = 6;   /* NEU-DET default */

    static struct option opts[] = {
        {"input",     required_argument, 0, 'i'},
        {"n-frames",  required_argument, 0, 'n'},
        {"n-classes", required_argument, 0, 'c'},
        {"verbose",   no_argument,       0, 'v'},
        {0, 0, 0, 0}
    };
    int opt;
    while ((opt = getopt_long(argc, argv, "i:n:c:v", opts, NULL)) != -1) {
        switch (opt) {
            case 'i': input_path = optarg; break;
            case 'n': n_frames   = atoi(optarg); break;
            case 'c': n_classes  = atoi(optarg); break;
            case 'v': verbose    = 1; break;
        }
    }

    printf("=== SPECTRA-YOLO ZCU102 Inference ===\n");
    printf("Input: %s  Frames: %d  Classes: %d\n",
           input_path ? input_path : "(camera)", n_frames, n_classes);

    /* 1. Open UIO device for PL register access */
    UioCtx *uio = uio_open("/dev/uio0", ACCEL_PHYS_BASE, ACCEL_MAP_SIZE);
    if (!uio) {
        fprintf(stderr, "ERROR: Cannot open /dev/uio0. Is PetaLinux UIO driver loaded?\n");
        fprintf(stderr, "  Run: modprobe uio_pdrv_genirq && dtoverlay spectra_yolo_uio\n");
        return 1;
    }

    /* 2. Initialise H3S scheduler */
    H3sCtx *h3s = h3s_init(uio, LAYER_SEQ, 23, 640, 640, n_classes);
    if (!h3s) { uio_close(uio); return 1; }

    /* 3. Build input file list (simple: replicate single path) */
    const char **inputs = NULL;
    if (input_path) {
        inputs = (const char **)malloc(n_frames * sizeof(char *));
        for (int i = 0; i < n_frames; i++) inputs[i] = input_path;
    }

    /* 4. Run H3S pipeline */
    h3s_run(h3s, inputs, n_frames, on_frame_done, &verbose);

    /* 5. Print performance summary (Table 5 format) */
    printf("\n=== Performance Summary ===\n");
    h3s_print_stats(h3s);

    free(inputs);
    h3s_destroy(h3s);
    uio_close(uio);
    return 0;
}
