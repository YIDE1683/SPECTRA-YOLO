/*=============================================================================
 * nms.h  –  ARM-side Non-Maximum Suppression
 *
 * Paper: Section 5.3 – "NMS executed on ARM Cortex-A53, T_post = 3.1 ms"
 *        H3S Level 1 Stage C (post-processing).
 *
 * SLF-YOLOv8 has 4 detection heads (P2/P3/P4/P5).
 * Each head produces raw predictions of shape [n_anchors × (5 + n_classes)]:
 *   [cx, cy, w, h, obj_conf, cls_conf_0, ..., cls_conf_{C-1}]
 *
 * NMS pipeline:
 *   1. Decode boxes: (cx,cy,w,h) → (x1,y1,x2,y2) absolute coordinates
 *   2. Threshold: keep candidates with obj_conf × max_cls_conf > conf_thresh
 *   3. Class-aware IoU-NMS: suppress overlapping boxes with IoU > iou_thresh
 *   4. Coordinate recovery: map from [0,640] space to original image space
 *      using LetterboxInfo (scale + pad_w/pad_h from preprocess.h)
 *
 * ARM optimisation:
 *   - IoU computation uses NEON float32x4_t for 4 box pairs/cycle
 *   - Candidate list sorted by confidence (insertion sort, n_det typically <50)
 *   - Measured T_post = 3.1 ms including dequantisation (640×640, 6 classes)
 *=============================================================================*/
#ifndef NMS_H
#define NMS_H

#include <stdint.h>
#include "preprocess.h"   /* LetterboxInfo */

/* NMS defaults (match EAAI paper Table 3 eval settings) */
#define NMS_CONF_THRESH  0.25f
#define NMS_IOU_THRESH   0.45f
#define NMS_MAX_DET      300

/*---------------------------------------------------------------------------
 * DetBox: a single detection result
 *---------------------------------------------------------------------------*/
typedef struct {
    float x1, y1, x2, y2;   /* bounding box in original image coordinates  */
    float confidence;         /* obj_conf × max_cls_conf                     */
    int   class_id;           /* argmax class index                          */
    float cx_raw, cy_raw;     /* raw network outputs (before coord recovery) */
} DetBox;

/*---------------------------------------------------------------------------
 * RawPred: one raw prediction from PL output buffer
 * Layout: [cx, cy, w, h, obj_conf, cls_0 ... cls_{C-1}]  (float32)
 *---------------------------------------------------------------------------*/
typedef struct {
    float cx, cy, w, h;
    float obj_conf;
    float cls_conf[64];   /* max 64 classes; use first n_classes entries     */
} RawPred;

/*---------------------------------------------------------------------------
 * API
 *---------------------------------------------------------------------------*/

/**
 * nms_run() – full NMS pipeline: decode → threshold → IoU-NMS
 *
 * @raw_output  : flat float32 buffer from PL (4 heads concatenated)
 *                layout: [P2_preds | P3_preds | P4_preds | P5_preds]
 *                each pred: [cx,cy,w,h,obj,cls_0,...,cls_{C-1}] (float32)
 * @out_boxes   : output array of DetBox (caller allocates, size NMS_MAX_DET)
 * @n_classes   : number of detection classes
 * @conf_thresh : minimum confidence (default NMS_CONF_THRESH = 0.25)
 * @iou_thresh  : IoU threshold for suppression (default NMS_IOU_THRESH = 0.45)
 *
 * Returns: number of valid detections (≤ NMS_MAX_DET)
 */
int nms_run(const float *raw_output, DetBox *out_boxes,
            int n_classes, float conf_thresh, float iou_thresh);

/**
 * nms_run_ex() – same as nms_run() but also applies coordinate recovery
 *
 * @lb_info : letterbox parameters from preprocess_image_ex()
 *            Converts DetBox coordinates from [0,640] to original image space.
 */
int nms_run_ex(const float *raw_output, DetBox *out_boxes,
               int n_classes, float conf_thresh, float iou_thresh,
               const LetterboxInfo *lb_info);

/**
 * nms_iou() – compute Intersection-over-Union of two boxes
 * Inline for NEON-vectorised batch IoU computation.
 */
static inline float nms_iou(float x1a, float y1a, float x2a, float y2a,
                              float x1b, float y1b, float x2b, float y2b)
{
    float inter_x1 = x1a > x1b ? x1a : x1b;
    float inter_y1 = y1a > y1b ? y1a : y1b;
    float inter_x2 = x2a < x2b ? x2a : x2b;
    float inter_y2 = y2a < y2b ? y2a : y2b;

    float inter_w = inter_x2 - inter_x1;
    float inter_h = inter_y2 - inter_y1;
    if (inter_w <= 0.0f || inter_h <= 0.0f) return 0.0f;

    float inter_area = inter_w * inter_h;
    float area_a = (x2a - x1a) * (y2a - y1a);
    float area_b = (x2b - x1b) * (y2b - y1b);
    return inter_area / (area_a + area_b - inter_area + 1e-7f);
}

/**
 * nms_print_results() – print detection results (debug / evaluation)
 * Format: "Frame N | class_id=X conf=0.92 [x1,y1,x2,y2]"
 */
void nms_print_results(const DetBox *boxes, int n_boxes,
                        const char * const *class_names);

#endif /* NMS_H */
