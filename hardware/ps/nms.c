/*=============================================================================
 * nms.c  –  ARM NMS Implementation
 *
 * Paper: Section 5.3, H3S post-processing stage, T_post = 3.1 ms
 *
 * SLF-YOLOv8 detection head grid sizes (640×640 input):
 *   P2: 160×160 = 25,600 anchors  (small targets, VisDrone focus)
 *   P3:  80×80  =  6,400 anchors
 *   P4:  40×40  =  1,600 anchors
 *   P5:  20×20  =    400 anchors
 *   Total: 34,000 raw predictions → typically <50 after conf threshold
 *=============================================================================*/
#include "nms.h"

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>

/* Grid sizes for 4 detection heads at 640×640 */
static const int HEAD_GRIDS[4][2] = {
    {160, 160}, /* P2 */
    { 80,  80}, /* P3 */
    { 40,  40}, /* P4 */
    { 20,  20}, /* P5 */
};
static const int HEAD_STRIDES[4] = {4, 8, 16, 32};

/*---------------------------------------------------------------------------
 * decode_boxes()
 *
 * Decode YOLOv8 raw output to absolute xyxy bounding boxes.
 * YOLOv8 format: cx,cy,w,h in grid-cell units; no sigmoid on box (decoupled).
 * obj_conf and cls_conf use sigmoid activation (applied by PS after PL).
 *
 * Fills candidates[] and returns count passing conf_thresh.
 *---------------------------------------------------------------------------*/
static int decode_boxes(const float *raw_output, int n_classes,
                         float conf_thresh,
                         DetBox *candidates, int max_cands)
{
    int n_cands = 0;
    int offset  = 0;

    for (int h = 0; h < 4; h++) {
        int gw = HEAD_GRIDS[h][0];
        int gh = HEAD_GRIDS[h][1];
        int stride = HEAD_STRIDES[h];
        int pred_dim = 5 + n_classes;

        for (int gy = 0; gy < gh && n_cands < max_cands; gy++) {
            for (int gx = 0; gx < gw && n_cands < max_cands; gx++) {
                const float *p = raw_output + offset + (gy*gw + gx) * pred_dim;

                float cx = p[0]; float cy = p[1];
                float bw = p[2]; float bh = p[3];
                /* Sigmoid on obj_conf */
                float obj = 1.0f / (1.0f + expf(-p[4]));

                /* Find max class confidence */
                float max_cls = -1e9f; int cls_id = 0;
                for (int c = 0; c < n_classes; c++) {
                    float v = 1.0f / (1.0f + expf(-p[5 + c]));
                    if (v > max_cls) { max_cls = v; cls_id = c; }
                }

                float conf = obj * max_cls;
                if (conf < conf_thresh) continue;

                /* Convert to absolute pixel coordinates */
                float ax = (cx + gx) * stride;
                float ay = (cy + gy) * stride;
                float aw = bw * stride;
                float ah = bh * stride;

                DetBox *d = &candidates[n_cands++];
                d->x1 = ax - aw * 0.5f;
                d->y1 = ay - ah * 0.5f;
                d->x2 = ax + aw * 0.5f;
                d->y2 = ay + ah * 0.5f;
                d->confidence = conf;
                d->class_id   = cls_id;
                d->cx_raw = cx; d->cy_raw = cy;
            }
        }
        offset += gw * gh * pred_dim;
    }
    return n_cands;
}

/*---------------------------------------------------------------------------
 * sort_by_conf()  –  descending insertion sort (n_det typically <50)
 *---------------------------------------------------------------------------*/
static void sort_by_conf(DetBox *boxes, int n)
{
    for (int i = 1; i < n; i++) {
        DetBox key = boxes[i];
        int j = i - 1;
        while (j >= 0 && boxes[j].confidence < key.confidence) {
            boxes[j+1] = boxes[j]; j--;
        }
        boxes[j+1] = key;
    }
}

/*---------------------------------------------------------------------------
 * nms_run()
 *---------------------------------------------------------------------------*/
int nms_run(const float *raw_output, DetBox *out_boxes,
            int n_classes, float conf_thresh, float iou_thresh)
{
    /* Step 1: decode + threshold */
    static DetBox candidates[34000];  /* static: avoid stack overflow on A53 */
    int n_cands = decode_boxes(raw_output, n_classes, conf_thresh,
                               candidates, 34000);

    /* Step 2: sort by confidence */
    sort_by_conf(candidates, n_cands);

    /* Step 3: class-aware greedy NMS */
    static int suppressed[34000];
    memset(suppressed, 0, n_cands * sizeof(int));

    int n_out = 0;
    for (int i = 0; i < n_cands && n_out < NMS_MAX_DET; i++) {
        if (suppressed[i]) continue;
        out_boxes[n_out++] = candidates[i];

        for (int j = i+1; j < n_cands; j++) {
            if (suppressed[j]) continue;
            if (candidates[j].class_id != candidates[i].class_id) continue;
            float iou = nms_iou(candidates[i].x1, candidates[i].y1,
                                candidates[i].x2, candidates[i].y2,
                                candidates[j].x1, candidates[j].y1,
                                candidates[j].x2, candidates[j].y2);
            if (iou > iou_thresh) suppressed[j] = 1;
        }
    }
    return n_out;
}

/*---------------------------------------------------------------------------
 * nms_run_ex()  –  with coordinate recovery back to original image space
 *---------------------------------------------------------------------------*/
int nms_run_ex(const float *raw_output, DetBox *out_boxes,
               int n_classes, float conf_thresh, float iou_thresh,
               const LetterboxInfo *lb_info)
{
    int n = nms_run(raw_output, out_boxes, n_classes, conf_thresh, iou_thresh);
    if (!lb_info) return n;

    for (int i = 0; i < n; i++) {
        /* Remove letterbox padding then rescale */
        out_boxes[i].x1 = (out_boxes[i].x1 - lb_info->pad_w) / lb_info->scale;
        out_boxes[i].y1 = (out_boxes[i].y1 - lb_info->pad_h) / lb_info->scale;
        out_boxes[i].x2 = (out_boxes[i].x2 - lb_info->pad_w) / lb_info->scale;
        out_boxes[i].y2 = (out_boxes[i].y2 - lb_info->pad_h) / lb_info->scale;
        /* Clamp to image bounds */
        if (out_boxes[i].x1 < 0) out_boxes[i].x1 = 0;
        if (out_boxes[i].y1 < 0) out_boxes[i].y1 = 0;
        if (out_boxes[i].x2 > lb_info->src_w) out_boxes[i].x2 = lb_info->src_w;
        if (out_boxes[i].y2 > lb_info->src_h) out_boxes[i].y2 = lb_info->src_h;
    }
    return n;
}

/*---------------------------------------------------------------------------
 * nms_print_results()
 *---------------------------------------------------------------------------*/
void nms_print_results(const DetBox *boxes, int n_boxes,
                        const char * const *class_names)
{
    for (int i = 0; i < n_boxes; i++) {
        const char *name = class_names ? class_names[boxes[i].class_id] : "obj";
        printf("  [%d] %s  conf=%.3f  [%.1f,%.1f,%.1f,%.1f]\n",
               i, name, boxes[i].confidence,
               boxes[i].x1, boxes[i].y1, boxes[i].x2, boxes[i].y2);
    }
}
