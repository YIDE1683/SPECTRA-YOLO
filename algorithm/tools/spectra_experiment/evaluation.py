"""Auditable CPU FP32 postprocessing and category-aware COCO-style bbox AP.

This module does not run a model, select a split, resolve NEU annotation policy,
or implement the official NEU/VisDrone benchmark protocols. Inputs must already
contain ALL eligible ordinary GT objects, not ROI-selected or class-agnostic GT.
Nonempty ignore regions and nonzero ignore/crowd/difficult flags are unsupported
and rejected, not translated to COCO iscrowd. A previously documented policy
such as legacy_include may supply difficult objects as ordinary boxes/labels;
the policy string itself is not an ignore flag.

Postprocess choices: ascending-stride levels, row-major anchors; one highest
sigmoid class per anchor (lowest class index breaks ties); score >= threshold;
UNCLAMPED input-pixel decode, inverse actual axis scales and integer padding,
then clip to original image and drop zero area. No extra center/ROI filter.
Stable class-aware greedy NMS suppresses IoU > threshold; score ties retain the
earlier original anchor, including across levels. Global all-class top-K is
applied AFTER NMS. This is CPU FP32 inference-only, not certified CUDA/AMP.

Evaluation uses installed pycocotools COCOeval.evaluate/accumulate, NOT custom
AP integration or summarize() (whose AP entry can hard-code maxDets=100).
IoU=linspace(.5,.95,10), recall=linspace(0,1,101), useCats=1. The all-area upper
bound covers every supported coordinate (2**24 pixels); no object is omitted
by COCO's otherwise finite default 'all' bound. Predictions are stably sorted
and capped globally per image before COCO's per-category maxDets [1,10,K].
AP is the unweighted mean over IoUs, recall points and classes having GT; absent
classes have None metrics, never artificial zero. Raw tables preserve COCO -1
undefined sentinels. Metric values are proportions; percentage-point conversion
is 100*AP (and a change in pp is 100*(AP_new-AP_old)).

Tied scores across images follow lexicographic sample_id order (independent of
input list order); ties within an image preserve supplied prediction order.
Source inspection 2026-08-31: official cocodataset/cocoapi PythonAPI/cocoeval.py
and locally installed pycocotools 2.0.10. Runtime provenance hashes the actual
local engine, compiled IoU extension and this wrapper; upstream master URLs
are source references, NOT a claim of byte-identical upstream revision.
"""

from __future__ import annotations

from contextlib import redirect_stdout
import hashlib
from importlib.metadata import version
import io
import math
from numbers import Real
from pathlib import Path

import numpy as np
import torch
from torch import Tensor


OFFICIAL_SOURCE = "https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/cocoeval.py"
DISTRIBUTION_SOURCE = "https://github.com/ppwwyyxx/cocoapi"
MAX_COORDINATE = 2**24


def _hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _limit(value):
    if type(value) is not int or value < 1:
        raise ValueError("max_detections must be a positive integer")
    return value


def _probability(value, name):
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError(f"{name} must be finite and in [0,1]")
    return float(value)


def _fp32(value, name):
    if not isinstance(value, Tensor) or value.dtype != torch.float32 or value.device.type != "cpu":
        raise ValueError(f"{name} requires a CPU FP32 tensor; no CUDA/AMP support")
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"{name} contains nonfinite values")
    return value.detach()


def _hw(value, name):
    if (not isinstance(value, (tuple, list)) or len(value) != 2
            or any(type(v) is not int or not 0 < v <= MAX_COORDINATE for v in value)):
        raise ValueError(f"{name} must be integer (H,W) in [1,2**24]")
    return tuple(value)


def _sample_id(record):
    if not isinstance(record, dict) or not isinstance(record.get("sample_id"), str) or not record["sample_id"].strip():
        raise ValueError("each image requires a nonempty string sample_id")
    return record["sample_id"]


def _transform(record):
    _sample_id(record)
    transform = record.get("transform")
    if not isinstance(transform, dict):
        raise ValueError("metadata requires transform with actual axis scales and padding")
    original = _hw(transform.get("original_hw"), "transform.original_hw")
    output = _hw(transform.get("output_hw"), "transform.output_hw")
    for key, expected in (("original_hw", original), ("image_hw", output)):
        if key in record and _hw(record[key], key) != expected:
            raise ValueError(f"metadata.{key} disagrees with transform")
    sx, sy = transform.get("scale_x"), transform.get("scale_y")
    for value in (sx, sy):
        if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value) or value <= 0:
            raise ValueError("scale_x/scale_y must be positive finite actual resize scales")
    left, top = transform.get("pad_left"), transform.get("pad_top")
    if any(type(v) is not int or v < 0 for v in (left, top)):
        raise ValueError("pad_left/pad_top must be nonnegative integers")
    resized = (original[0] * float(sy), original[1] * float(sx))
    if (not all(math.isfinite(v) and v > 0 for v in resized)
            or resized[0] + top > output[0] + 1e-6 or resized[1] + left > output[1] + 1e-6):
        raise ValueError("resized content and padding must fit output_hw")
    if "resized_hw" in transform:
        given = _hw(transform["resized_hw"], "transform.resized_hw")
        if any(not math.isclose(a, b, rel_tol=0, abs_tol=1e-6) for a, b in zip(given, resized)):
            raise ValueError("actual scales disagree with resized_hw/original_hw")
    for key, remaining in (("pad_bottom", output[0] - resized[0] - top),
                           ("pad_right", output[1] - resized[1] - left)):
        if key in transform and (type(transform[key]) is not int or transform[key] < 0
                                  or not math.isclose(transform[key], remaining, rel_tol=0, abs_tol=1e-6)):
            raise ValueError(f"{key} disagrees with content geometry")
    return {"original_hw": original, "output_hw": output, "scale_x": float(sx), "scale_y": float(sy),
            "pad_left": left, "pad_top": top}


def _decode(outputs, image_hw):
    if not isinstance(outputs, (list, tuple)) or not outputs:
        raise ValueError("outputs must be a nonempty sequence of raw levels")
    previous_stride, common = 0, None
    boxes, logits = [], []
    for level, output in enumerate(outputs):
        if not isinstance(output, dict) or not {"class_logits", "distances", "stride"} <= output.keys():
            raise ValueError("each raw output requires class_logits, distances and stride")
        cls = _fp32(output["class_logits"], f"level {level} class_logits")
        ltrb = _fp32(output["distances"], f"level {level} distances")
        if cls.ndim != 4 or min(cls.shape) <= 0:
            raise ValueError("class_logits must be nonempty NCHW")
        batch, classes, height, width = cls.shape
        if ltrb.shape != (batch, 4, height, width) or bool((ltrb < 0).any()):
            raise ValueError("distances must be nonnegative N4HW matching class_logits")
        stride = output["stride"]
        if type(stride) is not int or not previous_stride < stride <= MAX_COORDINATE:
            raise ValueError("positive integer strides must be strictly ascending")
        previous_stride = stride
        if (height * stride, width * stride) != image_hw:
            raise ValueError("all feature dimensions * stride must equal transform.output_hw")
        if common is not None and common != (batch, classes):
            raise ValueError("all raw levels must share batch and class count")
        common = (batch, classes)
        y, x = torch.meshgrid(torch.arange(height, dtype=torch.float32),
                              torch.arange(width, dtype=torch.float32), indexing="ij")
        centers = torch.stack((x + .5, y + .5), dim=-1).reshape(-1, 2) * stride
        distances = ltrb.permute(0, 2, 3, 1).reshape(batch, -1, 4) * stride
        decoded = torch.cat((centers[None] - distances[..., :2], centers[None] + distances[..., 2:]), dim=-1)
        if not bool(torch.isfinite(decoded).all()):
            raise ValueError("raw decode overflows FP32 before clipping")
        boxes.append(decoded)
        logits.append(cls.permute(0, 2, 3, 1).reshape(batch, -1, classes))
    return torch.cat(boxes, dim=1), torch.cat(logits, dim=1)


def _stable_nms(boxes, scores, labels, nms_iou, max_detections):
    # Stable global order is equivalent to classwise NMS followed by global
    # top-K. A different class cannot suppress a candidate. Early stop at K
    # cannot change any of the first K retained detections.
    order = torch.argsort(scores, descending=True, stable=True)
    retained = []
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    while order.numel() and len(retained) < max_detections:
        current = int(order[0])
        retained.append(current)
        rest = order[1:]
        if not rest.numel():
            break
        same = labels[rest] == labels[current]
        suppress = torch.zeros(rest.shape, dtype=torch.bool)
        candidates = rest[same]
        if candidates.numel():
            intersection_hw = (torch.minimum(boxes[current, 2:], boxes[candidates, 2:])
                               - torch.maximum(boxes[current, :2], boxes[candidates, :2])).clamp(min=0)
            intersection = intersection_hw[:, 0] * intersection_hw[:, 1]
            union = areas[current] + areas[candidates] - intersection
            iou = intersection / union
            if not bool(torch.isfinite(iou).all()):
                raise ValueError("nonfinite FP32 NMS IoU")
            suppress[same] = iou > nms_iou
        order = rest[~suppress]
    return torch.tensor(retained, dtype=torch.int64)


@torch.no_grad()
def postprocess(outputs, metadata, score_threshold=.001, nms_iou=.6, max_detections=300):
    """Return per-image detached CPU tensors in original-pixel coordinates.

    metadata: one dict per batch image, with sample_id and transform containing
    original_hw/output_hw, scale_x/scale_y and pad_left/pad_top (same names as
    spectra_rewrite_training.data). Extra image_hw/original_hw/resized_hw and
    opposite padding, if supplied, must agree. No source image files are read.
    """
    score_threshold = _probability(score_threshold, "score_threshold")
    nms_iou = _probability(nms_iou, "nms_iou")
    max_detections = _limit(max_detections)
    if not isinstance(metadata, (list, tuple)) or not metadata:
        raise ValueError("metadata requires one dictionary per image")
    transforms = [_transform(item) for item in metadata]
    ids = [_sample_id(item) for item in metadata]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate metadata sample_id")
    if any(t["output_hw"] != transforms[0]["output_hw"] for t in transforms):
        raise ValueError("batch transforms must share output_hw")
    decoded, logits = _decode(outputs, transforms[0]["output_hw"])
    if len(metadata) != decoded.shape[0]:
        raise ValueError("metadata count does not match raw batch size")
    scores, labels = logits.sigmoid().max(dim=-1)
    wrapper_hash = _hash(__file__)
    results = []
    for index, transform in enumerate(transforms):
        scale = torch.tensor([transform["scale_x"], transform["scale_y"]] * 2, dtype=torch.float32)
        pad = torch.tensor([transform["pad_left"], transform["pad_top"]] * 2, dtype=torch.float32)
        original = (decoded[index] - pad) / scale
        if not bool(torch.isfinite(original).all()):
            raise ValueError("inverse letterbox overflows FP32 before clipping")
        height, width = transform["original_hw"]
        original[:, 0::2].clamp_(0, width)
        original[:, 1::2].clamp_(0, height)
        sizes = original[:, 2:] - original[:, :2]
        candidate = (scores[index] >= score_threshold) & (sizes > 0).all(dim=1)
        # Guard area underflow as well as nonfinite arithmetic: positive sides
        # alone do not guarantee representable FP32 NMS area.
        areas = sizes[candidate, 0] * sizes[candidate, 1]
        if not bool(torch.isfinite(areas).all()) or bool((areas <= 0).any()):
            raise ValueError("positive candidate has nonrepresentable FP32 area")
        anchors = candidate.nonzero(as_tuple=False).flatten()
        boxes_i, scores_i, labels_i = original[candidate], scores[index, candidate], labels[index, candidate]
        keep = _stable_nms(boxes_i, scores_i, labels_i, nms_iou, max_detections)
        results.append({"sample_id": ids[index], "boxes": boxes_i[keep].clone(),
                        "scores": scores_i[keep].clone(), "labels": labels_i[keep].clone(),
                        "original_hw": transform["original_hw"],
                        "source_config": {"score_threshold": score_threshold, "nms_iou": nms_iou,
                            "max_detections_global_per_image": max_detections, "multi_label": False,
                            "nms_class_aware": True, "nms_suppression": "IoU > threshold",
                            "tie_rule": "score descending, original flattened anchor ascending; class tie lowest index",
                            "retained_anchor_indices": anchors[keep].tolist(), "transform": transform,
                            "decode": "unclamped input pixels -> inverse letterbox -> original image clip",
                            "backend": "deterministic CPU FP32 torch greedy NMS", "torch_version": torch.__version__,
                            "evaluation_py_sha256": wrapper_hash}})
    return results


def _reject_unsupported(record):
    for key in ("dataset", "dataset_name", "benchmark"):
        if isinstance(record.get(key), str) and "visdrone" in record[key].lower():
            raise ValueError("VisDrone evaluation protocol is not supported")
    for key in ("ignore_boxes", "ignore_metadata", "ignore_regions"):
        if key not in record:
            continue
        value = record[key]
        if isinstance(value, Tensor):
            empty = value.ndim >= 1 and value.shape[0] == 0
        elif isinstance(value, (list, tuple)):
            empty = len(value) == 0
        else:
            empty = False
        if not empty:
            raise ValueError(f"nonempty or malformed {key} is unsupported; freeze an explicit evaluation policy")
    for key in ("ignore", "iscrowd", "crowd", "difficult"):
        if key in record:
            try:
                flags = torch.as_tensor(record[key])
                safe = flags.device.type == "cpu" and bool(torch.isfinite(flags).all()) and not bool((flags != 0).any())
            except (ValueError, TypeError, RuntimeError):
                safe = False
            if not safe:
                raise ValueError(f"nonzero or malformed {key} flags are unsupported, not mapped to iscrowd")


def _records(records, classes, predictions=False):
    if not isinstance(records, (list, tuple)):
        raise ValueError("ground_truth/predictions must be sequences")
    result = {}
    for record in records:
        sample_id = _sample_id(record)
        if sample_id in result:
            raise ValueError(f"duplicate sample_id: {sample_id}")
        _reject_unsupported(record)
        boxes = _fp32(record.get("boxes"), f"{sample_id}.boxes")
        labels = record.get("labels")
        if boxes.ndim != 2 or boxes.shape[1] != 4:
            raise ValueError("boxes requires Nx4 including explicit empty (0,4)")
        if (not isinstance(labels, Tensor) or labels.dtype != torch.int64 or labels.device.type != "cpu"
                or labels.shape != (boxes.shape[0],) or bool(((labels < 0) | (labels >= classes)).any())):
            raise ValueError("labels requires CPU int64 N vector in class_names range")
        if bool((boxes < 0).any() or (boxes > MAX_COORDINATE).any() or (boxes[:, 2:] <= boxes[:, :2]).any()):
            raise ValueError("original boxes require positive-area half-open xyxy in [0,2**24]")
        if "original_hw" in record:
            height, width = _hw(record["original_hw"], "original_hw")
            if bool((boxes[:, 2] > width).any() or (boxes[:, 3] > height).any()):
                raise ValueError("boxes exceed original_hw; evaluation does not clip annotations")
        clean = {"boxes": boxes.clone(), "labels": labels.detach().clone(), "original_hw": record.get("original_hw")}
        if predictions:
            scores = _fp32(record.get("scores"), f"{sample_id}.scores")
            if scores.shape != (len(boxes),) or bool(((scores < 0) | (scores > 1)).any()):
                raise ValueError("scores requires finite FP32 N vector in [0,1]")
            clean["scores"] = scores.clone()
        result[sample_id] = clean
    return result


def _mean_defined(values):
    defined = values[values >= 0]
    return float(defined.mean()) if defined.size else None


@torch.no_grad()
def evaluate_predictions(ground_truth, predictions, class_names, max_detections=300):
    """Evaluate strictly paired original-image CPU tensors; return JSON-safe data.

    Each GT dict: sample_id (unique nonempty str), boxes (FP32 Nx4), labels
    (int64 N); predictions additionally scores (FP32 N in [0,1]). original_hw
    is optional but checked when provided; do not silently clip labels. Empty
    predictions still require a record for every GT image. Empty GT datasets
    return None metrics and undefined -1 tables. No ignored GT is silently
    dropped. Eligible categories without GT are excluded from macro means.
    """
    from pycocotools import coco as coco_module, cocoeval as eval_module, mask as mask_module, _mask

    max_detections = _limit(max_detections)
    if (not isinstance(class_names, (tuple, list)) or not class_names
            or any(not isinstance(name, str) or not name.strip() for name in class_names)
            or len(set(class_names)) != len(class_names)):
        raise ValueError("class_names must be a nonempty ordered sequence of unique nonempty names")
    gt = _records(ground_truth, len(class_names))
    pred = _records(predictions, len(class_names), predictions=True)
    if set(gt) != set(pred):
        raise ValueError(f"sample_id mismatch: missing={sorted(set(gt)-set(pred))}, unknown={sorted(set(pred)-set(gt))}")
    sample_ids = sorted(gt)
    categories = [{"id": i + 1, "name": name} for i, name in enumerate(class_names)]
    images, annotations, detections = [], [], []
    gt_counts = [0] * len(class_names)
    total_before = sum(len(item["boxes"]) for item in pred.values())
    for image_id, sample_id in enumerate(sample_ids, 1):
        target, prediction = gt[sample_id], pred[sample_id]
        if (target["original_hw"] is not None and prediction["original_hw"] is not None
                and tuple(target["original_hw"]) != tuple(prediction["original_hw"])):
            raise ValueError("GT and prediction original_hw disagree")
        known_hw = target["original_hw"] or prediction["original_hw"]
        if known_hw is not None:
            h, w = known_hw
            for data in (target, prediction):
                if bool((data["boxes"][:, 2] > w).any() or (data["boxes"][:, 3] > h).any()):
                    raise ValueError("GT/prediction exceeds paired image dimensions")
        images.append({"id": image_id, "sample_id": sample_id})
        for box, label in zip(target["boxes"].tolist(), target["labels"].tolist()):
            x1, y1, x2, y2 = box
            annotations.append({"id": len(annotations) + 1, "image_id": image_id, "category_id": label + 1,
                                "bbox": [x1, y1, x2-x1, y2-y1], "area": (x2-x1)*(y2-y1), "iscrowd": 0})
            gt_counts[label] += 1
        selected = torch.argsort(prediction["scores"], descending=True, stable=True)[:max_detections]
        for row in selected.tolist():
            x1, y1, x2, y2 = prediction["boxes"][row].tolist()
            detections.append({"image_id": image_id, "category_id": int(prediction["labels"][row]) + 1,
                               "bbox": [x1, y1, x2-x1, y2-y1], "score": float(prediction["scores"][row])})
    iou_thresholds = np.linspace(.5, .95, 10)
    recall_thresholds = np.linspace(0, 1, 101)
    engine_log = io.StringIO()
    with redirect_stdout(engine_log):
        coco_gt = coco_module.COCO()
        coco_gt.dataset = {"info": {"description": "explicit ordinary eligible GT only"}, "images": images,
                           "categories": categories, "annotations": annotations}
        coco_gt.createIndex()
        if detections:
            coco_dt = coco_gt.loadRes(detections)
        else:
            # loadRes([]) indexes anns[0] in installed pycocotools 2.0.10.
            coco_dt = coco_module.COCO()
            coco_dt.dataset = {"info": {}, "images": images, "categories": categories, "annotations": []}
            coco_dt.createIndex()
        evaluator = eval_module.COCOeval(coco_gt, coco_dt, "bbox")
        evaluator.params.imgIds = list(range(1, len(images)+1))
        evaluator.params.catIds = list(range(1, len(categories)+1))
        evaluator.params.useCats = 1
        evaluator.params.iouThrs = iou_thresholds
        evaluator.params.recThrs = recall_thresholds
        evaluator.params.areaRng = [[0., float(MAX_COORDINATE**2)]]
        evaluator.params.areaRngLbl = ["all"]
        evaluator.params.maxDets = [1, 10, max_detections]
        evaluator.evaluate()
        evaluator.accumulate()
    selected_max = evaluator.params.maxDets.index(max_detections)
    precision = evaluator.eval["precision"][:, :, :, 0, selected_max]
    recall = evaluator.eval["recall"][:, :, 0, selected_max]
    scores = evaluator.eval["scores"][:, :, :, 0, selected_max]
    if not all(np.isfinite(table).all() for table in (precision, recall, scores)):
        raise FloatingPointError("COCOeval returned nonfinite tables")
    metrics = {"ap50_95": _mean_defined(precision), "ap50": _mean_defined(precision[0]),
               "ap75": _mean_defined(precision[5]), "ar_max_detections": _mean_defined(recall)}
    classwise = [{"class_id": i, "class_name": name, "gt_count": gt_counts[i],
                  "ap50_95": _mean_defined(precision[:, :, i]), "ap50": _mean_defined(precision[0, :, i]),
                  "ap75": _mean_defined(precision[5, :, i]), "ar_max_detections": _mean_defined(recall[:, i])}
                 for i, name in enumerate(class_names)]
    source_hashes = {"evaluation.py": _hash(__file__), "pycocotools/coco.py": _hash(coco_module.__file__),
                     "pycocotools/cocoeval.py": _hash(eval_module.__file__), "pycocotools/mask.py": _hash(mask_module.__file__),
                     "pycocotools/_mask_binary": _hash(_mask.__file__)}
    return {**metrics, "status": "evaluated" if annotations else "undefined_no_ground_truth",
            "classwise": classwise, "image_count": len(images), "gt_count": len(annotations),
            "prediction_count_before_global_cap": total_before, "prediction_count_evaluated": len(detections),
            "sample_id_order": sample_ids,
            "precision": precision.tolist(), "recall": recall.tolist(), "recall_sample_scores": scores.tolist(),
            "protocol": {"name": "COCO-style category-aware macro bbox AP; NOT NEU/VisDrone official full protocol",
                         "iou_thresholds": iou_thresholds.tolist(), "recall_thresholds": recall_thresholds.tolist(),
                         "area_range": evaluator.params.areaRng[0], "area_label": "all", "use_categories": True,
                         "max_dets_requested": [1, 10, max_detections], "max_dets_engine_sorted": evaluator.params.maxDets,
                         "selected_max_detections": max_detections, "global_all_class_cap_per_image": max_detections,
                         "score_ties": "lexicographic sample_id, then original within-image prediction index",
                         "metric_unit": "proportion [0,1]; None when no eligible GT", "percentage_point_multiplier": 100,
                         "percentage_point_delta": "100 * (AP_new - AP_old)",
                         "macro_average": "equal weight per GT-present class; IoU and 101 recall points equally weighted",
                         "undefined_table_sentinel": -1, "precision_shape_axes": ["iou", "recall", "class"],
                         "recall_shape_axes": ["iou", "class"], "ignore_crowd_supported": False,
                         "gt_scope": "all caller-supplied eligible ordinary GT; no ROI filtering"},
            "provenance": {"pycocotools_version": version("pycocotools"), "numpy_version": np.__version__,
                           "torch_version": torch.__version__, "source_sha256": source_hashes,
                           "official_source": OFFICIAL_SOURCE, "distribution_source": DISTRIBUTION_SOURCE,
                           "source_inspection_date": "2026-08-31", "upstream_byte_identity_claimed": False,
                           "engine_calls": ["COCOeval.evaluate", "COCOeval.accumulate"],
                           "summarize_used": False}}
