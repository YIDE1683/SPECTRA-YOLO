"""FP32 detection-supervision components for the content-rewrite manuscript.

Implementation choices below refine an underspecified training protocol; this
is NOT a claim of exact equivalence to another library's TAL implementation or
of validated detection accuracy. No training, NMS, or result plotting occurs.

Contract and deliberate choices
-------------------------------
* Output levels are in ascending stride and cells are flattened row-major.
  Raw nonnegative ltrb uses feature-grid units. Training decode is UNCLAMPED
  input-pixel xyxy so out-of-image predictions retain regression gradients.
  This is different from clipped inference visualization/decode.
* Detached alignment is sigmoid(class_logit)^0.5 * IoU^6. A GT nominates up to
  10 geometrically valid centers strictly inside its rectangle. Zero alignment
  scores remain eligible: ties select the lower flattened anchor index.
  Conflicts between nominated GTs select highest IoU, then the lower original
  GT index. No candidate is created for a GT without an eligible center.
* Targets are integer class labels and annotated pixel boxes, not masks.
  Content eligibility uses [left,right) x [top,bottom) center membership.
  Ignore rectangles conservatively include their boundaries. If a GT has a
  positive-area intersection with any ignore rectangle, the whole GT is
  removed AND its complete rectangle is added to anchor ignore. This prevents
  its remaining visible portion from becoming false classification negatives.
  GT outside its valid content rectangle is rejected, never silently clipped.
* BCE uses binary one-hot targets, not soft alignment/quality targets. All
  eligible class logits contribute; ignore/padding anchors contribute none.
  BCE sum, (1-GIoU) sum, and four-side L1 sum all divide by max(total batch
  positives,1). GT ltrb pixels are divided by their anchor stride exactly once.
* Total = 0.5*classification + 7.5*giou + 1.0*ltrb. The optional reconstruction
  loss and its external 0.01 coefficient are the caller's responsibility.
* This stage requires FP32 raw predictions. It does not certify AMP, optimizer
  accumulation, or all parameter gradients. An entry gradient guard rejects
  nonfinite incoming gradients; the caller must still audit parameter grads.
"""

from __future__ import annotations

import math
from numbers import Real

import torch
from torch import Tensor
from torch.nn import functional as F


class _FinitePredictionGradient(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value):
        return value

    @staticmethod
    def backward(ctx, gradient):
        if not bool(torch.isfinite(gradient).all()):
            raise FloatingPointError("Nonfinite FP32 detection gradient; inspect upstream loss scaling and model gradients")
        return gradient


def _fp32(value, name):
    if not isinstance(value, Tensor) or value.dtype != torch.float32:
        raise ValueError(f"{name} must be an FP32 tensor; AMP is not supported here")
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"{name} contains nonfinite values")


def _image_size(image_hw):
    if (not isinstance(image_hw, (tuple, list)) or len(image_hw) != 2
            or any(type(size) is not int or size <= 0 or size > 2**24 for size in image_hw)):
        raise ValueError("image_hw requires positive integer (H,W), each no greater than 2**24")
    return tuple(image_hw)


def decode_training_outputs(outputs, image_hw):
    """Return differentiable UNCLAMPED pixel boxes and row-major raw tensors.

    Returned logits/distances: [B,A,C]/[B,A,4], boxes: [B,A,4], centers: [A,2],
    strides: [A]. All levels must cover the same image exactly. Grid centers
    equal (column+0.5,row+0.5)*stride. There is no confidence threshold or NMS.
    """
    height, width = _image_size(image_hw)
    if not isinstance(outputs, (list, tuple)) or not outputs:
        raise ValueError("outputs must be a nonempty list/tuple")
    logits_list, distances_list, centers_list, stride_list = [], [], [], []
    previous_stride, common = 0, None
    for level, output in enumerate(outputs):
        if not isinstance(output, dict) or not {"class_logits", "distances", "stride"} <= output.keys():
            raise ValueError("each output requires class_logits, distances, stride")
        logits, distances, stride = output["class_logits"], output["distances"], output["stride"]
        _fp32(logits, f"outputs[{level}].class_logits")
        _fp32(distances, f"outputs[{level}].distances")
        if logits.ndim != 4 or min(logits.shape) <= 0:
            raise ValueError("class_logits requires nonempty NCHW")
        batch, classes, rows, cols = logits.shape
        if distances.shape != (batch, 4, rows, cols) or distances.device != logits.device:
            raise ValueError("distances must match batch/spatial dimensions and logits device")
        if type(stride) is not int or stride <= previous_stride:
            raise ValueError("strides must be positive and strictly ascending")
        previous_stride = stride
        if (rows * stride, cols * stride) != (height, width):
            raise ValueError("every feature shape multiplied by stride must equal image_hw")
        if bool((distances < 0).any()):
            raise ValueError("raw ltrb distances must be nonnegative")
        current = (batch, classes, logits.device)
        if common is not None and common != current:
            raise ValueError("all levels require matching batch, classes, and device")
        common = current
        logits = _FinitePredictionGradient.apply(logits)
        distances = _FinitePredictionGradient.apply(distances)
        logits_list.append(logits.permute(0, 2, 3, 1).reshape(batch, -1, classes))
        distances_list.append(distances.permute(0, 2, 3, 1).reshape(batch, -1, 4))
        y, x = torch.meshgrid(torch.arange(rows, device=logits.device, dtype=torch.float32),
                              torch.arange(cols, device=logits.device, dtype=torch.float32), indexing="ij")
        centers_list.append(torch.stack(((x + 0.5) * stride, (y + 0.5) * stride), dim=-1).reshape(-1, 2))
        stride_list.append(torch.full((rows * cols,), stride, device=logits.device, dtype=torch.float32))
    logits, distances = torch.cat(logits_list, dim=1), torch.cat(distances_list, dim=1)
    centers, strides = torch.cat(centers_list), torch.cat(stride_list)
    pixel_distances = distances * strides[None, :, None]
    boxes = torch.cat((centers[None] - pixel_distances[..., :2],
                       centers[None] + pixel_distances[..., 2:]), dim=-1)
    if not bool(torch.isfinite(boxes).all()):
        raise ValueError("unclamped pixel decode overflows FP32")
    return {"class_logits": logits, "distances": distances, "boxes_unclamped": boxes,
            "centers": centers, "strides": strides}


def _boxes(value, name, image_hw, device):
    _fp32(value, name)
    if value.ndim != 2 or value.shape[1] != 4:
        raise ValueError(f"{name} requires [N,4], including empty [0,4]")
    value = value.detach().to(device)
    if not value.numel():
        return value
    height, width = image_hw
    sizes = value[:, 2:] - value[:, :2]
    if bool((sizes <= 0).any()):
        raise ValueError(f"{name} requires positive-area xyxy boxes")
    if bool((value[:, :2] < 0).any() or (value[:, 2] > width).any() or (value[:, 3] > height).any()):
        raise ValueError(f"{name} must be fully inside image_hw; labels are not silently clipped")
    areas = sizes[:, 0] * sizes[:, 1]
    if not bool(torch.isfinite(areas).all()) or bool((areas <= 0).any()):
        raise ValueError(f"{name} has nonrepresentable FP32 area")
    return value


def _targets_and_regions(targets, valid_regions, ignore_boxes, batch, classes, image_hw, device):
    if not isinstance(targets, (list, tuple)) or len(targets) != batch:
        raise ValueError("targets requires one dictionary per image")
    if valid_regions is None:
        height, width = image_hw
        valid_regions = [torch.tensor([0., 0., float(width), float(height)], device=device)] * batch
    if not isinstance(valid_regions, (tuple, list)) or len(valid_regions) != batch:
        raise ValueError("valid_regions requires one FP32 xyxy vector per image")
    if ignore_boxes is None:
        ignore_boxes = [torch.empty(0, 4, dtype=torch.float32, device=device) for _ in range(batch)]
    if not isinstance(ignore_boxes, (tuple, list)) or len(ignore_boxes) != batch:
        raise ValueError("ignore_boxes requires one FP32 [K,4] tensor per image")
    prepared = []
    for index, (target, valid, ignored) in enumerate(zip(targets, valid_regions, ignore_boxes)):
        if not isinstance(target, dict) or not {"boxes", "labels"} <= target.keys():
            raise ValueError("each target requires boxes and labels")
        gt = _boxes(target["boxes"], f"targets[{index}].boxes", image_hw, device)
        labels = target["labels"]
        if not isinstance(labels, Tensor) or labels.dtype != torch.int64 or labels.shape != (len(gt),):
            raise ValueError("labels require int64 [GT_count]")
        labels = labels.detach().to(device)
        if bool(((labels < 0) | (labels >= classes)).any()):
            raise ValueError("target label is outside output class range")
        _fp32(valid, f"valid_regions[{index}]")
        if valid.shape != (4,):
            raise ValueError("each valid region requires shape [4]")
        valid = _boxes(valid[None], f"valid_regions[{index}]", image_hw, device)[0]
        if len(gt) and bool(((gt[:, 0] < valid[0]) | (gt[:, 1] < valid[1])
                             | (gt[:, 2] > valid[2]) | (gt[:, 3] > valid[3])).any()):
            raise ValueError("GT crosses valid content boundary; no silent truncation is permitted")
        ignored = _boxes(ignored, f"ignore_boxes[{index}]", image_hw, device)
        prepared.append((gt, labels, valid, ignored))
    return prepared


def _intersections(left: Tensor, right: Tensor):
    dimensions = (torch.minimum(left[:, None, 2:], right[None, :, 2:])
                  - torch.maximum(left[:, None, :2], right[None, :, :2])).clamp_min(0)
    area = dimensions[..., 0] * dimensions[..., 1]
    if not bool(torch.isfinite(area).all()):
        raise ValueError("intersection area overflows FP32")
    return area


def _pairwise_iou(prediction: Tensor, target: Tensor):
    intersection = _intersections(prediction, target)
    prediction_size = (prediction[:, 2:] - prediction[:, :2]).clamp_min(0)
    target_size = target[:, 2:] - target[:, :2]
    prediction_area = prediction_size[:, 0] * prediction_size[:, 1]
    target_area = target_size[:, 0] * target_size[:, 1]
    union = prediction_area[:, None] + target_area[None, :] - intersection
    if not bool(torch.isfinite(union).all()) or bool((union <= 0).any()):
        raise ValueError("IoU union is not a positive finite FP32 area")
    iou = intersection / union
    if not bool(torch.isfinite(iou).all()):
        raise ValueError("IoU arithmetic is nonfinite")
    return iou


def _centers_in_rectangles(centers: Tensor, boxes: Tensor, *, strict: bool):
    if strict:
        return ((centers[:, None, 0] > boxes[None, :, 0]) & (centers[:, None, 0] < boxes[None, :, 2])
                & (centers[:, None, 1] > boxes[None, :, 1]) & (centers[:, None, 1] < boxes[None, :, 3]))
    return ((centers[:, None, 0] >= boxes[None, :, 0]) & (centers[:, None, 0] <= boxes[None, :, 2])
            & (centers[:, None, 1] >= boxes[None, :, 1]) & (centers[:, None, 1] <= boxes[None, :, 3]))


@torch.no_grad()
def _assign_one(logits, decoded, centers, gt, labels, valid, ignored, topk, class_exponent, iou_exponent):
    anchors = len(centers)
    content = ((centers[:, 0] >= valid[0]) & (centers[:, 0] < valid[2])
               & (centers[:, 1] >= valid[1]) & (centers[:, 1] < valid[3]))
    original_ignore = _centers_in_rectangles(centers, ignored, strict=False).any(dim=1)
    ignored_gt = (_intersections(gt, ignored) > 0).any(dim=1)
    propagated = _centers_in_rectangles(centers, gt[ignored_gt], strict=False).any(dim=1)
    eligible = content & ~original_ignore & ~propagated
    assigned = torch.full((anchors,), -1, dtype=torch.int64, device=centers.device)
    best_iou = torch.full((anchors,), -1., dtype=torch.float32, device=centers.device)
    nominations = torch.zeros(anchors, dtype=torch.int64, device=centers.device)
    candidates_by_gt, no_candidate = [], []
    if len(gt):
        ious = _pairwise_iou(decoded.detach(), gt)
        inside = _centers_in_rectangles(centers, gt, strict=True)
        scores = logits.detach().sigmoid()
        for index in range(len(gt)):
            if bool(ignored_gt[index]):
                candidates_by_gt.append(0)
                continue
            candidates = torch.nonzero(inside[:, index] & eligible, as_tuple=False).flatten()
            candidates_by_gt.append(len(candidates))
            if not len(candidates):
                no_candidate.append(index)
                continue
            metric = scores[candidates, labels[index]].pow(class_exponent) * ious[candidates, index].pow(iou_exponent)
            if not bool(torch.isfinite(metric).all()):
                raise ValueError("task-alignment metric is nonfinite")
            order = torch.argsort(metric, descending=True, stable=True)
            selected = candidates[order[:topk]]
            nominations[selected] += 1
            overlap = ious[selected, index]
            # GT iteration is ascending original index; strict > retains the
            # earlier GT for an exact IoU tie between nominations.
            replace = overlap > best_iou[selected]
            winners = selected[replace]
            assigned[winners] = index
            best_iou[winners] = overlap[replace]
    unassigned_gt = [index for index in range(len(gt))
                     if not bool(ignored_gt[index]) and not bool((assigned == index).any())]
    stats = {
        "gt_total": len(gt), "gt_ignored_overlap": int(ignored_gt.sum()),
        "gt_eligible": int((~ignored_gt).sum()), "gt_without_candidates": len(no_candidate),
        "gt_unmatched_after_conflict": len(unassigned_gt),
        "anchors_total": anchors, "anchors_padding": int((~content).sum()),
        "anchors_original_ignore_in_content": int((content & original_ignore).sum()),
        "anchors_gt_propagation_additional": int((content & ~original_ignore & propagated).sum()),
        "anchors_classification_eligible": int(eligible.sum()),
        "anchors_multiple_gt_nominations": int((nominations > 1).sum()),
        "positive_anchors": int((assigned >= 0).sum()),
    }
    diagnostics = {"candidate_counts_per_gt": candidates_by_gt, "no_candidate_gt_indices": no_candidate,
                   "unmatched_gt_indices": unassigned_gt, "ignored_gt_indices": torch.nonzero(ignored_gt).flatten().cpu().tolist()}
    return assigned, eligible, stats, diagnostics


def _matched_giou(prediction, target):
    intersection_size = (torch.minimum(prediction[:, 2:], target[:, 2:])
                         - torch.maximum(prediction[:, :2], target[:, :2])).clamp_min(0)
    intersection = intersection_size[:, 0] * intersection_size[:, 1]
    prediction_size = (prediction[:, 2:] - prediction[:, :2]).clamp_min(0)
    target_size = target[:, 2:] - target[:, :2]
    union = (prediction_size[:, 0] * prediction_size[:, 1]
             + target_size[:, 0] * target_size[:, 1] - intersection)
    enclosing_size = torch.maximum(prediction[:, 2:], target[:, 2:]) - torch.minimum(prediction[:, :2], target[:, :2])
    enclosing = enclosing_size[:, 0] * enclosing_size[:, 1]
    if (not bool(torch.isfinite(union).all() & torch.isfinite(enclosing).all())
            or bool((union <= 0).any() | (enclosing <= 0).any())):
        raise ValueError("GIoU areas are not positive finite FP32")
    result = intersection / union - (enclosing - union) / enclosing
    if not bool(torch.isfinite(result).all()):
        raise ValueError("GIoU arithmetic is nonfinite")
    return result


def detection_loss(outputs, targets, image_hw, *, valid_regions=None, ignore_boxes=None,
                   topk=10, class_exponent=0.5, iou_exponent=6.0, return_details=False):
    """Return Eq.(6) detection terms, detached assignment stats, optional details.

    ``targets[n]`` = {boxes: FP32 [G,4] pixel xyxy, labels: int64 [G]}.
    ``valid_regions`` = per-image FP32 [4] vectors, default full image.
    ``ignore_boxes`` = per-image FP32 [K,4], default empty. No mask input is
    accepted as a surrogate GT or verified background annotation.

    Returns {loss, classification, giou, ltrb, num_positive, stats}; each loss
    is a differentiable FP32 scalar. With return_details=True, ``assignment``
    contains detached matched targets/masks/indices/centers/strides. Normalizer
    is global batch positive count, not image mean or number of class logits.
    """
    if type(topk) is not int or topk <= 0:
        raise ValueError("topk must be a positive integer")
    for name, value in (("class_exponent", class_exponent), ("iou_exponent", iou_exponent)):
        if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and strictly positive")
    if type(return_details) is not bool:
        raise ValueError("return_details must be bool")
    image_hw = _image_size(image_hw)
    decoded = decode_training_outputs(outputs, image_hw)
    logits, distances, prediction_boxes = decoded["class_logits"], decoded["distances"], decoded["boxes_unclamped"]
    centers, strides = decoded["centers"], decoded["strides"]
    batch, anchors, classes = logits.shape
    prepared = _targets_and_regions(targets, valid_regions, ignore_boxes, batch, classes, image_hw, logits.device)
    assignments, masks, image_stats, debug = [], [], [], []
    target_classes = torch.zeros_like(logits, requires_grad=False)
    target_boxes = torch.zeros_like(prediction_boxes, requires_grad=False)
    target_ltrb = torch.zeros_like(distances, requires_grad=False)
    with torch.no_grad():
        for image, (gt, labels, valid, ignored) in enumerate(prepared):
            indices, eligible, stats, diagnostics = _assign_one(
                logits[image], prediction_boxes[image], centers, gt, labels, valid, ignored,
                topk, float(class_exponent), float(iou_exponent))
            assignments.append(indices)
            masks.append(eligible)
            image_stats.append(stats)
            debug.append(diagnostics)
            positive = indices >= 0
            positions = torch.nonzero(positive, as_tuple=False).flatten()
            if len(positions):
                matched_gt = gt[indices[positive]]
                target_boxes[image, positive] = matched_gt
                target_classes[image, positions, labels[indices[positive]]] = 1
                pixel_ltrb = torch.cat((centers[positive] - matched_gt[:, :2],
                                       matched_gt[:, 2:] - centers[positive]), dim=1)
                target_ltrb[image, positive] = pixel_ltrb / strides[positive, None]
    assigned = torch.stack(assignments)
    eligible = torch.stack(masks)
    positive = assigned >= 0
    count = int(positive.sum())
    normalizer = max(count, 1)
    element_bce = F.binary_cross_entropy_with_logits(logits, target_classes, reduction="none")
    classification = (element_bce * eligible[:, :, None]).sum(dtype=torch.float32) / normalizer
    if count:
        giou = (1 - _matched_giou(prediction_boxes[positive], target_boxes[positive])).sum(dtype=torch.float32) / normalizer
        ltrb = F.l1_loss(distances[positive], target_ltrb[positive], reduction="sum") / normalizer
    else:
        # Multiplication precedes summation to avoid overflow in a zero term.
        giou = (prediction_boxes * 0).sum(dtype=torch.float32)
        ltrb = (distances * 0).sum(dtype=torch.float32)
    total = 0.5 * classification + 7.5 * giou + ltrb
    if not bool(torch.stack((classification, giou, ltrb, total)).isfinite().all()):
        raise ValueError("FP32 detection loss arithmetic is nonfinite")
    totals = {key: sum(item[key] for item in image_stats) for key in image_stats[0]}
    result = {"loss": total, "classification": classification, "giou": giou, "ltrb": ltrb,
              "num_positive": count, "stats": {"totals": totals, "per_image": image_stats,
                  "normalizer": normalizer, "topk": topk, "class_exponent": float(class_exponent),
                  "iou_exponent": float(iou_exponent), "ltrb_reduction": "four sides sum then positive mean",
                  "decode": "unclamped training pixel xyxy", "assignment_detached": True}}
    if return_details:
        result["assignment"] = {"gt_indices": assigned.detach(), "positive_mask": positive.detach(),
            "classification_eligible_mask": eligible.detach(), "binary_class_targets": target_classes.detach(),
            "box_targets": target_boxes.detach(), "ltrb_targets": target_ltrb.detach(),
            "centers": centers.detach(), "strides": strides.detach(), "per_image": debug}
    return result
