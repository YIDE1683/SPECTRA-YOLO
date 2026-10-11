"""New-manuscript Eq. (4)/(5) feature reconstruction losses, not detector AP.

These functions deliberately do not reuse the earlier prototype's summed-energy
ROI loss: each object is independently normalized and then equally weighted.
Only the quantized branch receives gradients; the reference is always detached.
No dataset I/O, model training, board access, or scientific result generation is
performed in this module.

The prediction conversion checks gradient finiteness before and after returning
to the input dtype. This is only a loss-entry check: external AMP scaling,
other branches/losses, and accumulated parameter gradients remain the caller's
responsibility. The formulas and epsilon are never changed to hide overflow.
"""

from __future__ import annotations

import math
from numbers import Real
from typing import Sequence

import torch
from torch import Tensor


class _GradientCheckedFloat32(torch.autograd.Function):
    """Unchanged FP32 conversion with a fail-closed backward dtype boundary."""

    @staticmethod
    def forward(ctx, value: Tensor) -> Tensor:
        ctx.input_dtype = value.dtype
        return value.float()

    @staticmethod
    def backward(ctx, gradient: Tensor):
        if not bool(torch.isfinite(gradient).all()):
            raise FloatingPointError(
                "Nonfinite reconstruction gradient before input-dtype conversion; "
                "use a FP32 forward and inspect upstream loss scaling/gradients."
            )
        converted = gradient.to(dtype=ctx.input_dtype)
        if not bool(torch.isfinite(converted).all()):
            raise FloatingPointError(
                f"Reconstruction gradient overflows {ctx.input_dtype}; a FP32 forward "
                "is required for this case. No clipping or epsilon change was applied."
            )
        return converted


def _finite_float(value: Tensor, name: str) -> None:
    if not isinstance(value, Tensor) or not value.is_floating_point():
        raise ValueError(f"{name} must be a floating tensor")
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"{name} contains a nonfinite value")


def _positive_epsilon(eps: float) -> float:
    bounds = torch.finfo(torch.float32)
    if (isinstance(eps, bool) or not isinstance(eps, Real) or not math.isfinite(eps)
            or not bounds.tiny <= eps <= bounds.max):
        raise ValueError("eps must be finite, positive, and normal-representable in FP32")
    return float(eps)


def _feature_pairs(quantized_features: Sequence[Tensor], reference_features: Sequence[Tensor]):
    if (not isinstance(quantized_features, (tuple, list))
            or not isinstance(reference_features, (tuple, list))
            or not quantized_features or len(quantized_features) != len(reference_features)):
        raise ValueError("features must be nonempty equal-length lists/tuples")
    batch, device = None, None
    pairs = []
    for index, (prediction, reference) in enumerate(zip(quantized_features, reference_features)):
        _finite_float(prediction, f"quantized_features[{index}]")
        _finite_float(reference, f"reference_features[{index}]")
        if (prediction.ndim != 4 or min(prediction.shape) <= 0
                or prediction.shape != reference.shape):
            raise ValueError("each paired feature must have identical nonempty NCHW shape")
        if prediction.device != reference.device:
            raise ValueError("each feature/reference pair must have the same device")
        if batch is None:
            batch, device = prediction.shape[0], prediction.device
        elif prediction.shape[0] != batch or prediction.device != device:
            raise ValueError("all feature levels require the same batch and device")
        prediction32 = _GradientCheckedFloat32.apply(prediction)
        reference32 = reference.detach().float()
        if not bool(torch.isfinite(prediction32).all() & torch.isfinite(reference32).all()):
            raise ValueError("features overflow when converted to required FP32 reduction dtype")
        pairs.append((prediction32, reference32))
    return pairs, batch


def _image_sizes(image_hw, batch: int) -> list[tuple[int, int]]:
    def valid_pair(pair):
        return (isinstance(pair, (tuple, list)) and len(pair) == 2
                and all(type(size) is int and 0 < size <= 2**31 - 1 for size in pair))

    if valid_pair(image_hw):
        return [tuple(image_hw)] * batch
    if (isinstance(image_hw, (tuple, list)) and len(image_hw) == batch
            and all(valid_pair(pair) for pair in image_hw)):
        return [tuple(pair) for pair in image_hw]
    raise ValueError("image_hw requires shared (H,W) or one positive integer (H,W) pair per image")


def _clipped_boxes(boxes_xyxy, image_sizes):
    if not isinstance(boxes_xyxy, (tuple, list)) or len(boxes_xyxy) != len(image_sizes):
        raise ValueError("boxes_xyxy requires one floating [Mi,4] tensor per image")
    clipped = []
    for index, (boxes, (height, width)) in enumerate(zip(boxes_xyxy, image_sizes)):
        _finite_float(boxes, f"boxes_xyxy[{index}]")
        if boxes.ndim != 2 or boxes.shape[1] != 4:
            raise ValueError("each box tensor must have shape [Mi,4], including empty [0,4]")
        current = []
        for left, top, right, bottom in boxes.detach().double().cpu().tolist():
            if right <= left or bottom <= top:
                raise ValueError("box coordinates require positive width and height")
            left, right = max(0.0, min(width, left)), max(0.0, min(width, right))
            top, bottom = max(0.0, min(height, top)), max(0.0, min(height, bottom))
            if right <= left or bottom <= top:
                raise ValueError("box has no positive intersection with its image")
            current.append((left, top, right, bottom))
        clipped.append(current)
    return clipped


def _region_cells(box, image_hw, feature_hw):
    image_h, image_w = image_hw
    feature_h, feature_w = feature_hw
    left, top, right, bottom = box
    x1 = max(0, min(feature_w - 1, math.floor(left / image_w * feature_w)))
    y1 = max(0, min(feature_h - 1, math.floor(top / image_h * feature_h)))
    x2 = max(x1 + 1, min(feature_w, math.ceil(right / image_w * feature_w)))
    y2 = max(y1 + 1, min(feature_h, math.ceil(bottom / image_h * feature_h)))
    return x1, y1, x2, y2


def _squared_maps(prediction32: Tensor, reference32: Tensor):
    error, energy = (prediction32 - reference32).square(), reference32.square()
    if not bool(torch.isfinite(error).all() & torch.isfinite(energy).all()):
        raise ValueError("FP32 feature squared-error/energy arithmetic overflow")
    return error, energy


def _normalized_mean(error: Tensor, energy: Tensor, eps: float) -> Tensor:
    if error.numel() == 0 or error.shape != energy.shape:
        raise ValueError("internal normalized region must be nonempty with matching shapes")
    # FP32 mean is explicit even if the forward pass is autocast or FP16.
    numerator = error.mean(dtype=torch.float32)
    denominator = energy.mean(dtype=torch.float32) + eps
    if not bool(torch.isfinite(numerator) & torch.isfinite(denominator)):
        raise ValueError("FP32 region mean/denominator arithmetic overflow")
    value = numerator / denominator
    if not bool(torch.isfinite(value)):
        raise ValueError("FP32 normalized region loss arithmetic overflow")
    return value


def object_balanced_reconstruction_loss(
    quantized_features: Sequence[Tensor],
    reference_features: Sequence[Tensor],
    boxes_xyxy: Sequence[Tensor],
    image_hw,
    *,
    eps: float = 1e-6,
    background_weight: float = 0.1,
    return_details: bool = False,
):
    """Eq. (4)/(5): object mean + weighted union-complement background.

    ``boxes_xyxy[n]`` contains human GT pixel coordinates in the image described
    by ``image_hw[n]`` (or a shared ``(H,W)``). Inputs must already correspond to
    the same crop/letterbox used by the features; this helper does not undo data
    augmentation. Positive-area partially out-of-image boxes are clipped. Boxes
    with no positive image intersection, reversed/zero-area/nonfinite boxes are
    rejected, not silently removed. Floor(left/top), ceil(right/bottom), and
    clipping yield at least one cell for each valid box.

    All channels and cells within an object form that object's independent
    error/energy means. Overlaps contribute to each object separately. Object
    values are equally averaged; the background is the complement of their
    union and contributes once with weight 0.1. Empty object/background sets
    contribute differentiable zero. Images are averaged equally within each
    level, then available levels are averaged equally. A variable object count
    must NOT change per-image weighting. The reference is detached and all
    reductions/output are FP32. Eq. (6)'s coefficient 0.01 is applied by caller.
    Backward raises FloatingPointError if the incoming FP32 gradient is not
    finite or cannot be cast to the prediction's original dtype finitely.
    This guard does not certify external AMP or accumulated parameter gradients;
    callers must inspect every model parameter gradient before optimizer steps.

    Returns a scalar Tensor, or ``(scalar, details)`` when requested. Details
    contain detached Python values and are diagnostics, not accuracy metrics.
    """
    eps = _positive_epsilon(eps)
    if (isinstance(background_weight, bool) or not isinstance(background_weight, Real)
            or not math.isfinite(background_weight) or background_weight < 0
            or background_weight > torch.finfo(torch.float32).max):
        raise ValueError("background_weight must be finite, nonnegative, and FP32-representable")
    if type(return_details) is not bool:
        raise ValueError("return_details must be bool")
    pairs, batch = _feature_pairs(quantized_features, reference_features)
    sizes = _image_sizes(image_hw, batch)
    boxes = _clipped_boxes(boxes_xyxy, sizes)
    layer_values, layer_details = [], []
    for level, (prediction, reference) in enumerate(pairs):
        error, energy = _squared_maps(prediction, reference)
        feature_hw = prediction.shape[-2:]
        image_values, image_details = [], []
        for image in range(batch):
            zero = prediction[image].reshape(-1)[0] * 0
            union = torch.zeros(feature_hw, dtype=torch.bool, device=prediction.device)
            object_values, mapped = [], []
            for box in boxes[image]:
                x1, y1, x2, y2 = _region_cells(box, sizes[image], feature_hw)
                object_values.append(_normalized_mean(error[image, :, y1:y2, x1:x2],
                                                       energy[image, :, y1:y2, x1:x2], eps))
                union[y1:y2, x1:x2] = True
                if return_details:
                    mapped.append([x1, y1, x2, y2])
            object_mean = torch.stack(object_values).mean(dtype=torch.float32) if object_values else zero
            background = ~union
            background_error = (_normalized_mean(error[image, :, background], energy[image, :, background], eps)
                                if bool(background.any()) else zero)
            value = object_mean + float(background_weight) * background_error
            if not bool(torch.isfinite(value)):
                raise ValueError("FP32 object/background loss arithmetic overflow")
            image_values.append(value)
            if return_details:
                image_details.append({
                    "image": image, "objects": len(object_values), "mapped_xyxy": mapped,
                    "object_errors": [float(item.detach()) for item in object_values],
                    "object_mean": float(object_mean.detach()), "background_cells": int(background.sum()),
                    "background_error": float(background_error.detach()), "loss": float(value.detach()),
                })
        layer_value = torch.stack(image_values).mean(dtype=torch.float32)
        layer_values.append(layer_value)
        if return_details:
            layer_details.append({"level": level, "feature_hw": list(feature_hw),
                                  "images": image_details, "loss": float(layer_value.detach())})
    loss = torch.stack(layer_values).mean(dtype=torch.float32)
    if not bool(torch.isfinite(loss)):
        raise ValueError("FP32 batch/level mean arithmetic overflow")
    if not return_details:
        return loss
    return loss, {"loss_name": "object_balanced_region_mean", "eps": eps,
                  "background_weight": float(background_weight), "batch_size": batch,
                  "reduction": "object mean; image mean within level; level mean",
                  "reference_detached": True, "reduction_dtype": "float32", "layers": layer_details}


def global_reconstruction_loss(
    quantized_features: Sequence[Tensor],
    reference_features: Sequence[Tensor],
    *,
    eps: float = 1e-6,
    return_details: bool = False,
):
    """Control: full-map normalized MSE, image mean then matching-level mean.

    Each map ratio is mean((quantized-reference)^2)/(mean(reference^2)+eps).
    No boxes, foreground weighting, or background coefficient are applied.
    The same reference-detachment and FP32 reduction contract as Eq. (4) holds.
    Gradient conversion uses the same fail-closed finite checks as the object
    loss; callers must separately validate external AMP/multi-loss accumulation
    and all parameter gradients before an optimizer step.
    """
    eps = _positive_epsilon(eps)
    if type(return_details) is not bool:
        raise ValueError("return_details must be bool")
    pairs, batch = _feature_pairs(quantized_features, reference_features)
    values, details = [], []
    for level, (prediction, reference) in enumerate(pairs):
        error, energy = _squared_maps(prediction, reference)
        images = [_normalized_mean(error[index], energy[index], eps) for index in range(batch)]
        level_value = torch.stack(images).mean(dtype=torch.float32)
        values.append(level_value)
        if return_details:
            details.append({"level": level, "image_losses": [float(item.detach()) for item in images],
                            "loss": float(level_value.detach())})
    loss = torch.stack(values).mean(dtype=torch.float32)
    if not bool(torch.isfinite(loss)):
        raise ValueError("FP32 batch/level global mean arithmetic overflow")
    if not return_details:
        return loss
    return loss, {"loss_name": "global_energy_normalized_mean", "eps": eps, "batch_size": batch,
                  "reduction": "image mean within level; level mean", "reference_detached": True,
                  "reduction_dtype": "float32", "layers": details}
