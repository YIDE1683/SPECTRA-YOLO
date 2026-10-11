"""Untrained SPECTRA-A graph specified by contract.json.

This is a floating-point research prototype, not a trained detector or an FPGA
implementation. Quantization, assignment, NMS, training, and tiled execution
are outside this module. All coordinates returned by ``decode`` are in the
letterboxed input coordinate system; undoing letterbox is a caller operation.
"""

from __future__ import annotations

import copy
import math
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F


def _fold_conv_bn(conv: nn.Conv2d, bn: nn.BatchNorm2d) -> tuple[Tensor, Tensor]:
    """Fold evaluation statistics without modifying either input module."""
    if bn.training or not bn.track_running_stats:
        raise ValueError("BN folding requires eval mode and running statistics")
    if bn.running_mean is None or bn.running_var is None:
        raise ValueError("BN running statistics are unavailable")
    bias = conv.bias if conv.bias is not None else torch.zeros_like(bn.running_mean)
    gamma = bn.weight if bn.affine else torch.ones_like(bn.running_mean)
    beta = bn.bias if bn.affine else torch.zeros_like(bn.running_mean)
    scale = gamma * torch.rsqrt(bn.running_var + bn.eps)
    return conv.weight * scale[:, None, None, None], beta + (bias - bn.running_mean) * scale


def _conv_with_parameters(source: nn.Conv2d, kernel: Tensor, bias: Tensor) -> nn.Conv2d:
    result = nn.Conv2d(
        source.in_channels, source.out_channels, source.kernel_size,
        stride=source.stride, padding=source.padding, dilation=source.dilation,
        groups=source.groups, bias=True, padding_mode=source.padding_mode,
        device=kernel.device, dtype=kernel.dtype,
    )
    with torch.no_grad():
        result.weight.copy_(kernel)
        result.bias.copy_(bias)
    result.eval()
    return result


class ConvBN(nn.Module):
    """Linear Conv-BN, with an optional single post-BN ReLU."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
                 stride: int = 1, activate: bool = False) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                              stride=stride, padding=kernel_size // 2, bias=False)
        self.bn: nn.Module = nn.BatchNorm2d(out_channels)
        self.activation = nn.ReLU(inplace=False) if activate else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        return self.activation(self.bn(self.conv(x)))

    def equivalent_parameters(self) -> tuple[Tensor, Tensor]:
        if isinstance(self.bn, nn.Identity):
            assert self.conv.bias is not None
            return self.conv.weight, self.conv.bias
        return _fold_conv_bn(self.conv, self.bn)

    def fold(self) -> None:
        if self.training or any(module.training for module in self.modules()):
            raise ValueError("Conv-BN folding requires all modules in eval mode")
        if isinstance(self.bn, nn.Identity):
            return
        kernel, bias = self.equivalent_parameters()
        self.conv = _conv_with_parameters(self.conv, kernel, bias)
        self.bn = nn.Identity().eval()


class RepBlock(nn.Module):
    """3x3-BN + 1x1-BN + identity-BN, summed before one ReLU."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.branch_3x3 = ConvBN(channels, channels, 3)
        self.branch_1x1 = ConvBN(channels, channels, 1)
        self.identity_bn = nn.BatchNorm2d(channels)
        self.activation = nn.ReLU(inplace=False)
        self.reparam: nn.Conv2d | None = None

    def forward(self, x: Tensor) -> Tensor:
        if self.reparam is not None:
            return self.activation(self.reparam(x))
        return self.activation(self.branch_3x3(x) + self.branch_1x1(x) + self.identity_bn(x))

    def fold(self) -> None:
        if self.training or any(module.training for module in self.modules()):
            raise ValueError("Rep folding requires all modules in eval mode")
        if self.reparam is not None:
            return
        kernel3, bias3 = self.branch_3x3.equivalent_parameters()
        kernel1, bias1 = self.branch_1x1.equivalent_parameters()
        bn = self.identity_bn
        if bn.running_mean is None or bn.running_var is None:
            raise ValueError("Identity BN running statistics are unavailable")
        identity_kernel = torch.zeros_like(kernel3)
        channels = kernel3.shape[0]
        index = torch.arange(channels, device=kernel3.device)
        identity_kernel[index, index, 1, 1] = 1
        scale = bn.weight * torch.rsqrt(bn.running_var + bn.eps)
        kernel_id = identity_kernel * scale[:, None, None, None]
        bias_id = bn.bias - bn.running_mean * scale
        kernel = kernel3 + F.pad(kernel1, (1, 1, 1, 1)) + kernel_id
        bias = bias3 + bias1 + bias_id
        self.reparam = _conv_with_parameters(self.branch_3x3.conv, kernel, bias)
        del self.branch_3x3, self.branch_1x1, self.identity_bn


class BackboneStage(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, repeats: int) -> None:
        super().__init__()
        self.downsample = ConvBN(in_channels, out_channels, 3, stride=2, activate=True)
        self.blocks = nn.Sequential(*(RepBlock(out_channels) for _ in range(repeats)))

    def forward(self, x: Tensor) -> Tensor:
        return self.blocks(self.downsample(x))


class DetectionHead(nn.Module):
    def __init__(self, in_channels: int, tower_channels: int,
                 tower_depth: int, num_classes: int) -> None:
        super().__init__()

        def tower() -> nn.Sequential:
            return nn.Sequential(*(
                ConvBN(in_channels if i == 0 else tower_channels, tower_channels, 3, activate=True)
                for i in range(tower_depth)
            ))

        self.class_tower = tower()
        self.regression_tower = tower()
        self.class_output = nn.Conv2d(tower_channels, num_classes, 1, bias=True)
        self.regression_output = nn.Conv2d(tower_channels, 4, 1, bias=True)
        nn.init.constant_(self.class_output.bias, math.log(0.01 / 0.99))
        nn.init.constant_(self.regression_output.bias, 1.0)

    def forward(self, x: Tensor, stride: int) -> dict[str, Tensor | int]:
        return {
            "class_logits": self.class_output(self.class_tower(x)),
            "distances": F.relu(self.regression_output(self.regression_tower(x))),
            "stride": stride,
        }


class SpectraA(nn.Module):
    """Configurable, untrained floating-point student with static architecture."""

    def __init__(self, p2_channels: int = 0, neck_channels: int = 64,
                 reduced_depth: bool = False, num_classes: int = 6) -> None:
        super().__init__()
        for name, value, choices in (
            ("p2_channels", p2_channels, (0, 32, 64)),
            ("neck_channels", neck_channels, (64, 96)),
            ("num_classes", num_classes, (6, 10)),
        ):
            if type(value) is not int or value not in choices:
                raise ValueError(f"{name} must be one of {choices}, got {value!r}")
        if type(reduced_depth) is not bool:
            raise ValueError("reduced_depth must be a bool")
        self.p2_channels = p2_channels
        self.neck_channels = neck_channels
        self.reduced_depth = reduced_depth
        self.num_classes = num_classes
        self.repeats = (0, 1, 1, 1, 1) if reduced_depth else (0, 1, 2, 2, 1)
        channels = (32, 64, 128, 192, 256)
        self.stages = nn.ModuleList(
            BackboneStage(3 if index == 0 else channels[index - 1], width, self.repeats[index])
            for index, width in enumerate(channels)
        )
        self.laterals = nn.ModuleList(ConvBN(width, neck_channels, 1) for width in channels[2:])
        if p2_channels:
            self.p2_lateral = ConvBN(64, p2_channels, 1)
            self.p2_top = ConvBN(neck_channels, p2_channels, 1)
        self.strides = (4, 8, 16, 32) if p2_channels else (8, 16, 32)
        self.heads = nn.ModuleList()
        if p2_channels:
            self.heads.append(DetectionHead(p2_channels, 32, 1, num_classes))
        self.heads.extend(DetectionHead(neck_channels, 64, 2, num_classes) for _ in range(3))

    def architecture_config(self) -> dict[str, Any]:
        """Fresh deterministic JSON-serializable architecture description."""
        return {
            "name": "SPECTRA-A", "contract_schema_version": 1,
            "p2_channels": self.p2_channels, "neck_channels": self.neck_channels,
            "reduced_depth": self.reduced_depth, "num_classes": self.num_classes,
            "stage_channels": [32, 64, 128, 192, 256],
            "stage_strides": [2, 4, 8, 16, 32], "rep_blocks": list(self.repeats),
            "output_strides": list(self.strides),
            "p2_tower_channels": 32 if self.p2_channels else None,
            "p2_tower_depth": 1 if self.p2_channels else None,
            "p3_p5_tower_channels": 64, "p3_p5_tower_depth": 2,
        }

    def forward(self, x: Tensor) -> list[dict[str, Tensor | int]]:
        if not isinstance(x, Tensor) or x.ndim != 4:
            raise ValueError("input must be an NCHW tensor")
        if not x.is_floating_point():
            raise ValueError("input must have a floating dtype")
        if x.shape[0] <= 0 or x.shape[1] != 3:
            raise ValueError("input requires a positive batch and 3 RGB channels")
        if any(size <= 0 or size % 32 for size in x.shape[-2:]):
            raise ValueError("input H and W must be positive multiples of 32")
        # Keep only actual neck skips. This manages Python references, not
        # allocator peaks or tensors saved by autograd for training backward.
        features = {}
        for index, stage in enumerate(self.stages):
            x = stage(x)
            if index >= 2 or (index == 1 and self.p2_channels):
                features[index] = x
        del x
        lateral3 = self.laterals[0](features.pop(2))
        lateral4 = self.laterals[1](features.pop(3))
        lateral5 = self.laterals[2](features.pop(4))
        if self.p2_channels:
            c2 = features.pop(1)
        del features
        p5 = F.relu(lateral5)
        del lateral5
        p4 = F.relu(lateral4 + F.interpolate(p5, scale_factor=2, mode="nearest"))
        del lateral4
        p3 = F.relu(lateral3 + F.interpolate(p4, scale_factor=2, mode="nearest"))
        del lateral3
        pyramid = [p3, p4, p5]
        del p3, p4, p5
        if self.p2_channels:
            p2_lateral = self.p2_lateral(c2)
            del c2
            p2_top = self.p2_top(pyramid[0])
            p2 = F.relu(p2_lateral + F.interpolate(p2_top, scale_factor=2, mode="nearest"))
            del p2_lateral, p2_top
            pyramid.insert(0, p2)
            del p2
        outputs = []
        for head, stride in zip(self.heads, self.strides):
            feature = pyramid.pop(0)
            outputs.append(head(feature, stride))
            del feature
        return outputs

    def to_deploy(self) -> "SpectraA":
        """Return an independent, eval-mode, BN-free fused copy.

        This proves only floating-point branch equivalence. It does not apply
        quantization and does not export or program an accelerator.
        """
        if any(module.training for module in self.modules()):
            raise ValueError("to_deploy requires the whole model in eval mode")
        result = copy.deepcopy(self)
        for module in list(result.modules()):
            if isinstance(module, RepBlock):
                module.fold()
        for module in list(result.modules()):
            if isinstance(module, ConvBN):
                module.fold()
        return result.eval()

    @staticmethod
    def decode(outputs: list[dict[str, Tensor | int]],
               image_size: tuple[int, int]) -> dict[str, Tensor]:
        return decode(outputs, image_size)


def decode(outputs: list[dict[str, Tensor | int]],
           image_size: tuple[int, int]) -> dict[str, Tensor]:
    """Decode row-major cells; keep zero-area boxes, marking them invalid.

    No score threshold, label assignment, NMS, or letterbox reversal is applied.
    An invalid-box mask is a mandatory input for later postprocessing.
    """
    if (not isinstance(image_size, (tuple, list)) or len(image_size) != 2
            or any(type(size) is not int or size <= 0 for size in image_size)):
        raise ValueError("image_size must contain positive integer (H, W)")
    if not isinstance(outputs, (tuple, list)) or not outputs:
        raise ValueError("outputs must be a nonempty ascending-stride sequence")
    image_h, image_w = image_size
    all_boxes, all_scores, all_valid = [], [], []
    previous_stride = 0
    reference = None
    for output in outputs:
        if not isinstance(output, dict) or not {"class_logits", "distances", "stride"} <= output.keys():
            raise ValueError("every output requires class_logits, distances, and stride")
        logits, distances, stride = output["class_logits"], output["distances"], output["stride"]
        if type(stride) is not int or stride <= previous_stride:
            raise ValueError("strides must be positive, unique, and ascending")
        previous_stride = stride
        if not isinstance(logits, Tensor) or not isinstance(distances, Tensor):
            raise ValueError("logits and distances must be tensors")
        if logits.ndim != 4 or distances.ndim != 4:
            raise ValueError("logits and distances must be NCHW tensors")
        n, classes, height, width = logits.shape
        if (n <= 0 or classes not in (6, 10) or height <= 0 or width <= 0
                or distances.shape != (n, 4, height, width)):
            raise ValueError("invalid or inconsistent output tensor shapes")
        if (height * stride, width * stride) != (image_h, image_w):
            raise ValueError("feature shape times stride must equal image_size")
        if (not logits.is_floating_point() or not distances.is_floating_point()
                or logits.device != distances.device or logits.dtype != distances.dtype):
            raise ValueError("logits/distances require matching floating dtype and device")
        current = (n, classes, logits.dtype, logits.device)
        if reference is not None and current != reference:
            raise ValueError("all levels require matching batch, classes, dtype, and device")
        reference = current
        if (not torch.isfinite(logits).all() or not torch.isfinite(distances).all()
                or (distances < 0).any()):
            raise ValueError("decode requires finite logits and finite nonnegative distances")
        ys, xs = torch.meshgrid(
            torch.arange(height, dtype=logits.dtype, device=logits.device),
            torch.arange(width, dtype=logits.dtype, device=logits.device), indexing="ij")
        center_x, center_y = (xs + 0.5) * stride, (ys + 0.5) * stride
        left, top, right, bottom = distances.unbind(dim=1)
        boxes = torch.stack((
            (center_x - left * stride).clamp(0, image_w),
            (center_y - top * stride).clamp(0, image_h),
            (center_x + right * stride).clamp(0, image_w),
            (center_y + bottom * stride).clamp(0, image_h),
        ), dim=-1).reshape(n, -1, 4)
        valid = (boxes[..., 2] > boxes[..., 0]) & (boxes[..., 3] > boxes[..., 1])
        all_boxes.append(boxes)
        all_scores.append(logits.sigmoid().permute(0, 2, 3, 1).reshape(n, -1, classes))
        all_valid.append(valid)
    return {"boxes": torch.cat(all_boxes, dim=1),
            "scores": torch.cat(all_scores, dim=1),
            "valid": torch.cat(all_valid, dim=1)}
