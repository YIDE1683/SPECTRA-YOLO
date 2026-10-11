"""Feature access for the frozen paper graph without changing its implementation.

This adapter does not implement QAT, assignment, an optimizer, NMS, or integer
deployment. Head-input features are captured for the reconstruction objective.
Calls on a single model must not be concurrent: hooks are scoped to one forward.
"""

from __future__ import annotations

import copy
import hashlib
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn

from tools.spectra_a_prototype.model import SpectraA


CONFIGURATIONS = {
    "B0": {"p2_channels": 0, "reduced_depth": False},
    "B1": {"p2_channels": 0, "reduced_depth": True},
    "F32": {"p2_channels": 32, "reduced_depth": False},
    "R32": {"p2_channels": 32, "reduced_depth": True},
    "R64": {"p2_channels": 64, "reduced_depth": True},
}


def make_model(configuration: str, *, num_classes: int = 6) -> SpectraA:
    """Create the paper's neck-64 graph; all weights are untrained."""
    if configuration not in CONFIGURATIONS:
        raise ValueError(f"Unknown paper configuration: {configuration!r}")
    return SpectraA(neck_channels=64, num_classes=num_classes,
                    **CONFIGURATIONS[configuration])


@dataclass(frozen=True)
class FeatureForward:
    outputs: list[dict[str, Tensor | int]]
    features: tuple[Tensor, ...]
    strides: tuple[int, ...]


def forward_with_features(model: SpectraA, images: Tensor) -> FeatureForward:
    """Return exact tensors used by the existing detection heads.

    Hook registration is undone even when the underlying model rejects input.
    Captured tensors are not detached, copied or altered: gradients can reach
    the model, and outputs retain the unchanged prototype's semantics.
    """
    if not isinstance(model, SpectraA):
        raise TypeError("Expected the existing SpectraA prototype")
    captured: list[list[Tensor]] = [[] for _ in model.heads]
    handles = []

    def capture(index: int):
        def hook(_module: nn.Module, args: tuple[Any, ...]) -> None:
            if not args or not isinstance(args[0], Tensor):
                raise RuntimeError("Head input contract changed")
            captured[index].append(args[0])
        return hook

    try:
        for index, head in enumerate(model.heads):
            handles.append(head.register_forward_pre_hook(capture(index)))
        outputs = model(images)
    finally:
        for handle in handles:
            handle.remove()
    if any(len(values) != 1 for values in captured):
        raise RuntimeError("Each detection head must be called exactly once")
    return FeatureForward(outputs=outputs,
                          features=tuple(values[0] for values in captured),
                          strides=tuple(model.strides))


def freeze_reference(model: SpectraA) -> SpectraA:
    """Independent same-architecture copy with frozen parameters and BN state."""
    if not isinstance(model, SpectraA):
        raise TypeError("Expected the existing SpectraA prototype")
    reference = copy.deepcopy(model).eval()
    reference.requires_grad_(False)
    return reference


def state_sha256(model: nn.Module) -> str:
    """Stable digest over named parameter/buffer values, shapes and dtypes.

    This hashes model state, not serialized pickle bytes; no checkpoint is read.
    Names are sorted and byte records are length-prefixed to avoid ambiguity.
    """
    digest = hashlib.sha256()

    def add(value: bytes) -> None:
        digest.update(len(value).to_bytes(8, "little"))
        digest.update(value)

    for name, tensor in sorted(model.state_dict().items()):
        add(name.encode("utf-8"))
        add(str(tensor.dtype).encode("ascii"))
        add(str(tuple(tensor.shape)).encode("ascii"))
        raw = tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8)
        add(raw.numpy().tobytes())
    return digest.hexdigest()
