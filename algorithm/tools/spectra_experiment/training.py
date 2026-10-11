"""Deterministic CPU FP32 supervised training building blocks.

This is not QAT, a trained model, a benchmark result, or an approval to start a
paper experiment. Raw data remain owned by a caller-supplied verified Dataset.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile

import torch
from torch.nn import functional as F

from tools.spectra_rewrite_training.detection import detection_loss
from tools.spectra_rewrite_training.model_adapter import make_model, state_sha256


def stable_seed(seed: int, epoch: int, token: str) -> int:
    payload = json.dumps([seed, epoch, token], ensure_ascii=False, separators=(",", ":")).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little") % (2**63 - 1)


def file_sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def json_sha256(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    allow_nan=False, separators=(",", ":")).encode()).hexdigest()


def full_neu_identity(dataset) -> dict:
    """Derive source/policy identity from the actual verified loader, not a label."""
    from tools.spectra_experiment.dataset import FullNeuDataset
    if not isinstance(dataset, FullNeuDataset):
        raise TypeError("Expected the explicit-policy FullNeuDataset")
    return {"kind": "full_neu_existing_split", "manifest_sha256": dataset.manifest_sha256,
            "split": dataset.split, "coordinate_policy": dataset.coordinate_policy,
            "difficult_policy": dataset.difficult_policy, "image_size": dataset.image_size,
            "class_names": list(dataset.class_names), "letterbox_source_sha256": dataset.letterbox_source_sha256}


@dataclass(frozen=True)
class TrainingConfig:
    configuration: str = "R32"
    num_classes: int = 6
    image_size: int = 640
    epochs: int = 300
    batch_size: int = 16
    seed: int = 17
    learning_rate: float = 0.01
    momentum: float = 0.937
    weight_decay: float = 0.0005
    warmup_epochs: int = 3
    horizontal_flip: float = 0.5
    scale_min: float = 0.8
    scale_max: float = 1.2

    def validate(self):
        if self.configuration not in ("B0", "B1", "F32", "R32", "R64"):
            raise ValueError("Unknown frozen architecture")
        for key in ("num_classes", "image_size", "epochs", "batch_size"):
            value = getattr(self, key)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{key} must be a positive integer")
        if self.image_size < 64 or self.image_size % 32:
            raise ValueError("image_size must be >=64 and divisible by32")
        if type(self.seed) is not int or not 0 <= self.seed < 2**63:
            raise ValueError("seed must be a nonnegative int64")
        if type(self.warmup_epochs) is not int or not 0 <= self.warmup_epochs < self.epochs:
            raise ValueError("warmup_epochs must be in [0,epochs)")
        for key in ("learning_rate", "momentum", "weight_decay", "horizontal_flip", "scale_min", "scale_max"):
            value = getattr(self, key)
            if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value):
                raise ValueError(f"{key} must be finite")
        if self.learning_rate <= 0 or not 0 <= self.momentum < 1 or self.weight_decay < 0:
            raise ValueError("Invalid SGD parameters")
        if not 0 <= self.horizontal_flip <= 1 or not 0.5 <= self.scale_min <= self.scale_max <= 2:
            raise ValueError("Invalid augmentation parameters")


def learning_rate_at(config: TrainingConfig, step: int, steps_per_epoch: int) -> float:
    config.validate()
    if type(steps_per_epoch) is not int or steps_per_epoch <= 0:
        raise ValueError("steps_per_epoch must be positive")
    total = config.epochs * steps_per_epoch
    if type(step) is not int or not 0 <= step < total:
        raise ValueError("step is outside the training schedule")
    warmup = config.warmup_epochs * steps_per_epoch
    if step < warmup:
        return config.learning_rate * (step + 1) / warmup
    remaining = total - warmup
    # A one-step post-warmup phase retains base LR; otherwise its last step is0.
    progress = (step - warmup) / max(remaining - 1, 1)
    return config.learning_rate * 0.5 * (1 + math.cos(math.pi * progress))


def _geometry_boxes(boxes, scale_x, scale_y, offset_x, offset_y, width, height, flip):
    if not isinstance(boxes, torch.Tensor) or boxes.dtype != torch.float32 or boxes.device.type != "cpu":
        raise ValueError("Augmentation expects CPU FP32 xyxy")
    if boxes.ndim != 2 or boxes.shape[1] != 4 or not torch.isfinite(boxes).all():
        raise ValueError("Malformed augmentation boxes")
    if (boxes.numel() and ((boxes[:, 2:] <= boxes[:, :2]).any()
            or (boxes[:, :2] < 0).any() or (boxes[:, 2] > width).any() or (boxes[:, 3] > height).any())):
        raise ValueError("Source boxes must have positive area and be inside the source canvas")
    output = boxes.detach().clone()
    output[:, [0, 2]] = output[:, [0, 2]] * scale_x + offset_x
    output[:, [1, 3]] = output[:, [1, 3]] * scale_y + offset_y
    output[:, [0, 2]] = output[:, [0, 2]].clamp(0, width)
    output[:, [1, 3]] = output[:, [1, 3]].clamp(0, height)
    if flip:
        left = output[:, 0].clone()
        output[:, 0] = width - output[:, 2]
        output[:, 2] = width - left
    keep = (output[:, 2] > output[:, 0]) & (output[:, 3] > output[:, 1])
    return output[keep], keep


def augment_sample(sample, config: TrainingConfig, epoch: int):
    """Center scale/crop/pad, then horizontal flip; all choices hash-keyed.

    Jitter is applied to the letterboxed tensor with bilinear align_corners=False,
    antialias=True. Integer resized dimensions use floor(x+0.5). Actual two-axis
    scales are used for boxes. Padding114/255, no mosaic or color augmentation.
    Clipped partial GT is retained; zero-area GT is dropped and counted.
    """
    config.validate()
    if type(epoch) is not int or epoch < 0:
        raise ValueError("epoch must be nonnegative")
    sample_id = sample.get("sample_id")
    if not isinstance(sample_id, str) or not sample_id:
        raise ValueError("sample_id is required")
    source = sample["image"]
    expected = (3, config.image_size, config.image_size)
    if (not isinstance(source, torch.Tensor) or source.dtype != torch.float32
            or source.device.type != "cpu" or tuple(source.shape) != expected
            or not torch.isfinite(source).all() or source.min() < 0 or source.max() > 1):
        raise ValueError("Expected finite CPU FP32 CHW RGB [0,1] at configured size")
    labels = sample["labels"]
    if (not isinstance(labels, torch.Tensor) or labels.dtype != torch.int64 or labels.device.type != "cpu"
            or labels.ndim != 1 or len(labels) != len(sample["boxes"])
            or (labels.numel() and ((labels < 0).any() or (labels >= config.num_classes).any()))):
        raise ValueError("Malformed training labels")
    generator = torch.Generator().manual_seed(stable_seed(config.seed, epoch, sample_id))
    scale = config.scale_min + float(torch.rand((), generator=generator)) * (config.scale_max - config.scale_min)
    flip = float(torch.rand((), generator=generator)) < config.horizontal_flip
    height, width = source.shape[-2:]
    new_h, new_w = math.floor(height * scale + 0.5), math.floor(width * scale + 0.5)
    scale_x, scale_y = new_w / width, new_h / height
    # Odd remainders go to right/bottom; integer offsets avoid half-pixel boxes.
    offset_x = (width - new_w) // 2 if new_w <= width else -((new_w - width) // 2)
    offset_y = (height - new_h) // 2 if new_h <= height else -((new_h - height) // 2)
    resized = F.interpolate(source[None], size=(new_h, new_w), mode="bilinear",
                            align_corners=False, antialias=True)[0]
    image = torch.full_like(source, 114 / 255)
    dst_x, dst_y = max(offset_x, 0), max(offset_y, 0)
    src_x, src_y = max(-offset_x, 0), max(-offset_y, 0)
    copy_w, copy_h = min(width - dst_x, new_w - src_x), min(height - dst_y, new_h - src_y)
    image[:, dst_y:dst_y + copy_h, dst_x:dst_x + copy_w] = resized[:, src_y:src_y + copy_h, src_x:src_x + copy_w]
    if flip:
        image = image.flip(-1)
    boxes, keep = _geometry_boxes(sample["boxes"], scale_x, scale_y, offset_x, offset_y, width, height, flip)
    ignored, ignored_keep = _geometry_boxes(sample["ignore_boxes"], scale_x, scale_y, offset_x, offset_y, width, height, flip)
    region = sample["valid_region"]
    if not isinstance(region, torch.Tensor) or tuple(region.shape) != (4,):
        raise ValueError("valid_region must have shape[4]")
    valid, valid_keep = _geometry_boxes(region[None], scale_x, scale_y, offset_x, offset_y, width, height, flip)
    if len(valid) != 1 or not bool(valid_keep[0]):
        raise ValueError("Augmentation removed the full valid content region")
    target = {"boxes": boxes, "labels": labels[keep].clone(), "ignore_boxes": ignored,
              "valid_region": valid[0], "image_hw": (height, width)}
    provenance = {"sample_id": sample_id, "source_group": sample.get("source_group"),
                  "epoch": epoch, "scale_draw": scale, "scale_x": scale_x, "scale_y": scale_y,
                  "offset_x": offset_x, "offset_y": offset_y, "horizontal_flip": flip,
                  "gt_clipped_to_zero": int((~keep).sum()), "ignore_clipped_to_zero": int((~ignored_keep).sum())}
    return image, target, provenance


class SupervisedTrainer:
    """Single-process CPU trainer with exact epoch/batch-position resume.

    Dataset identity must describe source hashes/policy, not merely a path.
    Real-data orchestration and approval are deliberately not hidden in this API.
    """
    def __init__(self, dataset, config: TrainingConfig, *, dataset_identity: dict, model=None):
        config.validate()
        if not isinstance(dataset_identity, dict) or not dataset_identity:
            raise ValueError("An explicit source/policy dataset_identity is required")
        json_sha256(dataset_identity)
        self.dataset, self.config = dataset, config
        self.sample_ids = list(dataset.sample_ids)
        if (not self.sample_ids or len(self.sample_ids) != len(dataset)
                or any(not isinstance(s, str) or not s for s in self.sample_ids)
                or len(set(self.sample_ids)) != len(self.sample_ids)):
            raise ValueError("Dataset requires complete unique nonempty sample_ids")
        if getattr(dataset, "split", "train") != "train":
            raise ValueError("Trainer may only consume a training split")
        self.identity = json.loads(json.dumps(dataset_identity, allow_nan=False))
        from tools.spectra_experiment.dataset import FullNeuDataset
        self._full_neu = isinstance(dataset, FullNeuDataset)
        if self._full_neu and (self.identity != full_neu_identity(dataset)
                or config.image_size != dataset.image_size or config.num_classes != len(dataset.class_names)):
            raise ValueError("Identity/configuration does not match the actual NEU loader protocol")
        self.data_signature = json_sha256({"identity": self.identity, "sample_ids": self.sample_ids})
        self.steps_per_epoch = math.ceil(len(dataset) / config.batch_size)
        self.total_steps = config.epochs * self.steps_per_epoch
        torch.manual_seed(config.seed)
        torch.set_num_threads(2)
        torch.use_deterministic_algorithms(True)
        self.model = model if model is not None else make_model(config.configuration, num_classes=config.num_classes)
        if any(p.device.type != "cpu" or p.dtype != torch.float32 for p in self.model.parameters()):
            raise ValueError("This trainer is CPU FP32 only")
        self.model.train()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=config.learning_rate,
                                        momentum=config.momentum, weight_decay=config.weight_decay)
        self.global_step = 0
        self.failed = False

    def _batch(self):
        if self._full_neu and full_neu_identity(self.dataset) != self.identity:
            raise ValueError("NEU source/protocol changed after trainer construction")
        epoch, batch_index = divmod(self.global_step, self.steps_per_epoch)
        generator = torch.Generator().manual_seed(stable_seed(self.config.seed, epoch, "sample-order"))
        order = torch.randperm(len(self.dataset), generator=generator).tolist()
        indices = order[batch_index * self.config.batch_size:(batch_index + 1) * self.config.batch_size]
        samples = [self.dataset[i] for i in indices]
        if any(s["sample_id"] != self.sample_ids[i] for s, i in zip(samples, indices)):
            raise ValueError("Dataset sample order/identity changed")
        augmented = [augment_sample(s, self.config, epoch) for s in samples]
        images = torch.stack([s[0] for s in augmented])
        return images, [s[1] for s in augmented], [s[2] for s in augmented]

    def step(self) -> dict:
        if self.failed:
            raise RuntimeError("Trainer is invalid after a failed step; restore a verified checkpoint or recreate it")
        if self.global_step >= self.total_steps:
            raise StopIteration("Configured training schedule is complete")
        try:
            return self._step_impl()
        except BaseException:
            # Forward may already have changed BN buffers; an optimizer error
            # or interruption may leave partial state. Never silently retry it.
            self.failed = True
            raise

    def _step_impl(self) -> dict:
        self.model.train()
        images, targets, augmentation = self._batch()
        lr = learning_rate_at(self.config, self.global_step, self.steps_per_epoch)
        for group in self.optimizer.param_groups:
            group["lr"] = lr
        self.optimizer.zero_grad(set_to_none=True)
        values = detection_loss(self.model(images), targets, tuple(images.shape[-2:]),
            valid_regions=[t["valid_region"] for t in targets], ignore_boxes=[t["ignore_boxes"] for t in targets])
        values["loss"].backward()
        gradients = [p.grad for p in self.model.parameters() if p.grad is not None]
        if not gradients or any(not bool(torch.isfinite(g).all()) for g in gradients):
            raise FloatingPointError("Missing/nonfinite parameter gradients; optimizer step cancelled")
        gradient_sum = sum(float(g.detach().abs().sum()) for g in gradients)
        self.optimizer.step()
        if any(not bool(torch.isfinite(s).all()) for s in self.model.state_dict().values()):
            raise FloatingPointError("Nonfinite state after optimizer step; stop and discard failed state")
        for state in self.optimizer.state.values():
            if any(isinstance(v, torch.Tensor) and not bool(torch.isfinite(v).all()) for v in state.values()):
                raise FloatingPointError("Nonfinite optimizer state")
        epoch, batch_index = divmod(self.global_step, self.steps_per_epoch)
        record = {"global_step": self.global_step, "epoch": epoch, "batch_index": batch_index,
                  "learning_rate": lr, "sample_ids": [a["sample_id"] for a in augmentation],
                  "augmentation": augmentation, "loss": float(values["loss"].detach()),
                  "classification": float(values["classification"].detach()),
                  "giou": float(values["giou"].detach()), "ltrb": float(values["ltrb"].detach()),
                  "num_positive": values["num_positive"], "assignment_stats": values["stats"],
                  "gradient_abs_sum": gradient_sum, "model_state_sha256": state_sha256(self.model)}
        self.global_step += 1
        return record

    def run_steps(self, count: int) -> list[dict]:
        if type(count) is not int or count < 0 or self.global_step + count > self.total_steps:
            raise ValueError("Requested steps exceed the remaining schedule")
        return [self.step() for _ in range(count)]

    def save_checkpoint(self, path: str | Path) -> str:
        if self.failed:
            raise RuntimeError("Failed training state cannot be saved as a resumable checkpoint")
        path = Path(path)
        if path.exists():
            raise FileExistsError("Existing checkpoint will not be overwritten")
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"schema_version": 1, "scope": "supervised_fp32_state_not_paper_acceptance",
                   "configuration": asdict(self.config), "dataset_identity": self.identity,
                   "dataset_signature": self.data_signature, "global_step": self.global_step,
                   "steps_per_epoch": self.steps_per_epoch, "model": self.model.state_dict(),
                   "model_state_sha256": state_sha256(self.model), "optimizer": self.optimizer.state_dict(),
                   "torch_rng_state": torch.get_rng_state()}
        temporary = None
        try:
            # Publish only fully written bytes. Same-directory hard-link creation
            # is atomic/no-replace on the supported filesystem; no overwrite
            # fallback is used when the filesystem does not support hard links.
            with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".checkpoint-", suffix=".tmp", delete=False) as stream:
                temporary = Path(stream.name)
                torch.save(payload, stream)
                stream.flush()
                os.fsync(stream.fileno())
            os.link(temporary, path)
        finally:
            if temporary is not None:
                # Remove only this invocation's exclusive-created temporary file.
                temporary.unlink(missing_ok=True)
        return file_sha256(path)

    def load_checkpoint(self, path: str | Path, *, expected_sha256: str):
        path = Path(path)
        if not isinstance(expected_sha256, str) or len(expected_sha256) != 64:
            raise ValueError("A pinned checkpoint SHA256 is required")
        if file_sha256(path) != expected_sha256:
            raise ValueError("Checkpoint bytes differ from pinned hash")
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if file_sha256(path) != expected_sha256:
            raise ValueError("Checkpoint changed during read")
        if (payload.get("schema_version") != 1 or payload.get("configuration") != asdict(self.config)
                or payload.get("dataset_signature") != self.data_signature
                or payload.get("dataset_identity") != self.identity
                or payload.get("steps_per_epoch") != self.steps_per_epoch):
            raise ValueError("Checkpoint configuration/source/policy mismatch")
        step = payload.get("global_step")
        if type(step) is not int or not 0 <= step <= self.total_steps:
            raise ValueError("Checkpoint step is outside this schedule")
        # Validate on separate objects so rejected checkpoints do not partially
        # replace live parameters/optimizer/RNG state.
        restored_model = copy.deepcopy(self.model)
        restored_model.load_state_dict(payload["model"], strict=True)
        if (state_sha256(restored_model) != payload.get("model_state_sha256")
                or any(not bool(torch.isfinite(t).all()) for t in restored_model.state_dict().values())):
            raise ValueError("Restored model value hash mismatch")
        restored_optimizer = torch.optim.SGD(restored_model.parameters(), lr=self.config.learning_rate,
            momentum=self.config.momentum, weight_decay=self.config.weight_decay)
        expected_groups = restored_optimizer.state_dict()["param_groups"]
        saved_optimizer = payload["optimizer"]
        groups = saved_optimizer["param_groups"]
        if len(groups) != len(expected_groups):
            raise ValueError("Checkpoint optimizer group count mismatch")
        for group, expected in zip(groups, expected_groups):
            if set(group) != set(expected) or any(
                    type(group[k]) is not type(expected[k]) or group[k] != expected[k]
                    for k in expected if k != "lr"):
                raise ValueError("Checkpoint SGD flags/parameters mismatch")
        parameter_list = list(restored_model.parameters())
        for parameter_id, state in saved_optimizer["state"].items():
            if (type(parameter_id) is not int or not 0 <= parameter_id < len(parameter_list)
                    or not isinstance(state, dict) or set(state) - {"momentum_buffer"}):
                raise ValueError("Checkpoint SGD state schema mismatch")
            if "momentum_buffer" in state:
                buffer, parameter = state["momentum_buffer"], parameter_list[parameter_id]
                if (not isinstance(buffer, torch.Tensor) or buffer.shape != parameter.shape
                        or buffer.dtype != parameter.dtype or not bool(torch.isfinite(buffer).all())):
                    raise ValueError("Checkpoint momentum buffer shape/dtype/value mismatch")
        restored_optimizer.load_state_dict(saved_optimizer)
        for state in restored_optimizer.state.values():
            if any(isinstance(v, torch.Tensor) and not bool(torch.isfinite(v).all()) for v in state.values()):
                raise ValueError("Checkpoint contains nonfinite optimizer state")
        for group in restored_optimizer.param_groups:
            if (not math.isfinite(group["lr"]) or group["lr"] < 0
                    or group["momentum"] != self.config.momentum or group["weight_decay"] != self.config.weight_decay):
                raise ValueError("Checkpoint optimizer hyperparameters mismatch")
        torch.Generator().set_state(payload["torch_rng_state"])
        self.model, self.optimizer = restored_model, restored_optimizer
        torch.set_rng_state(payload["torch_rng_state"])
        self.global_step = step
        self.failed = False
