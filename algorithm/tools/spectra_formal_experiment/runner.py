"""Minimal deterministic FP32 trainer/evaluator for the frozen split adapter."""
from __future__ import annotations

import copy
import hashlib
import json
import math
import os
from pathlib import Path

import torch

from tools.spectra_experiment.evaluation import evaluate_predictions, postprocess
from tools.spectra_experiment.training import TrainingConfig, augment_sample, learning_rate_at, stable_seed
from tools.spectra_rewrite_training.detection import detection_loss
from tools.spectra_rewrite_training.model_adapter import make_model, state_sha256

CRITICAL_CODE_PATHS = (
    "tools/spectra_formal_experiment/cli.py", "tools/spectra_formal_experiment/dataset.py",
    "tools/spectra_formal_experiment/runner.py", "tools/spectra_experiment/dataset.py",
    "tools/spectra_experiment/evaluation.py", "tools/spectra_experiment/training.py",
    "tools/spectra_rewrite_training/data.py", "tools/spectra_rewrite_training/detection.py",
    "tools/spectra_rewrite_training/model_adapter.py", "tools/spectra_a_prototype/model.py",
)


def file_sha256(path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""): digest.update(chunk)
    return digest.hexdigest()


def canonical_sha(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
        ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def critical_code_hashes() -> dict[str, str]:
    root = Path(__file__).resolve().parents[2]
    return {relative: file_sha256(root / relative) for relative in CRITICAL_CODE_PATHS}


def _digest(value, name):
    if (not isinstance(value, str) or len(value) != 64
            or any(c not in "0123456789abcdef" for c in value)):
        raise ValueError(f"{name} must be a lowercase SHA256")
    return value


def _device(value: str) -> torch.device:
    if value not in ("cpu", "cuda"):
        raise ValueError("device must be cpu or cuda")
    if value == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if value == "cuda":
        # torch.device("cuda") has index=None while tensors materialize on the
        # current concrete device. Normalize once and preserve strict index checks.
        return torch.device("cuda", torch.cuda.current_device())
    return torch.device("cpu")


def _move_targets(targets, device):
    moved = []
    for target in targets:
        item = {}
        for key, value in target.items():
            item[key] = value.to(device) if isinstance(value, torch.Tensor) else value
        moved.append(item)
    return moved


class FormalFPTrainer:
    """B0/B1/F32/R32/R64 FP32 optimizer with exact, hash-pinned resume.

    No AMP/QAT/reference reconstruction is hidden here.  The model, images,
    targets and optimizer tensors share one explicit CPU or CUDA device.
    """
    def __init__(self, dataset, config: TrainingConfig, *, source_run_identity,
                 source_run_manifest_sha256, device="cpu", model=None):
        config.validate()
        if getattr(dataset, "split", None) != "train":
            raise ValueError("Training consumes the train role only")
        self.identity = dataset.identity()
        self.source_run_identity = _digest(source_run_identity, "source_run_identity")
        self.source_run_manifest_sha256 = _digest(source_run_manifest_sha256, "source_run_manifest_sha256")
        if self.identity.get("split_role") != "train" or config.image_size != dataset.image_size:
            raise ValueError("Dataset identity/config mismatch")
        self.dataset, self.config, self.device = dataset, config, _device(device)
        self.sample_ids = list(dataset.sample_ids)
        if len(self.sample_ids) != len(dataset) or len(set(self.sample_ids)) != len(self.sample_ids):
            raise ValueError("Dataset sample IDs are incomplete or duplicate")
        self.data_signature = canonical_sha({"identity": self.identity, "sample_ids": self.sample_ids})
        self.steps_per_epoch = math.ceil(len(dataset) / config.batch_size)
        self.total_steps = config.epochs * self.steps_per_epoch
        torch.set_num_threads(2); torch.manual_seed(config.seed)
        if self.device.type == "cuda": torch.cuda.manual_seed_all(config.seed)
        torch.use_deterministic_algorithms(True)
        self.model = (model if model is not None else make_model(config.configuration,
                      num_classes=config.num_classes)).to(self.device, dtype=torch.float32)
        if any(p.device != self.device or p.dtype != torch.float32 for p in self.model.parameters()):
            raise ValueError("Every parameter must be FP32 on the requested device")
        self.initial_state_sha256 = state_sha256(self.model)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=config.learning_rate,
                                         momentum=config.momentum, weight_decay=config.weight_decay)
        self.global_step, self.failed = 0, False

    def _batch(self):
        epoch, batch_index = divmod(self.global_step, self.steps_per_epoch)
        order = torch.randperm(len(self.dataset), generator=torch.Generator().manual_seed(
            stable_seed(self.config.seed, epoch, "sample-order"))).tolist()
        indices = order[batch_index*self.config.batch_size:(batch_index+1)*self.config.batch_size]
        samples = [self.dataset[i] for i in indices]
        if [x["sample_id"] for x in samples] != [self.sample_ids[i] for i in indices]:
            raise ValueError("Dataset identity/order changed")
        augmented = [augment_sample(x, self.config, epoch) for x in samples]
        images = torch.stack([x[0] for x in augmented]).to(self.device)
        targets = _move_targets([x[1] for x in augmented], self.device)
        if images.dtype != torch.float32 or any(any(isinstance(v, torch.Tensor) and v.device != self.device
             for v in t.values()) for t in targets):
            raise RuntimeError("Image/target device invariant failed")
        return images, targets, [x[2] for x in augmented]

    def step(self):
        if self.failed: raise RuntimeError("Trainer is invalid after a failed step")
        if self.global_step >= self.total_steps: raise StopIteration
        try:
            self.model.train(); images, targets, augmentation = self._batch()
            lr = learning_rate_at(self.config, self.global_step, self.steps_per_epoch)
            for group in self.optimizer.param_groups: group["lr"] = lr
            self.optimizer.zero_grad(set_to_none=True)
            values = detection_loss(self.model(images), targets, tuple(images.shape[-2:]),
                valid_regions=[x["valid_region"] for x in targets],
                ignore_boxes=[x["ignore_boxes"] for x in targets])
            values["loss"].backward()
            grads = [p.grad for p in self.model.parameters() if p.grad is not None]
            if not grads or any(not torch.isfinite(g).all() for g in grads):
                raise FloatingPointError("Missing or nonfinite gradient")
            self.optimizer.step()
            if any(not torch.isfinite(x).all() for x in self.model.state_dict().values()):
                raise FloatingPointError("Nonfinite model state")
            epoch, batch = divmod(self.global_step, self.steps_per_epoch)
            record = {"global_step": self.global_step, "epoch": epoch, "batch_index": batch,
                "learning_rate": lr, "sample_ids": [x["sample_id"] for x in augmentation],
                "loss": float(values["loss"].detach()), "classification": float(values["classification"].detach()),
                "giou": float(values["giou"].detach()), "ltrb": float(values["ltrb"].detach()),
                "num_positive": values["num_positive"]}
            self.global_step += 1
            return record
        except BaseException:
            self.failed = True
            raise

    def run_to_epoch_boundary(self):
        stop = min(self.total_steps, ((self.global_step // self.steps_per_epoch) + 1) * self.steps_per_epoch)
        return [self.step() for _ in range(stop - self.global_step)]

    def save_checkpoint(self, path):
        if self.failed:
            raise RuntimeError("Failed training state cannot be saved as a resumable checkpoint")
        path = Path(path)
        if path.exists(): raise FileExistsError(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        config = vars(self.config).copy()
        payload = {"format": "spectra_formal_fp_v1", "config": config, "dataset_identity": self.identity,
            "data_signature": self.data_signature, "device": str(self.device),
            "critical_code_sha256": critical_code_hashes(),
            "source_run_identity": self.source_run_identity,
            "source_run_manifest_sha256": self.source_run_manifest_sha256,
            "initialization": {"kind": "random_from_scratch", "configuration": self.config.configuration,
                               "seed": self.config.seed,
                               "initial_state_sha256": self.initial_state_sha256},
            "global_step": self.global_step, "model_state": {k: v.detach().cpu() for k,v in self.model.state_dict().items()},
            "model_state_sha256": state_sha256(self.model), "optimizer_state": self.optimizer.state_dict(),
            "rng_state": torch.get_rng_state(),
            "cuda_rng_state_all": torch.cuda.get_rng_state_all() if self.device.type == "cuda" else None}
        tmp = path.with_name(path.name + ".partial")
        if tmp.exists(): raise FileExistsError(tmp)
        try:
            torch.save(payload, tmp)
            os.link(tmp, path); tmp.unlink()
        finally:
            if tmp.exists(): tmp.unlink()
        return {"path": str(path.resolve()), "sha256": file_sha256(path),
                "model_state_sha256": payload["model_state_sha256"], "global_step": self.global_step}

    def load_checkpoint(self, path, expected_sha256, *, expected_source_run_identity,
                        expected_source_run_manifest_sha256):
        path = Path(path).resolve(strict=True)
        if (not isinstance(expected_sha256, str) or len(expected_sha256) != 64
                or any(c not in "0123456789abcdef" for c in expected_sha256)):
            raise ValueError("A lowercase pinned checkpoint SHA256 is required")
        if file_sha256(path) != expected_sha256: raise ValueError("Checkpoint file SHA256 mismatch")
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if file_sha256(path) != expected_sha256: raise ValueError("Checkpoint changed during read")
        if payload.get("critical_code_sha256") != critical_code_hashes():
            raise ValueError("Checkpoint critical-code identity differs from current code")
        if (payload.get("source_run_identity") != _digest(expected_source_run_identity,
                "expected_source_run_identity") or payload.get("source_run_manifest_sha256") !=
                _digest(expected_source_run_manifest_sha256, "expected_source_run_manifest_sha256")):
            raise ValueError("Checkpoint source-run identity/manifest mismatch")
        if payload.get("format") != "spectra_formal_fp_v1" or payload.get("config") != vars(self.config):
            raise ValueError("Checkpoint format/config mismatch")
        if payload.get("device") != str(self.device):
            raise ValueError("Checkpoint execution device mismatch")
        if payload.get("dataset_identity") != self.identity or payload.get("data_signature") != self.data_signature:
            raise ValueError("Checkpoint split/source identity mismatch")
        init = payload.get("initialization", {})
        if (init.get("kind") != "random_from_scratch"
                or init.get("configuration") != self.config.configuration
                or init.get("initial_state_sha256") != self.initial_state_sha256):
            raise ValueError("Random initialization identity mismatch")
        step = payload.get("global_step")
        if type(step) is not int or not 0 <= step <= self.total_steps: raise ValueError("Invalid checkpoint step")
        try:
            restored_model = copy.deepcopy(self.model)
            restored_model.load_state_dict(payload["model_state"], strict=True)
        except (KeyError, TypeError, RuntimeError) as exc:
            raise ValueError("Checkpoint model schema mismatch") from exc
        if (state_sha256(restored_model) != payload.get("model_state_sha256")
                or any(not bool(torch.isfinite(t).all()) for t in restored_model.state_dict().values())):
            raise ValueError("Checkpoint model-state hash/value mismatch")
        restored_optimizer = torch.optim.SGD(restored_model.parameters(), lr=self.config.learning_rate,
            momentum=self.config.momentum, weight_decay=self.config.weight_decay)
        saved_optimizer = payload.get("optimizer_state")
        if not isinstance(saved_optimizer, dict) or set(saved_optimizer) != {"state", "param_groups"}:
            raise ValueError("Checkpoint optimizer top-level schema mismatch")
        groups, expected_groups = saved_optimizer["param_groups"], restored_optimizer.state_dict()["param_groups"]
        if not isinstance(groups, list) or len(groups) != len(expected_groups):
            raise ValueError("Checkpoint optimizer group count mismatch")
        for group, expected in zip(groups, expected_groups):
            if (not isinstance(group, dict) or set(group) != set(expected)
                    or any(type(group[key]) is not type(expected[key]) or group[key] != expected[key]
                           for key in expected if key != "lr")
                    or isinstance(group.get("lr"), bool) or not isinstance(group.get("lr"), (int, float))
                    or not math.isfinite(group["lr"]) or group["lr"] < 0):
                raise ValueError("Checkpoint SGD flags/parameter mapping mismatch")
        states = saved_optimizer["state"]
        parameters = list(restored_model.parameters())
        if not isinstance(states, dict):
            raise ValueError("Checkpoint SGD state must be a mapping")
        for parameter_id, state in states.items():
            if (type(parameter_id) is not int or not 0 <= parameter_id < len(parameters)
                    or not isinstance(state, dict) or set(state) - {"momentum_buffer"}):
                raise ValueError("Checkpoint SGD state schema mismatch")
            if "momentum_buffer" in state:
                buffer, parameter = state["momentum_buffer"], parameters[parameter_id]
                if (not isinstance(buffer, torch.Tensor) or buffer.shape != parameter.shape
                        or buffer.dtype != parameter.dtype or not bool(torch.isfinite(buffer).all())):
                    raise ValueError("Checkpoint momentum shape/dtype/value mismatch")
        try:
            restored_optimizer.load_state_dict(saved_optimizer)
            torch.Generator().set_state(payload["rng_state"])
        except (KeyError, TypeError, RuntimeError, ValueError) as exc:
            raise ValueError("Checkpoint optimizer/RNG restore rejected") from exc
        for state in restored_optimizer.state.values():
            if any(not isinstance(value, torch.Tensor) or not bool(torch.isfinite(value).all())
                   for value in state.values()):
                raise ValueError("Checkpoint optimizer contains unsupported/nonfinite state")
        cuda_rng = payload.get("cuda_rng_state_all")
        if self.device.type == "cpu" and cuda_rng is not None:
            raise ValueError("CPU checkpoint unexpectedly contains CUDA RNG state")
        if self.device.type == "cuda":
            if (not isinstance(cuda_rng, list) or len(cuda_rng) != torch.cuda.device_count()
                    or any(not isinstance(value, torch.Tensor) or value.dtype != torch.uint8
                           or value.device.type != "cpu" for value in cuda_rng)):
                raise ValueError("Checkpoint CUDA RNG schema mismatch")
        self.model, self.optimizer = restored_model, restored_optimizer
        torch.set_rng_state(payload["rng_state"])
        if self.device.type == "cuda": torch.cuda.set_rng_state_all(cuda_rng)
        self.global_step, self.failed = step, False


@torch.no_grad()
def collect_evaluation_inputs(model, dataset, *, device="cpu", batch_size=16, score_threshold=.001,
                              nms_iou=.6, max_detections=300):
    """Collect serializable predictions and GT; metrics are intentionally not computed here."""
    dev = _device(device)
    if getattr(dataset, "split", None) not in ("tune", "evaluation"):
        raise ValueError("Evaluation consumes tune or evaluation role only")
    if type(batch_size) is not int or batch_size <= 0: raise ValueError("batch_size must be positive")
    model = model.to(dev, dtype=torch.float32).eval()
    predictions, ground_truth = [], []
    for start in range(0, len(dataset), batch_size):
        samples = [dataset[i] for i in range(start, min(start+batch_size, len(dataset)))]
        images = torch.stack([x["image"] for x in samples]).to(dev)
        outputs = model(images)
        cpu_outputs = [{"class_logits": x["class_logits"].detach().cpu(),
                        "distances": x["distances"].detach().cpu(), "stride": x["stride"]} for x in outputs]
        metadata = [{"sample_id": x["sample_id"], "transform": x["transform"]} for x in samples]
        predictions.extend(postprocess(cpu_outputs, metadata, score_threshold, nms_iou, max_detections))
        ground_truth.extend({"sample_id": x["sample_id"], "boxes": x["original_boxes"],
                             "labels": x["original_labels"]} for x in samples)
    serial = [{"sample_id": x["sample_id"], "boxes": x["boxes"].tolist(), "scores": x["scores"].tolist(),
               "labels": x["labels"].tolist(), "source_config": x["source_config"]} for x in predictions]
    if [x["sample_id"] for x in serial] != list(dataset.sample_ids):
        raise RuntimeError("Evaluation did not emit every frozen ID exactly once")
    return serial, ground_truth


def validate_prediction_archive(records, expected_sample_ids, num_classes):
    """Strictly rebuild evaluator tensors from a JSON-decoded frozen archive."""
    if not isinstance(records, list) or len(records) != len(expected_sample_ids):
        raise ValueError("Prediction archive must contain every expected ID exactly once")
    rebuilt = []
    for index, (record, expected_id) in enumerate(zip(records, expected_sample_ids)):
        if not isinstance(record, dict) or set(record) != {"sample_id", "boxes", "scores", "labels", "source_config"}:
            raise ValueError(f"Prediction archive record {index} has an unexpected schema")
        if record["sample_id"] != expected_id or not isinstance(record["source_config"], dict):
            raise ValueError("Prediction archive ID order/source configuration mismatch")
        raw_boxes, raw_scores, raw_labels = record["boxes"], record["scores"], record["labels"]
        if (not isinstance(raw_boxes, list) or any(not isinstance(box, list) or len(box) != 4
                or any(isinstance(value, bool) or not isinstance(value, (int, float))
                       or not math.isfinite(value) for value in box) for box in raw_boxes)
                or not isinstance(raw_scores, list) or any(isinstance(value, bool)
                    or not isinstance(value, (int, float)) or not math.isfinite(value) for value in raw_scores)
                or not isinstance(raw_labels, list) or any(type(value) is not int for value in raw_labels)):
            raise ValueError("Prediction archive numeric JSON types/values are invalid")
        try:
            boxes = torch.tensor(raw_boxes, dtype=torch.float32).reshape(-1, 4)
            scores = torch.tensor(raw_scores, dtype=torch.float32).reshape(-1)
            labels = torch.tensor(raw_labels, dtype=torch.int64).reshape(-1)
        except (TypeError, ValueError, RuntimeError) as exc:
            raise ValueError("Prediction archive contains malformed numeric arrays") from exc
        if len(boxes) != len(scores) or len(scores) != len(labels):
            raise ValueError("Prediction archive boxes/scores/labels lengths disagree")
        if (not bool(torch.isfinite(boxes).all()) or not bool(torch.isfinite(scores).all())
                or bool((scores < 0).any()) or bool((scores > 1).any())
                or bool((labels < 0).any()) or bool((labels >= num_classes).any())
                or (boxes.numel() and bool((boxes[:, 2:] <= boxes[:, :2]).any()))):
            raise ValueError("Prediction archive contains nonfinite/out-of-range/zero-area values")
        transform = record["source_config"].get("transform")
        original_hw = transform.get("original_hw") if isinstance(transform, dict) else None
        if (not isinstance(original_hw, (list, tuple)) or len(original_hw) != 2
                or any(type(value) is not int or value <= 0 for value in original_hw)):
            raise ValueError("Prediction archive source transform/original size is invalid")
        height, width = original_hw
        if (boxes.numel() and (bool((boxes[:, :2] < 0).any())
                or bool((boxes[:, 2] > width).any()) or bool((boxes[:, 3] > height).any()))):
            raise ValueError("Prediction archive box lies outside its original image")
        if scores.numel() > 1 and bool((scores[1:] > scores[:-1]).any()):
            raise ValueError("Prediction archive scores are not in stable descending order")
        rebuilt.append({"sample_id": expected_id, "boxes": boxes, "scores": scores, "labels": labels})
    return rebuilt


def run_evaluation(model, dataset, *, device="cpu", batch_size=16, score_threshold=.001,
                   nms_iou=.6, max_detections=300):
    """In-memory convenience for tests; formal CLI evaluates a reread frozen JSON archive."""
    serial, ground_truth = collect_evaluation_inputs(model, dataset, device=device, batch_size=batch_size,
        score_threshold=score_threshold, nms_iou=nms_iou, max_detections=max_detections)
    # JSON round-trip exercises the same archive schema, but only the formal CLI
    # provides durable atomic persistence and a pinned file hash.
    decoded = json.loads(json.dumps(serial, allow_nan=False))
    predictions = validate_prediction_archive(decoded, dataset.sample_ids, len(dataset.class_names))
    return serial, evaluate_predictions(ground_truth, predictions, dataset.class_names, max_detections)


def verify_all_payloads(datasets, *, expected_objects=None):
    """Traverse every role row and trigger byte/XML/RGB/label checks with zero optimizer steps."""
    roles = {dataset.split: dataset for dataset in datasets}
    if set(roles) != {"train", "tune", "evaluation"}:
        raise ValueError("verify-all requires exactly one dataset for every frozen role")
    seen, output, total_objects = set(), {}, 0
    for role in ("train", "tune", "evaluation"):
        dataset = roles[role]
        class_counts = [0] * len(dataset.class_names); difficult = objects = 0
        for index in range(len(dataset)):
            sample = dataset[index]
            if sample["sample_id"] in seen: raise ValueError("verify-all found a cross-role duplicate ID")
            seen.add(sample["sample_id"])
            if (sample["image"].dtype != torch.float32 or not bool(torch.isfinite(sample["image"]).all())
                    or len(sample["boxes"]) != len(sample["labels"])):
                raise ValueError("verify-all found malformed image/label tensors")
            objects += len(sample["raw_objects"])
            difficult += sum(int(obj["difficult"]) for obj in sample["raw_objects"])
            for label in sample["labels"].tolist(): class_counts[label] += 1
        total_objects += objects
        output[role] = {"images": len(dataset), "objects": objects, "difficult_objects": difficult,
                        "class_object_counts": dict(zip(dataset.class_names, class_counts)),
                        "dataset_identity": dataset.identity()}
    expected_images = sum(next(iter(roles.values())).contract.role_counts)
    if len(seen) != expected_images or (expected_objects is not None and total_objects != expected_objects):
        raise ValueError("verify-all total image/object counts differ from the frozen protocol")
    return {"roles": output, "total_images": len(seen), "total_objects": total_objects,
            "optimizer_steps": 0, "status": "all_payloads_reverified"}
