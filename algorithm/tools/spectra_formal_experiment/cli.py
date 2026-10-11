"""Explicit entry point for the preregistered FP32 experiment lifecycle."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import platform
import sys
import time
import traceback
from pathlib import Path

import torch

from tools.spectra_experiment.training import TrainingConfig
from tools.spectra_experiment.evaluation import evaluate_predictions
from tools.spectra_rewrite_training.model_adapter import make_model, state_sha256
from .dataset import (FROZEN_MANIFEST_SHA256, FROZEN_SPLIT_SHA256,
                      FrozenSplitContract, FrozenSplitDataset)
from .runner import (FormalFPTrainer, canonical_sha, collect_evaluation_inputs, critical_code_hashes,
                     file_sha256, validate_prediction_archive, verify_all_payloads)


def _atomic_json(path: Path, value):
    tmp = path.with_name(path.name + ".partial")
    with tmp.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False); stream.write("\n")
        stream.flush(); os.fsync(stream.fileno())
    os.replace(tmp, path)


def _jsonl(path: Path, value):
    with path.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")
        stream.flush(); os.fsync(stream.fileno())


def _formal_split_contract(supplied_sha256):
    if supplied_sha256 != FROZEN_SPLIT_SHA256:
        raise ValueError("--split-sha256 must equal the code-pinned formal CSV SHA256")
    return FrozenSplitContract.neu_main_v1()


def _dataset(args, role):
    contract = _formal_split_contract(args.split_sha256)
    return FrozenSplitDataset(args.manifest, args.raw_root, args.split_csv, role,
                              contract=contract, image_size=640)


def _config(args):
    batch_size = args.batch_size if args.mode == "probe" else 16
    return TrainingConfig(configuration=args.configuration, image_size=640, batch_size=batch_size,
                          seed=args.seed, epochs=300)


def _create_output(path):
    output = Path(path).resolve()
    output.mkdir(parents=True, exist_ok=False)
    return output


def _probe_count(value):
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError("probe-steps must be an integer") from exc
    if not 1 <= result <= 16:
        raise argparse.ArgumentTypeError("probe-steps must be in [1,16]")
    return result


def _probe_start(value):
    try: result = int(value)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError("probe-start-step must be an integer") from exc
    if result < 0: raise argparse.ArgumentTypeError("probe-start-step must be nonnegative")
    return result


def _probe_window(dataset_size, batch_size, start_step, count):
    if any(type(value) is not int for value in (dataset_size, batch_size, start_step, count)):
        raise ValueError("probe window inputs must be integers")
    if dataset_size <= 0 or batch_size <= 0 or count <= 0:
        raise ValueError("probe dataset, batch and count must be positive")
    steps_per_epoch = (dataset_size + batch_size - 1) // batch_size
    if not 0 <= start_step < steps_per_epoch:
        raise ValueError("probe-start-step must be in [0, steps_per_epoch-1]")
    if start_step + count > steps_per_epoch:
        raise ValueError("probe may not cross its single fresh-model epoch")
    counts = [min(batch_size, dataset_size - step * batch_size)
              for step in range(start_step, start_step + count)]
    return {"steps_per_epoch": steps_per_epoch, "batch_indices": list(range(start_step, start_step + count)),
            "sample_counts": counts}


def _run_probe(trainer, count):
    """Run exactly 1..16 optimizer steps; never creates a reusable checkpoint."""
    if type(count) is not int or not 1 <= count <= 16:
        raise ValueError("probe step count must be in [1,16]")
    if trainer.global_step + count > trainer.total_steps:
        raise ValueError("probe steps exceed the configured schedule")
    return [trainer.step() for _ in range(count)]


def _synchronize(device):
    if device.type == "cuda": torch.cuda.synchronize(device)


def _reset_cuda_peak(device):
    if device.type == "cuda":
        _synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)


def _cuda_memory(device):
    if device.type != "cuda":
        return {"cuda_current_allocated_bytes": None, "cuda_peak_allocated_bytes": None,
                "cuda_current_reserved_bytes": None, "cuda_peak_reserved_bytes": None}
    _synchronize(device)
    return {"cuda_current_allocated_bytes": int(torch.cuda.memory_allocated(device)),
            "cuda_peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
            "cuda_current_reserved_bytes": int(torch.cuda.memory_reserved(device)),
            "cuda_peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device))}


def _measure_probe(trainer, count):
    """Measure exactly the bounded optimizer steps after explicit device synchronization."""
    _reset_cuda_peak(trainer.device)
    _synchronize(trainer.device)
    started = time.perf_counter()
    records = _run_probe(trainer, count)
    _synchronize(trainer.device)
    wall = time.perf_counter() - started
    return records, {"measurement_scope": "NOT_FOR_RESULTS host/GPU software probe; not board or paper performance",
                     "wall_time_seconds": wall, "seconds_per_step": wall / len(records),
                     **_cuda_memory(trainer.device)}


def _measure_epoch(trainer, checkpoint_path):
    """Measure one optimizer epoch and its checkpoint without changing either operation."""
    _reset_cuda_peak(trainer.device)
    _synchronize(trainer.device)
    epoch_started = time.perf_counter()
    optimizer_started = time.perf_counter()
    records = trainer.run_to_epoch_boundary()
    _synchronize(trainer.device)
    optimizer_wall = time.perf_counter() - optimizer_started
    checkpoint_started = time.perf_counter()
    checkpoint = trainer.save_checkpoint(checkpoint_path)
    _synchronize(trainer.device)
    checkpoint_wall = time.perf_counter() - checkpoint_started
    epoch_wall = time.perf_counter() - epoch_started
    measurement = {"measurement_scope": "measured host/GPU FP32 execution; not board or paper performance",
                   "optimizer_wall_time_seconds": optimizer_wall,
                   "checkpoint_wall_time_seconds": checkpoint_wall,
                   "epoch_total_wall_time_seconds": epoch_wall,
                   **_cuda_memory(trainer.device)}
    return records, checkpoint, measurement


def _nested_equal(left, right):
    if isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor):
        return left.device == right.device and left.dtype == right.dtype and torch.equal(left, right)
    if type(left) is not type(right): return False
    if isinstance(left, dict):
        return set(left) == set(right) and all(_nested_equal(left[key], right[key]) for key in left)
    if isinstance(left, (list, tuple)):
        return len(left) == len(right) and all(_nested_equal(a, b) for a, b in zip(left, right))
    return left == right


def _probe_checkpoint_roundtrip(trainer, dataset, config, checkpoint_path):
    path = Path(checkpoint_path)
    if path.name != "NOT_FOR_RESULTS_probe_roundtrip.pt":
        raise ValueError("Probe roundtrip checkpoint requires the fixed NOT_FOR_RESULTS filename")
    checkpoint = trainer.save_checkpoint(path)
    restored = FormalFPTrainer(dataset, config, device=str(trainer.device.type),
        source_run_identity=trainer.source_run_identity,
        source_run_manifest_sha256=trainer.source_run_manifest_sha256)
    restored.load_checkpoint(path, checkpoint["sha256"],
        expected_source_run_identity=trainer.source_run_identity,
        expected_source_run_manifest_sha256=trainer.source_run_manifest_sha256)
    passed = (restored.global_step == trainer.global_step
              and state_sha256(restored.model) == state_sha256(trainer.model)
              and _nested_equal(restored.optimizer.state_dict(), trainer.optimizer.state_dict())
              and restored.device == trainer.device)
    if not passed: raise RuntimeError("Probe checkpoint roundtrip state comparison failed")
    return {"roundtrip_passed": True, "checkpoint": checkpoint,
            "restored_global_step": restored.global_step, "restored_device": str(restored.device),
            "model_state_sha256": state_sha256(restored.model),
            "scope": "NOT_FOR_RESULTS probe checkpoint; prohibited as formal resume/evaluation source"}


def _runtime_identity(args):
    gpu = None
    if torch.cuda.is_available():
        index = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(index)
        gpu = {"index": index, "device": str(torch.device("cuda", index)),
               "name": torch.cuda.get_device_name(index), "total_memory_bytes": props.total_memory,
               "compute_capability": [props.major, props.minor], "device_count": torch.cuda.device_count()}
    return {"requested_device": args.device, "python_version": platform.python_version(),
            "python_executable": sys.executable, "platform": platform.platform(),
            "torch_version": torch.__version__, "torch_cuda_version": torch.version.cuda,
            "cudnn_version": torch.backends.cudnn.version(), "cuda_available": torch.cuda.is_available(),
            "gpu_current": gpu, "code_sha256": critical_code_hashes()}


def _resume_source(args):
    path = Path(args.source_run_manifest).resolve(strict=True)
    if path.name != "run.final.json":
        raise ValueError("Formal source must be an explicit run.final.json")
    expected = args.source_run_manifest_sha256
    if (not isinstance(expected, str) or len(expected) != 64
            or any(c not in "0123456789abcdef" for c in expected)):
        raise ValueError("A lowercase source-run manifest SHA256 is required")
    final_content = path.read_bytes()
    if hashlib.sha256(final_content).hexdigest() != expected:
        raise ValueError("Explicit source-run manifest SHA256 mismatch")
    try: final = json.loads(final_content)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("Explicit source-run manifest is malformed") from exc
    if (not isinstance(final, dict) or final.get("format") != "spectra_formal_run_v1"
            or final.get("status") != "passed" or final.get("mode") not in ("train", "resume")
            or not isinstance(final.get("result"), dict)
            or final.get("run_identity_sha256") != canonical_sha(final["result"])):
        raise ValueError("Formal source must be a passed train/resume final manifest")
    initial_path = path.with_name("run.json")
    try: initial_content = initial_path.read_bytes(); initial = json.loads(initial_content)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("Formal source requires its original sibling run.json") from exc
    if (not isinstance(initial, dict) or initial.get("format") != "spectra_formal_run_v1"
            or initial.get("status") != "running" or initial.get("mode") != final["mode"]):
        raise ValueError("Explicit source-run manifest has the wrong format")
    expected_keys = set(initial) | {"result", "run_identity_sha256"}
    if set(final) != expected_keys or any(final.get(key) != value for key, value in initial.items() if key != "status"):
        raise ValueError("Source run.json and run.final.json identity fields disagree")
    return canonical_sha(initial), hashlib.sha256(initial_content).hexdigest()


def execute(args) -> dict:
    """Execute one explicit mode. Output directory creation always refuses overwrite."""
    run_started = time.perf_counter()
    output = _create_output(args.output_dir)
    run = {"format": "spectra_formal_run_v1", "mode": args.mode,
           "configuration": args.configuration, "status": "running",
           "training_config": vars(_config(args)), "runtime": _runtime_identity(args),
           "input_chain": {"manifest_path": str(Path(args.manifest).resolve()),
                           "manifest_sha256_expected": FROZEN_MANIFEST_SHA256,
                           "split_csv_path": str(Path(args.split_csv).resolve()),
                           "split_csv_sha256_expected": FROZEN_SPLIT_SHA256,
                           "split_csv_sha256_supplied": args.split_sha256,
                           "checkpoint_path": str(Path(args.checkpoint).resolve()) if hasattr(args, "checkpoint") else None,
                           "checkpoint_sha256_expected": getattr(args, "checkpoint_sha256", None),
                           "source_run_manifest_path": str(Path(args.source_run_manifest).resolve())
                                if hasattr(args, "source_run_manifest") else None,
                           "source_run_manifest_sha256_expected":
                                getattr(args, "source_run_manifest_sha256", None)},
           "evidence_scope": "formal real-data path" if args.mode not in ("probe", "dry-run", "verify-all") else
                             "bounded preflight only; not a scientific result"}
    _atomic_json(output / "run.json", run)
    current_run_manifest_sha256 = file_sha256(output / "run.json")
    current_run_identity = canonical_sha(run)
    try:
        if args.mode == "dry-run":
            datasets = [_dataset(args, role) for role in ("train", "tune", "evaluation")]
            if args.device == "cuda" and not torch.cuda.is_available():
                raise RuntimeError("CUDA was requested but is unavailable")
            torch.manual_seed(args.seed)
            actual_device = torch.device("cuda", torch.cuda.current_device()) if args.device == "cuda" else torch.device("cpu")
            model = make_model(args.configuration).to(actual_device, dtype=torch.float32)
            result = {"configuration": args.configuration, "training_config": vars(_config(args)),
                "device": str(actual_device),
                "identities": [x.identity() for x in datasets], "initialization":
                {"kind": "random_from_scratch", "configuration": args.configuration, "seed": args.seed,
                 "initial_state_sha256": state_sha256(model)}, "optimizer_steps": 0}
        elif args.mode in ("train", "resume"):
            trainer = FormalFPTrainer(_dataset(args, "train"), _config(args), device=args.device,
                source_run_identity=current_run_identity,
                source_run_manifest_sha256=current_run_manifest_sha256)
            if args.mode == "resume":
                source_identity, source_manifest_sha = _resume_source(args)
                trainer.load_checkpoint(args.checkpoint, args.checkpoint_sha256,
                    expected_source_run_identity=source_identity,
                    expected_source_run_manifest_sha256=source_manifest_sha)
            records = output / "epochs.jsonl"
            checkpoints = []
            while trainer.global_step < trainer.total_steps:
                next_epoch = trainer.global_step // trainer.steps_per_epoch + 1
                epoch_records, ckpt, measurement = _measure_epoch(
                    trainer, output / f"epoch_{next_epoch:04d}.pt")
                summary = {"epoch_completed": trainer.global_step // trainer.steps_per_epoch,
                           "steps": epoch_records, "checkpoint": ckpt,
                           "model_state_sha256": ckpt["model_state_sha256"],
                           **measurement}
                _jsonl(records, summary); checkpoints.append(ckpt)
            result = {"configuration": args.configuration, "training_config": vars(trainer.config),
                      "device": str(trainer.device), "dataset_identity": trainer.identity,
                      "global_step": trainer.global_step, "checkpoints": checkpoints,
                      "initialization": {"kind": "random_from_scratch",
                                         "configuration": args.configuration, "seed": args.seed,
                                         "initial_state_sha256": trainer.initial_state_sha256}}
        elif args.mode == "probe":
            probe_dataset = _dataset(args, "train")
            trainer = FormalFPTrainer(probe_dataset, _config(args), device=args.device,
                source_run_identity=current_run_identity,
                source_run_manifest_sha256=current_run_manifest_sha256)
            window = _probe_window(len(probe_dataset), trainer.config.batch_size,
                                   args.probe_start_step, args.probe_steps)
            trainer.global_step = args.probe_start_step
            step_records, measurement = _measure_probe(trainer, args.probe_steps)
            actual_indices = [record["batch_index"] for record in step_records]
            actual_counts = [len(record["sample_ids"]) for record in step_records]
            if actual_indices != window["batch_indices"] or actual_counts != window["sample_counts"]:
                raise RuntimeError("Probe batch index/sample-count contract mismatch")
            roundtrip = None
            if args.checkpoint_roundtrip:
                roundtrip = _probe_checkpoint_roundtrip(trainer, probe_dataset, trainer.config,
                    output / "NOT_FOR_RESULTS_probe_roundtrip.pt")
            probe_payload = {"evidence_scope": "bounded optimizer-path check only; not a scientific result",
                             "requested_steps": args.probe_steps, "actual_steps": len(step_records),
                             "sample_ids": [record["sample_ids"] for record in step_records],
                             "actual_batch_indices": actual_indices, "actual_sample_counts": actual_counts,
                             "probe_start_step": args.probe_start_step,
                             "trajectory_gap": args.probe_start_step,
                             "not_a_resume": True,
                             "trajectory_note": "fresh random model; skipped optimizer steps were not replayed",
                             "steps": step_records, "checkpoint_written": bool(roundtrip),
                             "checkpoint_roundtrip": roundtrip, **measurement}
            _atomic_json(output / "probe_steps.json", probe_payload)
            result = {"configuration": args.configuration, **probe_payload,
                      "training_config": vars(trainer.config), "device": str(trainer.device),
                      "dataset_identity": trainer.identity,
                      "probe_steps_sha256": file_sha256(output / "probe_steps.json"),
                      "initialization": {"kind": "random_from_scratch",
                                         "configuration": args.configuration, "seed": args.seed,
                                         "initial_state_sha256": trainer.initial_state_sha256}}
        elif args.mode == "evaluate":
            trainer = FormalFPTrainer(_dataset(args, "train"), _config(args), device=args.device,
                source_run_identity=current_run_identity,
                source_run_manifest_sha256=current_run_manifest_sha256)
            source_identity, source_manifest_sha = _resume_source(args)
            trainer.load_checkpoint(args.checkpoint, args.checkpoint_sha256,
                expected_source_run_identity=source_identity,
                expected_source_run_manifest_sha256=source_manifest_sha)
            evaluation_dataset = _dataset(args, args.role)
            predictions, ground_truth = collect_evaluation_inputs(trainer.model, evaluation_dataset,
                device=args.device, batch_size=args.eval_batch_size)
            archive = output / "predictions.json"
            _atomic_json(archive, predictions)
            prediction_sha = file_sha256(archive)
            archived_bytes = archive.read_bytes()
            if hashlib.sha256(archived_bytes).hexdigest() != prediction_sha:
                raise RuntimeError("Prediction archive changed during verified reread")
            try: archived_records = json.loads(archived_bytes)
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise ValueError("Frozen prediction archive is malformed") from exc
            rebuilt_predictions = validate_prediction_archive(archived_records,
                evaluation_dataset.sample_ids, len(evaluation_dataset.class_names))
            metrics = evaluate_predictions(ground_truth, rebuilt_predictions,
                evaluation_dataset.class_names, 300)
            _atomic_json(output / "metrics.json", metrics)
            result = {"configuration": args.configuration, "training_config": vars(trainer.config),
                      "device": str(trainer.device), "evaluation_batch_size": args.eval_batch_size,
                      "train_dataset_identity": trainer.identity,
                      "evaluation_dataset_identity": evaluation_dataset.identity(),
                      "checkpoint_input": {"path": str(Path(args.checkpoint).resolve()),
                                           "sha256": args.checkpoint_sha256},
                      "role": args.role, "prediction_count": len(predictions),
                      "predictions_sha256": prediction_sha,
                      "metrics_sha256": file_sha256(output / "metrics.json")}
        elif args.mode == "verify-all":
            datasets = [_dataset(args, role) for role in ("train", "tune", "evaluation")]
            verification = verify_all_payloads(datasets, expected_objects=4189)
            _atomic_json(output / "verification.json", verification)
            result = {"configuration": args.configuration, "training_config": vars(_config(args)),
                      **verification, "verification_sha256": file_sha256(output / "verification.json")}
        else: raise ValueError("Unknown mode")
        if args.device == "cuda" and torch.cuda.is_available(): _synchronize(torch.device("cuda"))
        result["run_wall_time_seconds"] = time.perf_counter() - run_started
        run.update(status="passed", result=result, run_identity_sha256=canonical_sha(result))
        _atomic_json(output / "run.final.json", run)
        return run
    except BaseException as exc:
        if args.device == "cuda" and torch.cuda.is_available(): _synchronize(torch.device("cuda"))
        error = {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()}
        _atomic_json(output / "error.json", error)
        run.update(status="failed", run_wall_time_seconds=time.perf_counter() - run_started,
                   error={"type": error["type"], "message": error["message"]})
        _atomic_json(output / "run.final.json", run)
        raise


def parser():
    p = argparse.ArgumentParser(description="SPECTRA frozen-split formal FP32 lifecycle")
    sub = p.add_subparsers(dest="mode", required=True)
    for mode in ("dry-run", "verify-all", "probe", "train", "resume", "evaluate"):
        q = sub.add_parser(mode)
        q.add_argument("--manifest", required=True); q.add_argument("--raw-root", required=True)
        q.add_argument("--split-csv", required=True); q.add_argument("--split-sha256", required=True)
        q.add_argument("--output-dir", required=True); q.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
        q.add_argument("--configuration", choices=("B0", "B1", "F32", "R32", "R64"), default="R32")
        q.add_argument("--seed", type=int, choices=(17, 29, 43), default=17)
        if mode == "probe":
            q.add_argument("--probe-steps", type=_probe_count, default=1)
            q.add_argument("--probe-start-step", type=_probe_start, default=0)
            q.add_argument("--checkpoint-roundtrip", action="store_true")
            q.add_argument("--batch-size", type=_probe_count, default=1,
                           help="bounded probe-only batch size in [1,16]")
        if mode in ("resume", "evaluate"):
            q.add_argument("--checkpoint", required=True); q.add_argument("--checkpoint-sha256", required=True)
            q.add_argument("--source-run-manifest", required=True)
            q.add_argument("--source-run-manifest-sha256", required=True)
        if mode == "evaluate":
            q.add_argument("--role", choices=("tune", "evaluation"), default="evaluation")
            q.add_argument("--eval-batch-size", type=_probe_count, default=16,
                           help="execution-only evaluation batch; not checkpoint training identity")
    return p


def main(argv=None):
    try: execute(parser().parse_args(argv))
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr); return 1
    return 0


if __name__ == "__main__": raise SystemExit(main())
