"""Synthetic unit fixtures only; no scientific model result is produced."""
from __future__ import annotations

import csv
import copy
import hashlib
import io
import json
import tempfile
import unittest
from unittest import mock
from pathlib import Path
from types import SimpleNamespace

import torch
from PIL import Image
from torch import nn

from tools.spectra_experiment.dataset import DatasetIntegrityError
from tools.spectra_experiment.training import TrainingConfig
from tools.spectra_formal_experiment.cli import (_config, _create_output, _formal_split_contract,
                                                 _measure_epoch, _measure_probe,
                                                 _probe_checkpoint_roundtrip, _probe_window,
                                                 _resume_source, _run_probe, parser)
from tools.spectra_formal_experiment.dataset import (FROZEN_SPLIT_SHA256, FrozenSplitContract,
                                                     FrozenSplitDataset)
from tools.spectra_formal_experiment.runner import (FormalFPTrainer, _move_targets, canonical_sha,
    critical_code_hashes, file_sha256, run_evaluation, validate_prediction_archive, verify_all_payloads)
from tools.spectra_rewrite_training.model_adapter import state_sha256

torch.set_num_threads(2)
RUN_IDENTITY, RUN_MANIFEST_SHA = "1" * 64, "2" * 64


def sha(data): return hashlib.sha256(data).hexdigest()


class Fixture:
    def __init__(self, root: Path):
        self.root = root; (root / "images").mkdir(); (root / "annotations").mkdir()
        samples = []
        for index in range(4):
            sid = f"sample_{index}"
            image = Image.new("RGB", (64, 48), (10 + index * 30, 40, 80))
            image_path, xml_path = root / "images" / f"{sid}.jpg", root / "annotations" / f"{sid}.xml"
            image.save(image_path, quality=95)
            with Image.open(image_path) as decoded: rgb = decoded.convert("RGB"); pixels = rgb.tobytes()
            xml = (f"<annotation><filename>{sid}.jpg</filename><size><width>64</width><height>48</height></size>"
                   f"<object><name>crazing</name><difficult>{int(index == 0)}</difficult><truncated>0</truncated>"
                   "<bndbox><xmin>8</xmin><ymin>8</ymin><xmax>24</xmax><ymax>28</ymax></bndbox>"
                   "</object></annotation>").encode()
            xml_path.write_bytes(xml); ib = image_path.read_bytes()
            samples.append({"sample_id": sid, "source_image": str(image_path), "source_image_sha256": sha(ib),
                "source_annotation": str(xml_path), "source_annotation_sha256": sha(xml),
                "image": {"path": str(image_path), "sha256": sha(ib), "bytes": len(ib), "width": 64,
                          "height": 48, "decoded_rgb_sha256": sha(b"RGB:64x48:" + pixels)},
                "annotation": {"path": str(xml_path), "sha256": sha(xml), "bytes": len(xml)},
                "decoded_size": [64, 48]})
        manifest = {"provenance": {"source_root": str(root), "class_names_from_existing_yaml":
                    ["crazing", "inclusion", "patches", "pitted_surface", "rolled-in_scale", "scratches"]},
                    "samples": samples}
        self.manifest = root / "manifest.json"; self.manifest.write_text(json.dumps(manifest), encoding="utf-8")
        self.rows = [("sample_0", samples[0]["image"]["decoded_rgb_sha256"], "train"),
                     ("sample_1", samples[1]["image"]["decoded_rgb_sha256"], "train"),
                     ("sample_2", samples[2]["image"]["decoded_rgb_sha256"], "tune"),
                     ("sample_3", samples[3]["image"]["decoded_rgb_sha256"], "evaluation")]
        self.split = root / "split.csv"
        self.write_rows(self.rows)

    def write_rows(self, rows):
        with self.split.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream, lineterminator="\n"); writer.writerow(["sample_id", "source_group", "split_role"])
            writer.writerows(rows)

    def contract(self):
        return FrozenSplitContract("fixture_v1", file_sha256(self.manifest), file_sha256(self.split), (2, 1, 1))

    def dataset(self, role):
        return FrozenSplitDataset(self.manifest, self.root, self.split, role, contract=self.contract(), image_size=64)


class EmptyModel(nn.Module):
    def __init__(self): super().__init__(); self.anchor = nn.Parameter(torch.zeros(()))
    def forward(self, images):
        n = len(images); logits = torch.full((n, 6, 8, 8), -20., device=images.device) + self.anchor * 0
        return [{"class_logits": logits, "distances": torch.zeros((n, 4, 8, 8), device=images.device), "stride": 8}]


class ProbeStub:
    def __init__(self):
        self.global_step, self.total_steps, self.device, self.checkpoint_calls = 0, 32, torch.device("cpu"), 0
    def step(self):
        record = {"global_step": self.global_step, "sample_ids": [f"s{self.global_step}"]}
        self.global_step += 1
        return record
    def save_checkpoint(self, _path):
        self.checkpoint_calls += 1
        return {"model_state_sha256": "0" * 64}
    def run_to_epoch_boundary(self):
        return [self.step(), self.step()]


class FormalTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.fx = Fixture(Path(self.tmp.name))
    def tearDown(self): self.tmp.cleanup()

    def trainer(self, config, **kwargs):
        return FormalFPTrainer(self.fx.dataset("train"), config,
            source_run_identity=RUN_IDENTITY, source_run_manifest_sha256=RUN_MANIFEST_SHA, **kwargs)

    def test_roles_are_complete_read_only_and_difficult_included(self):
        train, tune, evaluation = (self.fx.dataset(x) for x in ("train", "tune", "evaluation"))
        self.assertEqual((len(train), len(tune), len(evaluation)), (2, 1, 1))
        self.assertEqual(train.identity()["split_csv_sha256"], file_sha256(self.fx.split))
        sample = train[0]
        self.assertEqual(sample["labels"].tolist(), [0]); self.assertEqual(len(sample["ignore_boxes"]), 0)
        self.assertTrue(sample["provenance"]["payload_bytes_and_sha256_reverified"])

    def test_split_tampering_and_group_cross_role_rejected(self):
        contract = self.fx.contract()
        self.fx.split.write_text(self.fx.split.read_text() + "\n", encoding="utf-8")
        with self.assertRaisesRegex(DatasetIntegrityError, "SHA256"): FrozenSplitDataset(
            self.fx.manifest, self.fx.root, self.fx.split, "train", contract=contract, image_size=64)
        rows = list(self.fx.rows); rows[1] = (rows[1][0], rows[0][1], "tune"); self.fx.write_rows(rows)
        with self.assertRaisesRegex(DatasetIntegrityError, "source_group"):
            FrozenSplitDataset(self.fx.manifest, self.fx.root, self.fx.split, "train",
                               contract=self.fx.contract(), image_size=64)

    def test_cpu_devices_and_one_step(self):
        config = TrainingConfig(configuration="R32", image_size=64, batch_size=2, epochs=4,
                                warmup_epochs=1, horizontal_flip=0., scale_min=1., scale_max=1.)
        trainer = self.trainer(config, device="cpu")
        record = trainer.step()
        self.assertEqual(record["global_step"], 0); self.assertTrue(record["loss"] >= 0)
        moved = _move_targets([{"x": torch.ones(1), "image_hw": (64, 64)}], torch.device("cpu"))
        self.assertEqual(moved[0]["x"].device.type, "cpu")

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_real_cuda_trainer_construct_step_checkpoint_and_device_index(self):
        expected = torch.device("cuda", torch.cuda.current_device())
        target = _move_targets([{"boxes": torch.ones(1, 4), "labels": torch.zeros(1, dtype=torch.int64)}],
                               expected)[0]
        self.assertEqual({v.device for v in target.values()}, {expected})
        config = TrainingConfig(configuration="B1", image_size=64, batch_size=1, epochs=4,
                                warmup_epochs=1, horizontal_flip=0., scale_min=1., scale_max=1.)
        trainer = self.trainer(config, device="cuda")
        self.assertEqual(trainer.device, expected)
        self.assertEqual({parameter.device for parameter in trainer.model.parameters()}, {expected})
        images, targets, _ = trainer._batch()
        self.assertEqual(images.device, expected)
        self.assertEqual({value.device for item in targets for value in item.values()
                          if isinstance(value, torch.Tensor)}, {expected})
        del images, targets
        record = trainer.step(); self.assertEqual(record["global_step"], 0)
        path = Path(self.tmp.name) / "cuda.pt"; info = trainer.save_checkpoint(path)
        payload = torch.load(path, map_location="cpu", weights_only=True)
        self.assertEqual(payload["device"], str(expected))
        restored = self.trainer(config, device="cuda")
        restored.load_checkpoint(path, info["sha256"], expected_source_run_identity=RUN_IDENTITY,
                                 expected_source_run_manifest_sha256=RUN_MANIFEST_SHA)
        optimizer_devices = {value.device for state in restored.optimizer.state.values()
                             for value in state.values() if isinstance(value, torch.Tensor)}
        self.assertEqual(optimizer_devices, {expected})

    def test_checkpoint_exact_resume_and_file_tamper_rejection(self):
        config = TrainingConfig(configuration="R32", image_size=64, batch_size=2, epochs=4,
                                warmup_epochs=1, horizontal_flip=0., scale_min=1., scale_max=1.)
        first = self.trainer(config); first.step()
        path = Path(self.tmp.name) / "one.pt"; info = first.save_checkpoint(path)
        payload = torch.load(path, map_location="cpu", weights_only=True)
        self.assertEqual(payload["initialization"]["configuration"], "R32")
        restored = self.trainer(config)
        restored.load_checkpoint(path, info["sha256"], expected_source_run_identity=RUN_IDENTITY,
                                 expected_source_run_manifest_sha256=RUN_MANIFEST_SHA)
        self.assertEqual(restored.global_step, 1)
        tampered = Path(self.tmp.name) / "tampered.pt"; tampered.write_bytes(path.read_bytes() + b"x")
        with self.assertRaisesRegex(ValueError, "file SHA256"):
            restored.load_checkpoint(tampered, info["sha256"], expected_source_run_identity=RUN_IDENTITY,
                                     expected_source_run_manifest_sha256=RUN_MANIFEST_SHA)
        with self.assertRaises(FileExistsError): first.save_checkpoint(path)
        first.failed = True
        with self.assertRaisesRegex(RuntimeError, "Failed training state"):
            first.save_checkpoint(Path(self.tmp.name) / "failed.pt")

        wrong_architecture = self.trainer(TrainingConfig(configuration="B0", image_size=64,
            batch_size=2, epochs=4, warmup_epochs=1, horizontal_flip=0., scale_min=1., scale_max=1.))
        with self.assertRaisesRegex(ValueError, "config mismatch"):
            wrong_architecture.load_checkpoint(path, info["sha256"],
                expected_source_run_identity=RUN_IDENTITY,
                expected_source_run_manifest_sha256=RUN_MANIFEST_SHA)
        with self.assertRaisesRegex(ValueError, "source-run"):
            restored.load_checkpoint(path, info["sha256"], expected_source_run_identity="3" * 64,
                                     expected_source_run_manifest_sha256=RUN_MANIFEST_SHA)
        self.assertEqual(payload["critical_code_sha256"], critical_code_hashes())
        bad_code = copy.deepcopy(payload); bad_code["critical_code_sha256"]["x.py"] = "0" * 64
        bad_code_path = Path(self.tmp.name) / "bad_code.pt"; torch.save(bad_code, bad_code_path)
        with self.assertRaisesRegex(ValueError, "critical-code"):
            restored.load_checkpoint(bad_code_path, file_sha256(bad_code_path),
                expected_source_run_identity=RUN_IDENTITY,
                expected_source_run_manifest_sha256=RUN_MANIFEST_SHA)

    def test_optimizer_checkpoint_schema_and_finite_values_are_rejected_before_commit(self):
        config = TrainingConfig(configuration="R32", image_size=64, batch_size=2, epochs=4,
                                warmup_epochs=1, horizontal_flip=0., scale_min=1., scale_max=1.)
        source = self.trainer(config); source.step()
        good = Path(self.tmp.name) / "good.pt"; source.save_checkpoint(good)
        original = torch.load(good, map_location="cpu", weights_only=True)
        variants = []
        flags = copy.deepcopy(original); flags["optimizer_state"]["param_groups"][0]["weight_decay"] = 0.25
        variants.append(flags)
        nonfinite = copy.deepcopy(original)
        first_state = next(iter(nonfinite["optimizer_state"]["state"].values()))
        first_state["momentum_buffer"].reshape(-1)[0] = float("inf"); variants.append(nonfinite)
        wrong_shape = copy.deepcopy(original)
        next(iter(wrong_shape["optimizer_state"]["state"].values()))["momentum_buffer"] = torch.zeros(1)
        variants.append(wrong_shape)
        wrong_mapping = copy.deepcopy(original)
        state = wrong_mapping["optimizer_state"]["state"]
        state[len(list(source.model.parameters())) + 1] = state.pop(next(iter(state))); variants.append(wrong_mapping)
        for index, payload in enumerate(variants):
            path = Path(self.tmp.name) / f"bad_{index}.pt"; torch.save(payload, path)
            restored = self.trainer(config)
            before = state_sha256(restored.model)
            with self.assertRaises(ValueError): restored.load_checkpoint(path, file_sha256(path),
                expected_source_run_identity=RUN_IDENTITY,
                expected_source_run_manifest_sha256=RUN_MANIFEST_SHA)
            self.assertEqual(state_sha256(restored.model), before)

    def test_empty_predictions_retain_complete_ids(self):
        predictions, metrics = run_evaluation(EmptyModel(), self.fx.dataset("evaluation"), batch_size=1)
        self.assertEqual([x["sample_id"] for x in predictions], ["sample_3"])
        self.assertEqual(predictions[0]["boxes"], [])
        self.assertFalse(predictions[0]["source_config"]["multi_label"])
        self.assertEqual(metrics["ap50_95"], 0.0)
        rebuilt = validate_prediction_archive(json.loads(json.dumps(predictions)), ["sample_3"], 6)
        self.assertEqual(tuple(rebuilt[0]["boxes"].shape), (0, 4))
        bad = copy.deepcopy(predictions); bad[0]["scores"] = [float("nan")]
        with self.assertRaises(ValueError): validate_prediction_archive(bad, ["sample_3"], 6)
        bad = copy.deepcopy(predictions); bad[0]["unexpected"] = True
        with self.assertRaises(ValueError): validate_prediction_archive(bad, ["sample_3"], 6)

    def test_verify_all_traverses_every_fixture_payload_without_optimizer(self):
        result = verify_all_payloads([self.fx.dataset(role) for role in ("train", "tune", "evaluation")],
                                     expected_objects=4)
        self.assertEqual((result["total_images"], result["total_objects"], result["optimizer_steps"]), (4, 4, 0))
        self.assertEqual(result["roles"]["train"]["difficult_objects"], 1)
        with self.assertRaisesRegex(ValueError, "total image/object"):
            verify_all_payloads([self.fx.dataset(role) for role in ("train", "tune", "evaluation")],
                                expected_objects=5)

    def test_cli_modes_and_output_refusal(self):
        for mode in ("dry-run", "verify-all", "probe", "train", "resume", "evaluate"):
            self.assertIn(mode, parser()._subparsers._group_actions[0].choices)
        path = Path(self.tmp.name) / "output"; _create_output(path)
        with self.assertRaises(FileExistsError): _create_output(path)

    def test_all_configurations_and_bounded_probe_argument(self):
        base = ["--manifest", "m", "--raw-root", "r", "--split-csv", "s",
                "--split-sha256", FROZEN_SPLIT_SHA256, "--output-dir", "o"]
        for configuration in ("B0", "B1", "F32", "R32", "R64"):
            args = parser().parse_args(["probe", *base, "--configuration", configuration,
                                        "--probe-steps", "3", "--checkpoint-roundtrip"])
            self.assertEqual((args.configuration, args.probe_steps), (configuration, 3))
            self.assertTrue(args.checkpoint_roundtrip)
        records = _run_probe(ProbeStub(), 3)
        self.assertEqual(([x["sample_ids"] for x in records], len(records)),
                         ([["s0"], ["s1"], ["s2"]], 3))
        for invalid in (0, 17):
            with self.assertRaises(ValueError): _run_probe(ProbeStub(), invalid)
        with self.assertRaises(SystemExit), mock.patch("sys.stderr", new=io.StringIO()):
            parser().parse_args(["probe", *base, "--probe-steps", "17"])
        self.assertEqual(_formal_split_contract(FROZEN_SPLIT_SHA256).split_sha256, FROZEN_SPLIT_SHA256)
        with self.assertRaisesRegex(ValueError, "code-pinned"):
            _formal_split_contract("0" * 64)
        train_args = parser().parse_args(["train", *base, "--configuration", "B1"])
        self.assertEqual((_config(train_args).image_size, _config(train_args).batch_size), (640, 16))
        evaluation_args = parser().parse_args(["evaluate", *base, "--checkpoint", "c",
            "--checkpoint-sha256", "0" * 64, "--source-run-manifest", "run.json",
            "--source-run-manifest-sha256", "1" * 64, "--eval-batch-size", "3"])
        self.assertEqual((_config(evaluation_args).batch_size, evaluation_args.eval_batch_size), (16, 3))
        with self.assertRaises(SystemExit), mock.patch("sys.stderr", new=io.StringIO()):
            parser().parse_args(["train", *base, "--batch-size", "2"])
        self.assertIn("tools/spectra_rewrite_training/data.py", critical_code_hashes())
        last = _probe_window(1260, 16, 78, 1)
        self.assertEqual((last["steps_per_epoch"], last["batch_indices"], last["sample_counts"]),
                         (79, [78], [12]))
        with self.assertRaisesRegex(ValueError, "may not cross"):
            _probe_window(1260, 16, 78, 2)

    def test_measured_probe_and_epoch_fields_are_real_nonnegative_cpu_values(self):
        probe = ProbeStub()
        records, measurement = _measure_probe(probe, 3)
        self.assertEqual((len(records), probe.checkpoint_calls), (3, 0))
        self.assertIn("NOT_FOR_RESULTS", measurement["measurement_scope"])
        for key in ("wall_time_seconds", "seconds_per_step"):
            self.assertIs(type(measurement[key]), float); self.assertGreaterEqual(measurement[key], 0.)
        for key in ("cuda_current_allocated_bytes", "cuda_peak_allocated_bytes",
                    "cuda_current_reserved_bytes", "cuda_peak_reserved_bytes"):
            self.assertIsNone(measurement[key])
        epoch = ProbeStub()
        epoch_records, checkpoint, timing = _measure_epoch(epoch, Path(self.tmp.name) / "unused.pt")
        self.assertEqual((len(epoch_records), epoch.checkpoint_calls, checkpoint["model_state_sha256"]),
                         (2, 1, "0" * 64))
        for key in ("optimizer_wall_time_seconds", "checkpoint_wall_time_seconds",
                    "epoch_total_wall_time_seconds"):
            self.assertIs(type(timing[key]), float); self.assertGreaterEqual(timing[key], 0.)
        self.assertGreaterEqual(timing["epoch_total_wall_time_seconds"],
                                timing["optimizer_wall_time_seconds"])

    def test_probe_checkpoint_roundtrip_is_explicit_and_not_formal(self):
        config = TrainingConfig(configuration="B1", image_size=64, batch_size=2, epochs=4,
                                warmup_epochs=1, horizontal_flip=0., scale_min=1., scale_max=1.)
        trainer = self.trainer(config); trainer.step()
        path = Path(self.tmp.name) / "NOT_FOR_RESULTS_probe_roundtrip.pt"
        result = _probe_checkpoint_roundtrip(trainer, trainer.dataset, config, path)
        self.assertTrue(result["roundtrip_passed"]); self.assertTrue(path.is_file())
        self.assertIn("NOT_FOR_RESULTS", result["scope"])
        with self.assertRaisesRegex(ValueError, "fixed NOT_FOR_RESULTS"):
            _probe_checkpoint_roundtrip(trainer, trainer.dataset, config, Path(self.tmp.name) / "wrong.pt")

    def test_explicit_source_run_manifest_identity(self):
        initial = {"format": "spectra_formal_run_v1", "mode": "train", "status": "running", "configuration": "R32"}
        initial_path = Path(self.tmp.name) / "run.json"
        initial_path.write_text(json.dumps(initial), encoding="utf-8")
        result = {"global_step": 10}
        final = {**initial, "status": "passed", "result": result, "run_identity_sha256": canonical_sha(result)}
        path = Path(self.tmp.name) / "run.final.json"; path.write_text(json.dumps(final), encoding="utf-8")
        args = SimpleNamespace(source_run_manifest=str(path), source_run_manifest_sha256=file_sha256(path))
        identity, digest = _resume_source(args)
        self.assertEqual((identity, digest), (canonical_sha(initial), file_sha256(initial_path)))
        args.source_run_manifest_sha256 = "0" * 64
        with self.assertRaisesRegex(ValueError, "manifest SHA256"):
            _resume_source(args)
        probe_final = {**final, "mode": "probe"}; probe_final["run_identity_sha256"] = canonical_sha(result)
        path.write_text(json.dumps(probe_final), encoding="utf-8")
        args.source_run_manifest_sha256 = file_sha256(path)
        with self.assertRaisesRegex(ValueError, "passed train/resume"):
            _resume_source(args)
        failed_final = {**final, "status": "failed"}
        path.write_text(json.dumps(failed_final), encoding="utf-8")
        args.source_run_manifest_sha256 = file_sha256(path)
        with self.assertRaisesRegex(ValueError, "passed train/resume"):
            _resume_source(args)


if __name__ == "__main__": unittest.main()
