"""Analytical augmentation/schedule and synthetic training-state fixtures."""
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import torch
from torch import nn

from tools.spectra_experiment.training import (TrainingConfig, SupervisedTrainer, augment_sample,
                                               file_sha256, learning_rate_at, stable_seed)
from tools.spectra_rewrite_training.model_adapter import state_sha256


class SyntheticDataset:
    """Five generated tensor samples, never a detector dataset or result."""
    split = "train"
    sample_ids = [f"synthetic_{i}" for i in range(5)]
    class_names = ("class0", "class1")

    def __len__(self):
        return len(self.sample_ids)

    def __getitem__(self, index):
        generator = torch.Generator().manual_seed(100 + index)
        image = torch.rand((3, 64, 64), generator=generator) * 0.1
        image[0, 8:40, 8:40] += 0.7
        return {"sample_id": self.sample_ids[index], "source_group": f"synthetic_group_{index}",
                "image": image, "boxes": torch.tensor([[8., 8., 40., 40.]]),
                "labels": torch.tensor([index % 2], dtype=torch.int64),
                "ignore_boxes": torch.empty((0, 4)), "valid_region": torch.tensor([0., 0., 64., 64.])}


class TinyDetector(nn.Module):
    """Training-loop test model, not a SPECTRA alternative or paper architecture."""
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(nn.Conv2d(3, 4, 3, stride=8, padding=1), nn.BatchNorm2d(4), nn.ReLU())
        self.cls = nn.Conv2d(4, 2, 1)
        self.reg = nn.Conv2d(4, 4, 1)

    def forward(self, images):
        feature = self.features(images)
        return [{"class_logits": self.cls(feature), "distances": self.reg(feature).square(), "stride": 8}]


def config(**kwargs):
    base = TrainingConfig(configuration="B0", num_classes=2, image_size=64, epochs=3,
                          batch_size=2, learning_rate=1e-4, warmup_epochs=1)
    return replace(base, **kwargs)


def trainer(**kwargs):
    torch.manual_seed(17)
    return SupervisedTrainer(SyntheticDataset(), config(**kwargs),
                             dataset_identity={"kind": "synthetic_test_only", "version": 1}, model=TinyDetector())


class TrainingTests(unittest.TestCase):
    def test_config_rejects_invalid_values(self):
        for field, value in (("batch_size", True), ("image_size", 32), ("epochs", 0), ("seed", -1),
                             ("warmup_epochs", 3), ("scale_min", .4), ("learning_rate", float("nan")),
                             ("horizontal_flip", 1.1), ("configuration", "YOLO")):
            with self.subTest(field=field), self.assertRaises(ValueError):
                config(**{field: value}).validate()

    def test_warmup_cosine_endpoints(self):
        cfg = config(learning_rate=.01)
        values = [learning_rate_at(cfg, i, 3) for i in range(9)]
        self.assertAlmostEqual(values[0], .01 / 3)
        self.assertEqual(values[2:4], [.01, .01])
        self.assertEqual(values[-1], 0)
        self.assertTrue(all(a >= b for a, b in zip(values[3:], values[4:])))
        with self.assertRaises(ValueError):
            learning_rate_at(cfg, 9, 3)

    def test_identity_augmentation_does_not_mutate(self):
        sample = SyntheticDataset()[0]
        original_image, original_boxes = sample["image"].clone(), sample["boxes"].clone()
        image, target, metadata = augment_sample(sample, config(scale_min=1., scale_max=1., horizontal_flip=0), 0)
        torch.testing.assert_close(image, original_image, rtol=0, atol=0)
        torch.testing.assert_close(target["boxes"], original_boxes, rtol=0, atol=0)
        torch.testing.assert_close(sample["image"], original_image, rtol=0, atol=0)
        torch.testing.assert_close(sample["boxes"], original_boxes, rtol=0, atol=0)
        self.assertEqual(metadata["gt_clipped_to_zero"], 0)

    def test_horizontal_flip_pixels_boxes_and_ignore(self):
        sample = SyntheticDataset()[0]
        sample["ignore_boxes"] = torch.tensor([[0., 2., 6., 10.]])
        image, target, metadata = augment_sample(sample, config(scale_min=1., scale_max=1., horizontal_flip=1), 0)
        torch.testing.assert_close(image, sample["image"].flip(-1), rtol=0, atol=0)
        torch.testing.assert_close(target["boxes"], torch.tensor([[24., 8., 56., 40.]]))
        torch.testing.assert_close(target["ignore_boxes"], torch.tensor([[58., 2., 64., 10.]]))
        self.assertTrue(metadata["horizontal_flip"])

    def test_shrink_padding_and_region(self):
        image, target, meta = augment_sample(SyntheticDataset()[0], config(scale_min=.5, scale_max=.5, horizontal_flip=0), 0)
        torch.testing.assert_close(target["valid_region"], torch.tensor([16., 16., 48., 48.]))
        torch.testing.assert_close(target["boxes"], torch.tensor([[20., 20., 36., 36.]]))
        self.assertEqual(meta["offset_x"], 16)
        self.assertTrue(torch.all(image[:, :16] == torch.tensor(114 / 255, dtype=torch.float32)))

    def test_center_crop_counts_fully_removed_gt(self):
        sample = SyntheticDataset()[0]
        sample["boxes"] = torch.tensor([[0., 0., 2., 2.], [8., 8., 40., 40.]])
        sample["labels"] = torch.tensor([0, 1])
        _, target, meta = augment_sample(sample, config(scale_min=2., scale_max=2., horizontal_flip=0), 0)
        torch.testing.assert_close(target["boxes"], torch.tensor([[0., 0., 48., 48.]]))
        self.assertEqual(target["labels"].tolist(), [1])
        self.assertEqual(meta["gt_clipped_to_zero"], 1)

    def test_sample_keyed_augmentation_is_order_independent(self):
        first = augment_sample(SyntheticDataset()[0], config(), 1)
        torch.manual_seed(998)
        augment_sample(SyntheticDataset()[1], config(), 99)
        second = augment_sample(SyntheticDataset()[0], config(), 1)
        torch.testing.assert_close(first[0], second[0], rtol=0, atol=0)
        self.assertEqual(first[2], second[2])
        self.assertNotEqual(stable_seed(17, 1, "a"), stable_seed(17, 2, "a"))

    def test_invalid_source_geometry_rejected(self):
        sample = SyntheticDataset()[0]
        sample["boxes"] = torch.tensor([[0., 0., -1., 2.]])
        with self.assertRaises(ValueError):
            augment_sample(sample, config(), 0)

    def test_training_split_and_source_identity_required(self):
        data = SyntheticDataset()
        data.split = "val"
        with self.assertRaises(ValueError):
            SupervisedTrainer(data, config(), dataset_identity={"a": 1}, model=TinyDetector())
        with self.assertRaises(ValueError):
            SupervisedTrainer(SyntheticDataset(), config(), dataset_identity={}, model=TinyDetector())

    def test_epoch_order_covers_each_sample_and_last_partial_batch(self):
        run = trainer()
        records = run.run_steps(3)
        ids = [s for record in records for s in record["sample_ids"]]
        self.assertEqual(sorted(ids), sorted(SyntheticDataset.sample_ids))
        self.assertEqual(len(records[-1]["sample_ids"]), 1)
        self.assertTrue(all(record["gradient_abs_sum"] > 0 for record in records))

    def test_exact_resume_across_epoch_boundary_and_schedule(self):
        with tempfile.TemporaryDirectory() as temporary:
            first = trainer()
            first.run_steps(2)
            path = Path(temporary) / "synthetic.pt"
            checksum = first.save_checkpoint(path)
            continued = first.run_steps(4)
            restored = trainer()
            restored.load_checkpoint(path, expected_sha256=checksum)
            repeated = restored.run_steps(4)
            self.assertEqual(continued, repeated)
            self.assertEqual(state_sha256(first.model), state_sha256(restored.model))
            self.assertEqual(first.global_step, restored.global_step)

    def test_checkpoint_rejects_overwrite_tamper_and_policy_change(self):
        with tempfile.TemporaryDirectory() as temporary:
            run = trainer()
            path = Path(temporary) / "synthetic.pt"
            checksum = run.save_checkpoint(path)
            with self.assertRaises(FileExistsError):
                run.save_checkpoint(path)
            with self.assertRaises(ValueError):
                run.load_checkpoint(path, expected_sha256="0" * 64)
            altered = trainer(horizontal_flip=0)
            with self.assertRaises(ValueError):
                altered.load_checkpoint(path, expected_sha256=checksum)
            self.assertEqual(file_sha256(path), checksum)

    def test_rejected_nonfinite_checkpoint_preserves_live_state(self):
        with tempfile.TemporaryDirectory() as temporary:
            run = trainer()
            path = Path(temporary) / "original.pt"
            run.save_checkpoint(path)
            payload = torch.load(path, weights_only=True)
            name = next(iter(payload["model"]))
            payload["model"][name].reshape(-1)[0] = float("nan")
            bad_path = Path(temporary) / "bad_fixture.pt"
            torch.save(payload, bad_path)
            before = state_sha256(run.model)
            with self.assertRaises(ValueError):
                run.load_checkpoint(bad_path, expected_sha256=file_sha256(bad_path))
            self.assertEqual(state_sha256(run.model), before)
            self.assertEqual(run.global_step, 0)

    def test_schedule_completion_has_no_extra_step(self):
        run = trainer(epochs=1, warmup_epochs=0)
        records = run.run_steps(3)
        self.assertEqual(records[-1]["learning_rate"], 0)
        with self.assertRaises(StopIteration):
            run.step()
        with self.assertRaises(ValueError):
            run.run_steps(1)

    def test_failed_step_cannot_retry_or_save_until_verified_restore(self):
        with tempfile.TemporaryDirectory() as temporary:
            run = trainer()
            checkpoint = Path(temporary) / "safe.pt"
            checksum = run.save_checkpoint(checkpoint)
            with torch.no_grad():
                next(run.model.parameters()).reshape(-1)[0] = float("nan")
            with self.assertRaises(ValueError):
                run.step()
            self.assertTrue(run.failed)
            with self.assertRaises(RuntimeError):
                run.step()
            with self.assertRaises(RuntimeError):
                run.save_checkpoint(Path(temporary) / "failed.pt")
            run.load_checkpoint(checkpoint, expected_sha256=checksum)
            self.assertFalse(run.failed)
            self.assertEqual(run.step()["global_step"], 0)

    def test_interrupt_after_forward_marks_bn_state_invalid(self):
        run = trainer()
        before = state_sha256(run.model)
        original_forward = run.model.forward

        def interrupted(images):
            original_forward(images)
            raise KeyboardInterrupt("synthetic interruption after BN update")

        run.model.forward = interrupted
        with self.assertRaises(KeyboardInterrupt):
            run.step()
        self.assertTrue(run.failed)
        self.assertEqual(run.global_step, 0)
        self.assertNotEqual(state_sha256(run.model), before)
        with self.assertRaises(RuntimeError):
            run.step()
        with tempfile.TemporaryDirectory() as temporary, self.assertRaises(RuntimeError):
            run.save_checkpoint(Path(temporary) / "interrupted.pt")

    def test_checkpoint_rejects_sgd_flag_and_momentum_shape_change(self):
        with tempfile.TemporaryDirectory() as temporary:
            run = trainer()
            run.step()
            path = Path(temporary) / "safe.pt"
            run.save_checkpoint(path)
            before = state_sha256(run.model)
            for kind in ("maximize", "momentum_shape"):
                payload = torch.load(path, weights_only=True)
                if kind == "maximize":
                    payload["optimizer"]["param_groups"][0]["maximize"] = True
                else:
                    index = next(iter(payload["optimizer"]["state"]))
                    payload["optimizer"]["state"][index]["momentum_buffer"] = torch.zeros(1)
                bad = Path(temporary) / f"{kind}.pt"
                torch.save(payload, bad)
                with self.assertRaises(ValueError):
                    run.load_checkpoint(bad, expected_sha256=file_sha256(bad))
                self.assertEqual(state_sha256(run.model), before)

    def test_partial_save_never_publishes_destination_and_cleans_own_temporary(self):
        with tempfile.TemporaryDirectory() as temporary:
            run = trainer()
            path = Path(temporary) / "safe.pt"

            def failed_save(payload, stream):
                stream.write(b"partial")
                raise OSError("synthetic I/O failure")

            with mock.patch("tools.spectra_experiment.training.torch.save", side_effect=failed_save):
                with self.assertRaises(OSError):
                    run.save_checkpoint(path)
            self.assertFalse(path.exists())
            self.assertEqual(list(Path(temporary).iterdir()), [])
            checksum = run.save_checkpoint(path)
            self.assertEqual(file_sha256(path), checksum)

    def test_atomic_save_never_replaces_a_concurrent_destination(self):
        with tempfile.TemporaryDirectory() as temporary:
            run = trainer()
            path = Path(temporary) / "raced.pt"

            def racing_link(source, destination):
                with Path(destination).open("xb") as stream:
                    stream.write(b"other writer")
                raise FileExistsError("synthetic competing writer")

            with mock.patch("tools.spectra_experiment.training.os.link", side_effect=racing_link):
                with self.assertRaises(FileExistsError):
                    run.save_checkpoint(path)
            self.assertEqual(path.read_bytes(), b"other writer")
            self.assertEqual(list(Path(temporary).iterdir()), [path])


if __name__ == "__main__":
    unittest.main()
