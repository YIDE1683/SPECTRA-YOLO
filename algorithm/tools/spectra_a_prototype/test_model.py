"""Synthetic unit fixtures only: no trained-detector or performance evidence."""

import itertools
import json
import math
from pathlib import Path
import unittest

import torch
from torch import nn
from torch.nn import functional as F

from model import ConvBN, RepBlock, SpectraA, decode


torch.set_num_threads(2)


def nontrivial_bn(module):
    """Set all four affine/statistical terms away from initialization defaults."""
    with torch.no_grad():
        for layer in module.modules():
            if isinstance(layer, nn.BatchNorm2d):
                width = layer.num_features
                layer.running_mean.copy_(torch.linspace(-0.2, 0.3, width))
                layer.running_var.copy_(torch.linspace(0.6, 1.4, width))
                layer.weight.copy_(torch.linspace(-0.9, 1.1, width))
                layer.bias.copy_(torch.linspace(-0.1, 0.2, width))


class ModelTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)

    def test_contract_and_all_24_shapes(self):
        contract = json.loads(Path(__file__).with_name("contract.json").read_text(encoding="utf-8"))
        self.assertEqual(contract["backbone"]["base_reps"], [0, 1, 2, 2, 1])
        self.assertEqual(contract["backbone"]["reduced_reps"], [0, 1, 1, 1, 1])
        count = 0
        for p2, neck, reduced, classes in itertools.product((0, 32, 64), (64, 96), (False, True), (6, 10)):
            with self.subTest(p2=p2, neck=neck, reduced=reduced, classes=classes):
                model = SpectraA(p2, neck, reduced, classes).eval()
                with torch.no_grad():
                    outputs = model(torch.rand(1, 3, 64, 64))
                expected = [4, 8, 16, 32] if p2 else [8, 16, 32]
                self.assertEqual([output["stride"] for output in outputs], expected)
                for output, stride in zip(outputs, expected):
                    self.assertEqual(output["class_logits"].shape, (1, classes, 64 // stride, 64 // stride))
                    self.assertEqual(output["distances"].shape, (1, 4, 64 // stride, 64 // stride))
                    self.assertTrue(torch.isfinite(output["class_logits"]).all())
                    self.assertTrue((output["distances"] >= 0).all())
                config = model.architecture_config()
                self.assertEqual(config["rep_blocks"], contract["backbone"]["reduced_reps" if reduced else "base_reps"])
                self.assertEqual(config["p2_channels"], p2)
                self.assertEqual(config["neck_channels"], neck)
                self.assertEqual(config["num_classes"], classes)
                self.assertEqual(config, model.architecture_config())
                config["stage_channels"][0] = 999
                self.assertEqual(model.architecture_config()["stage_channels"][0], 32)
                count += 1
        self.assertEqual(count, 24)

    def test_representative_640_and_rectangular_input(self):
        model = SpectraA(p2_channels=32, neck_channels=96, reduced_depth=True, num_classes=10).eval()
        with torch.no_grad():
            outputs = model(torch.rand(1, 3, 640, 640))
            decoded = model.decode(outputs, (640, 640))
            rectangular = model(torch.rand(2, 3, 64, 96))
        self.assertEqual([o["class_logits"].shape[-2:] for o in outputs],
                         [torch.Size((160, 160)), torch.Size((80, 80)), torch.Size((40, 40)), torch.Size((20, 20))])
        self.assertEqual(decoded["boxes"].shape, (1, 34000, 4))
        self.assertEqual(decoded["scores"].shape, (1, 34000, 10))
        self.assertEqual(decoded["valid"].shape, (1, 34000))
        self.assertEqual(rectangular[-1]["class_logits"].shape, (2, 10, 2, 3))

    def test_exact_topology_and_output_biases(self):
        model = SpectraA(64, 96, False, 6)
        self.assertEqual(sum(isinstance(module, RepBlock) for module in model.modules()), 6)
        self.assertEqual(len(model.laterals), 3)
        self.assertEqual(model.p2_top.conv.in_channels, 96)
        self.assertEqual(model.p2_top.conv.out_channels, 64)
        self.assertIsInstance(model.p2_top.activation, nn.Identity)
        self.assertTrue(all(isinstance(lateral.activation, nn.Identity) for lateral in model.laterals))
        self.assertEqual([len(head.class_tower) for head in model.heads], [1, 2, 2, 2])
        self.assertEqual([head.class_output.in_channels for head in model.heads], [32, 64, 64, 64])
        for head in model.heads:
            self.assertFalse(head.class_tower[0] is head.regression_tower[0])
            torch.testing.assert_close(head.class_output.bias, torch.full_like(head.class_output.bias, math.log(0.01 / 0.99)))
            torch.testing.assert_close(head.regression_output.bias, torch.ones_like(head.regression_output.bias))

    def test_last_use_reference_cleanup_preserves_graph_outputs(self):
        def retained_reference(model, x):
            features = []
            for stage in model.stages:
                x = stage(x)
                features.append(x)
            l3, l4, l5 = [layer(feature) for layer, feature in zip(model.laterals, features[2:])]
            p5 = F.relu(l5)
            p4 = F.relu(l4 + F.interpolate(p5, scale_factor=2, mode="nearest"))
            p3 = F.relu(l3 + F.interpolate(p4, scale_factor=2, mode="nearest"))
            pyramid = [p3, p4, p5]
            if model.p2_channels:
                p2 = F.relu(model.p2_lateral(features[1]) + F.interpolate(
                    model.p2_top(p3), scale_factor=2, mode="nearest"))
                pyramid.insert(0, p2)
            return [head(feature, stride) for head, feature, stride in zip(model.heads, pyramid, model.strides)]

        for p2, neck in itertools.product((0, 32, 64), (64, 96)):
            with self.subTest(p2=p2, neck=neck):
                model = SpectraA(p2, neck, True, 6).eval()
                x = torch.rand(1, 3, 64, 96)
                with torch.no_grad():
                    expected = retained_reference(model, x)
                    actual = model(x)
                for before, after in zip(expected, actual):
                    self.assertEqual(before["stride"], after["stride"])
                    for key in ("class_logits", "distances"):
                        torch.testing.assert_close(before[key], after[key], rtol=0, atol=0)

    def test_conv_bn_fusion_with_bias_nontrivial_statistics_and_border(self):
        module = ConvBN(3, 5, 3, stride=2, activate=True).double().eval()
        # Exercise the fold primitive's nonzero convolution-bias path too.
        module.conv.bias = nn.Parameter(torch.linspace(-0.3, 0.4, 5, dtype=torch.float64))
        nontrivial_bn(module)
        x = torch.zeros(2, 3, 10, 12, dtype=torch.float64)
        x[:, :, 0, :] = 2
        x[:, :, -1, :] = -3
        x[:, :, :, 0] = 4
        x[:, :, :, -1] = -1
        before = module(x)
        module.fold()
        torch.testing.assert_close(before, module(x), rtol=1e-12, atol=1e-12)
        self.assertIsInstance(module.bn, nn.Identity)
        module.fold()
        torch.testing.assert_close(before, module(x), rtol=1e-12, atol=1e-12)

    def test_rep_fusion_nontrivial_statistics_and_borders(self):
        module = RepBlock(7).double().eval()
        nontrivial_bn(module)
        x = torch.randn(2, 7, 9, 11, dtype=torch.float64)
        x[:, :, 0, :] = 3
        x[:, :, -1, :] = -2
        x[:, :, :, 0] = 4
        x[:, :, :, -1] = -5
        before = module(x)
        module.fold()
        torch.testing.assert_close(before, module(x), rtol=1e-11, atol=1e-11)
        self.assertFalse(any(isinstance(layer, nn.BatchNorm2d) for layer in module.modules()))
        self.assertEqual(module.reparam.kernel_size, (3, 3))

    def test_whole_graph_fusion_copy_and_idempotence(self):
        for p2 in (0, 32, 64):
            with self.subTest(p2=p2):
                model = SpectraA(p2, 96, True, 10).eval()
                nontrivial_bn(model)
                source_state = {name: tensor.clone() for name, tensor in model.state_dict().items()}
                x = torch.rand(1, 3, 64, 96)
                x[:, :, 0, :] = 0
                x[:, :, -1, :] = 1
                with torch.no_grad():
                    before = model(x)
                    deployed = model.to_deploy()
                    after = deployed(x)
                    twice = deployed.to_deploy()(x)
                self.assertFalse(any(layer.training for layer in deployed.modules()))
                self.assertFalse(any(isinstance(layer, nn.BatchNorm2d) for layer in deployed.modules()))
                self.assertTrue(any(isinstance(layer, nn.BatchNorm2d) for layer in model.modules()))
                self.assertEqual(model.architecture_config(), deployed.architecture_config())
                for expected, actual, again in zip(before, after, twice):
                    for key in ("class_logits", "distances"):
                        torch.testing.assert_close(actual[key], expected[key], rtol=1e-4, atol=2e-5)
                        torch.testing.assert_close(again[key], actual[key], rtol=0, atol=0)
                for name, tensor in model.state_dict().items():
                    torch.testing.assert_close(tensor, source_state[name], rtol=0, atol=0)
                self.assertNotEqual(next(model.parameters()).data_ptr(), next(deployed.parameters()).data_ptr())

    def test_fusion_refuses_training_or_mixed_mode(self):
        model = SpectraA()
        with self.assertRaisesRegex(ValueError, "eval"):
            model.to_deploy()
        model.eval()
        model.stages[0].downsample.bn.train()
        with self.assertRaisesRegex(ValueError, "eval"):
            model.to_deploy()
        with self.assertRaisesRegex(ValueError, "eval"):
            RepBlock(3).fold()
        with self.assertRaisesRegex(ValueError, "eval"):
            ConvBN(3, 3, 3).fold()

    def test_finite_nonzero_gradients(self):
        model = SpectraA(32, 64, True, 6).train()
        x = torch.rand(2, 3, 64, 64, requires_grad=True)
        outputs = model(x)
        loss = sum(output["class_logits"].square().mean() + output["distances"].square().mean()
                   for output in outputs)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.isfinite(x.grad).all())
        self.assertGreater(x.grad.abs().sum().item(), 0)
        for name, parameter in model.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(torch.isfinite(parameter.grad).all(), name)
        self.assertGreater(model.stages[0].downsample.conv.weight.grad.abs().sum().item(), 0)

    def test_invalid_configuration_and_input(self):
        for kwargs in ({"p2_channels": 16}, {"p2_channels": False}, {"neck_channels": 32},
                       {"neck_channels": 64.0}, {"reduced_depth": 1}, {"num_classes": 7}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                SpectraA(**kwargs)
        model = SpectraA().eval()
        for x in (None, torch.rand(3, 64, 64), torch.rand(1, 1, 64, 64),
                  torch.rand(0, 3, 64, 64), torch.rand(1, 3, 63, 64),
                  torch.rand(1, 3, 64, 65), torch.rand(1, 3, 0, 64),
                  torch.zeros(1, 3, 64, 64, dtype=torch.int64)):
            with self.subTest(shape=getattr(x, "shape", None)), self.assertRaises(ValueError):
                model(x)


class DecodeTests(unittest.TestCase):
    @staticmethod
    def fixture():
        return [{"class_logits": torch.zeros(1, 6, 1, 2),
                 "distances": torch.tensor([[[[1.0, 0.0]], [[1.0, 0.0]],
                                               [[1.0, 0.0]], [[1.0, 0.0]]]]),
                 "stride": 8}]

    def test_grid_stride_clipping_zero_invalid_and_sigmoid(self):
        result = decode(self.fixture(), (8, 16))
        torch.testing.assert_close(result["boxes"], torch.tensor([[[0., 0., 12., 8.], [12., 4., 12., 4.]]]))
        torch.testing.assert_close(result["scores"], torch.full((1, 2, 6), 0.5))
        self.assertEqual(result["valid"].dtype, torch.bool)
        self.assertEqual(result["valid"].tolist(), [[True, False]])

    def test_row_major_multilevel_and_gradient(self):
        logits = torch.arange(24, dtype=torch.float64).reshape(1, 6, 2, 2).requires_grad_()
        distances = torch.full((1, 4, 2, 2), 0.25, dtype=torch.float64, requires_grad=True)
        second = {"class_logits": torch.zeros(1, 6, 1, 1, dtype=torch.float64),
                  "distances": torch.ones(1, 4, 1, 1, dtype=torch.float64), "stride": 16}
        decoded = decode([{"class_logits": logits, "distances": distances, "stride": 8}, second], (16, 16))
        torch.testing.assert_close(decoded["boxes"][0, :4], torch.tensor(
            [[2., 2., 6., 6.], [10., 2., 14., 6.], [2., 10., 6., 14.], [10., 10., 14., 14.]], dtype=torch.float64))
        torch.testing.assert_close(decoded["scores"][0, :4, 0], torch.arange(4, dtype=torch.float64).sigmoid())
        self.assertEqual(decoded["boxes"].shape, (1, 5, 4))
        (decoded["boxes"].square().sum() + decoded["scores"].sum()).backward()
        self.assertTrue(torch.isfinite(logits.grad).all())
        self.assertTrue(torch.isfinite(distances.grad).all())
        self.assertGreater(distances.grad.abs().sum().item(), 0)

    def test_rejects_malformed_outputs(self):
        for image_size in ((0, 16), (8.0, 16), (8,), None, (8, 17)):
            with self.subTest(image_size=image_size), self.assertRaises(ValueError):
                decode(self.fixture(), image_size)
        for outputs in ([], None, [{}], [torch.tensor(1)]):
            with self.subTest(outputs=outputs), self.assertRaises(ValueError):
                decode(outputs, (8, 16))
        for key, value in (("stride", 0), ("stride", 8.0),
                           ("class_logits", torch.zeros(1, 7, 1, 2)),
                           ("distances", torch.zeros(1, 3, 1, 2)),
                           ("distances", torch.full((1, 4, 1, 2), -1.0)),
                           ("distances", torch.full((1, 4, 1, 2), float("nan"))),
                           ("class_logits", torch.full((1, 6, 1, 2), float("inf"))),
                           ("distances", torch.zeros(1, 4, 1, 2, dtype=torch.float64))):
            outputs = self.fixture()
            outputs[0][key] = value
            with self.subTest(key=key, value_shape=getattr(value, "shape", None)), self.assertRaises(ValueError):
                decode(outputs, (8, 16))
        with self.assertRaisesRegex(ValueError, "ascending"):
            decode(self.fixture() * 2, (8, 16))
        outputs = [{"class_logits": torch.zeros(1, 6, 2, 2), "distances": torch.ones(1, 4, 2, 2), "stride": 8},
                   {"class_logits": torch.zeros(2, 6, 1, 1), "distances": torch.ones(2, 4, 1, 1), "stride": 16}]
        with self.assertRaisesRegex(ValueError, "matching batch"):
            decode(outputs, (16, 16))


if __name__ == "__main__":
    unittest.main(verbosity=2)
