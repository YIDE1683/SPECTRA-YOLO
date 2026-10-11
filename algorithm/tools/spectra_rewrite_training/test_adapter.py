"""Tests cover feature-access compatibility, not detector accuracy."""

import unittest

import torch

from model_adapter import (CONFIGURATIONS, forward_with_features,
                           freeze_reference, make_model, state_sha256)


class FeatureAdapterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def setUp(self):
        torch.manual_seed(17)

    def test_all_graphs_preserve_outputs_and_feature_shapes(self):
        image = torch.rand(1, 3, 64, 96)
        for configuration in CONFIGURATIONS:
            with self.subTest(configuration=configuration):
                model = make_model(configuration).eval()
                with torch.no_grad():
                    plain = model(image)
                    captured = forward_with_features(model, image)
                self.assertEqual(tuple(model.strides), captured.strides)
                self.assertEqual(len(plain), len(captured.features))
                for index, (expected, actual, feature, stride) in enumerate(zip(
                        plain, captured.outputs, captured.features, captured.strides)):
                    self.assertTrue(torch.equal(expected["class_logits"], actual["class_logits"]))
                    self.assertTrue(torch.equal(expected["distances"], actual["distances"]))
                    channels = model.p2_channels if stride == 4 else 64
                    self.assertEqual(tuple(feature.shape), (1, channels, 64 // stride, 96 // stride))
                    self.assertEqual(len(model.heads[index]._forward_pre_hooks), 0)

    def test_hook_cleanup_after_invalid_forward(self):
        model = make_model("R32").eval()
        with self.assertRaises(ValueError):
            forward_with_features(model, torch.ones(1, 3, 63, 64))
        self.assertTrue(all(not head._forward_pre_hooks for head in model.heads))

    def test_reference_is_independent_and_frozen(self):
        model = make_model("R32").train()
        before = state_sha256(model)
        reference = freeze_reference(model)
        self.assertTrue(model.training)
        self.assertTrue(any(parameter.requires_grad for parameter in model.parameters()))
        self.assertTrue(all(not module.training for module in reference.modules()))
        self.assertTrue(all(not parameter.requires_grad for parameter in reference.parameters()))
        self.assertEqual(before, state_sha256(reference))
        with torch.no_grad():
            next(model.parameters()).add_(1)
        self.assertNotEqual(state_sha256(model), state_sha256(reference))

    def test_features_retain_gradient_without_updating_state(self):
        model = make_model("R32").eval()
        before = state_sha256(model)
        result = forward_with_features(model, torch.rand(1, 3, 64, 64))
        sum(feature.square().mean() for feature in result.features).backward()
        gradients = [parameter.grad for parameter in model.parameters()
                     if parameter.grad is not None]
        self.assertTrue(gradients)
        self.assertTrue(all(torch.isfinite(gradient).all() for gradient in gradients))
        self.assertGreater(sum(float(gradient.abs().sum()) for gradient in gradients), 0)
        self.assertEqual(before, state_sha256(model))

    def test_invalid_configuration_rejected(self):
        with self.assertRaises(ValueError):
            make_model("unknown")

    def test_ten_class_graph_uses_same_feature_contract(self):
        model = make_model("R32", num_classes=10).eval()
        with torch.no_grad():
            result = forward_with_features(model, torch.rand(1, 3, 64, 64))
        self.assertTrue(all(output["class_logits"].shape[1] == 10 for output in result.outputs))


if __name__ == "__main__":
    unittest.main()
