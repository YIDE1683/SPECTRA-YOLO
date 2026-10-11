"""Hand-computable synthetic loss fixtures; no detector/board performance data."""

import math
import unittest

import torch

from losses import global_reconstruction_loss, object_balanced_reconstruction_loss


torch.set_num_threads(2)
EPS = 1e-6


class ReconstructionLossTests(unittest.TestCase):
    def test_unequal_areas_objects_equal_not_pixel_weighted(self):
        reference = torch.ones(1, 1, 1, 4)
        prediction = torch.tensor([[[[3., 2., 2., 2.]]]], requires_grad=True)
        boxes = [torch.tensor([[0., 0., 1., 1.], [1., 0., 4., 1.]])]
        loss, detail = object_balanced_reconstruction_loss([prediction], [reference], boxes, (1, 4), return_details=True)
        self.assertAlmostEqual(loss.item(), (4 + 1) / 2 / (1 + EPS), places=6)
        self.assertEqual(detail["layers"][0]["images"][0]["background_cells"], 0)
        global_loss = global_reconstruction_loss([prediction], [reference])
        self.assertAlmostEqual(global_loss.item(), (4 + 1 + 1 + 1) / 4 / (1 + EPS), places=6)
        self.assertNotAlmostEqual(loss.item(), global_loss.item(), places=2)

    def test_each_object_energy_normalized_independently(self):
        reference = torch.tensor([[[[1., 3., 3., 3.]]]])
        prediction = reference + 1
        boxes = [torch.tensor([[0., 0., 1., 1.], [1., 0., 4., 1.]])]
        loss = object_balanced_reconstruction_loss([prediction], [reference], boxes, (1, 4))
        self.assertAlmostEqual(loss.item(), (1 / (1 + EPS) + 1 / (9 + EPS)) / 2, places=6)
        self.assertNotAlmostEqual(loss.item(), 1 / (7 + EPS), places=3)

    def test_overlaps_count_per_object_background_union_once(self):
        reference = torch.ones(1, 1, 1, 4)
        prediction = reference + torch.tensor([[[[1., 2., 3., 4.]]]])
        boxes = [torch.tensor([[0., 0., 2., 1.], [1., 0., 3., 1.]])]
        loss, detail = object_balanced_reconstruction_loss([prediction], [reference], boxes, (1, 4), return_details=True)
        # Objects: (1+4)/2=2.5 and (4+9)/2=6.5; bg fourth cell=16.
        self.assertAlmostEqual(loss.item(), (4.5 + 0.1 * 16) / (1 + EPS), places=5)
        image = detail["layers"][0]["images"][0]
        self.assertEqual(image["objects"], 2)
        self.assertEqual(image["background_cells"], 1)

    def test_empty_objects_use_background_and_full_coverage_zero_background(self):
        reference = torch.ones(1, 2, 2, 2)
        prediction = reference + 2
        empty, detail = object_balanced_reconstruction_loss([prediction], [reference], [torch.empty(0, 4)], (2, 2), return_details=True)
        full, full_detail = object_balanced_reconstruction_loss([prediction], [reference], [torch.tensor([[0., 0., 2., 2.]])], (2, 2), return_details=True)
        self.assertAlmostEqual(empty.item(), 0.4 / (1 + EPS), places=6)
        self.assertEqual(detail["layers"][0]["images"][0]["object_mean"], 0)
        self.assertAlmostEqual(full.item(), 4 / (1 + EPS), places=6)
        self.assertEqual(full_detail["layers"][0]["images"][0]["background_error"], 0)

    def test_image_coordinates_scale_floor_ceil_clip_and_minimum_cell(self):
        reference = torch.ones(1, 1, 2, 4)
        prediction = reference + 1
        boxes = [torch.tensor([[10., 10., 30., 30.], [-5., -5., 0.1, 0.1], [79.9, 39.9, 90., 45.]])]
        _, detail = object_balanced_reconstruction_loss([prediction], [reference], boxes, (40, 80), return_details=True)
        mapped = detail["layers"][0]["images"][0]["mapped_xyxy"]
        self.assertEqual(mapped, [[0, 0, 2, 2], [0, 0, 1, 1], [3, 1, 4, 2]])
        self.assertTrue(all(x2 > x1 and y2 > y1 for x1, y1, x2, y2 in mapped))

    def test_batch_images_are_equal_weight_not_object_counts(self):
        reference = torch.ones(2, 1, 1, 4)
        prediction = reference + torch.tensor([[[[1., 1., 1., 1.]]], [[[2., 2., 2., 2.]]]])
        boxes = [torch.tensor([[0., 0., 4., 1.]]), torch.tensor([[0., 0., 2., 1.], [2., 0., 4., 1.]])]
        loss = object_balanced_reconstruction_loss([prediction], [reference], boxes, [(1, 4), (1, 4)])
        self.assertAlmostEqual(loss.item(), 2.5 / (1 + EPS), places=6)
        self.assertNotAlmostEqual(loss.item(), 3 / (1 + EPS), places=3)

    def test_level_mean_does_not_weight_feature_map_area(self):
        references = [torch.ones(1, 1, 2, 2), torch.ones(1, 1, 1, 1)]
        predictions = [references[0] + 1, references[1] + 2]
        boxes = [torch.tensor([[0., 0., 40., 40.]])]
        ob = object_balanced_reconstruction_loss(predictions, references, boxes, (40, 40))
        global_loss = global_reconstruction_loss(predictions, references)
        self.assertAlmostEqual(ob.item(), 2.5 / (1 + EPS), places=6)
        self.assertAlmostEqual(global_loss.item(), ob.item(), places=6)

    def test_reference_detached_prediction_gradient_and_boxes_detached(self):
        reference = torch.ones(1, 2, 2, 2, requires_grad=True)
        prediction = torch.zeros(1, 2, 2, 2, requires_grad=True)
        boxes = torch.tensor([[0., 0., 1., 1.]], requires_grad=True)
        ob = object_balanced_reconstruction_loss([prediction], [reference], [boxes], (2, 2))
        global_loss = global_reconstruction_loss([prediction], [reference])
        (ob + global_loss).backward()
        self.assertIsNone(reference.grad)
        self.assertIsNone(boxes.grad)
        self.assertTrue(torch.isfinite(prediction.grad).all())
        self.assertGreater(prediction.grad.abs().sum().item(), 0)

    def test_half_precision_zero_is_finite_fp32_and_differentiable(self):
        for dtype in (torch.float16, torch.bfloat16):
            with self.subTest(dtype=dtype):
                reference = torch.zeros(1, 2, 2, 2, dtype=dtype, requires_grad=True)
                prediction = torch.zeros_like(reference, requires_grad=True)
                ob = object_balanced_reconstruction_loss([prediction], [reference], [torch.empty(0, 4)], (2, 2))
                global_loss = global_reconstruction_loss([prediction], [reference])
                self.assertEqual(ob.dtype, torch.float32)
                self.assertEqual(global_loss.dtype, torch.float32)
                self.assertEqual(ob.item(), 0)
                self.assertEqual(global_loss.item(), 0)
                (ob + global_loss).backward()
                self.assertTrue(torch.equal(prediction.grad, torch.zeros_like(prediction)))
                self.assertIsNone(reference.grad)

    def test_fp16_error_uses_fp32_square_not_overflowing_fp16(self):
        reference = torch.full((1, 1, 2, 2), 300, dtype=torch.float16)
        prediction = torch.zeros_like(reference, requires_grad=True)
        ob = object_balanced_reconstruction_loss([prediction], [reference], [torch.tensor([[0., 0., 2., 2.]])], (2, 2))
        self.assertAlmostEqual(ob.item(), 1, places=6)
        ob.backward()
        self.assertTrue(torch.isfinite(prediction.grad).all())

    def test_background_zero_weight_and_empty_gt_keep_zero_grad(self):
        reference = torch.ones(1, 1, 1, 1, requires_grad=True)
        prediction = torch.zeros_like(reference, requires_grad=True)
        loss = object_balanced_reconstruction_loss([prediction], [reference], [torch.empty(0, 4)], (1, 1), background_weight=0)
        loss.backward()
        self.assertEqual(loss.item(), 0)
        self.assertEqual(prediction.grad.item(), 0)
        self.assertIsNone(reference.grad)

    @staticmethod
    def _single_cell_loss(kind, prediction, reference):
        if kind == "object":
            return object_balanced_reconstruction_loss(
                [prediction], [reference], [torch.tensor([[0., 0., 1., 1.]])], (1, 1))
        return global_reconstruction_loss([prediction], [reference])

    def test_fp16_finite_loss_but_unrepresentable_gradient_fails_closed(self):
        # d(q^2 / eps)/dq is about 200000: finite FP32, not finite FP16.
        for kind in ("object", "global"):
            with self.subTest(kind=kind):
                prediction = torch.full((1, 1, 1, 1), 0.1, dtype=torch.float16, requires_grad=True)
                reference = torch.zeros_like(prediction, requires_grad=True)
                loss = self._single_cell_loss(kind, prediction, reference)
                self.assertTrue(torch.isfinite(loss))
                with self.assertRaisesRegex(FloatingPointError, "FP32 forward"):
                    loss.backward()
                self.assertIsNone(prediction.grad)
                self.assertIsNone(reference.grad)

    def test_fp16_external_point_zero_one_coefficient_can_still_overflow(self):
        # The paper's 0.01 coefficient gives grad=80000 for q=4, ref=0.
        for kind in ("object", "global"):
            with self.subTest(kind=kind):
                prediction = torch.full((1, 1, 1, 1), 4., dtype=torch.float16, requires_grad=True)
                loss = 0.01 * self._single_cell_loss(kind, prediction, torch.zeros_like(prediction))
                self.assertTrue(torch.isfinite(loss))
                with self.assertRaisesRegex(FloatingPointError, "overflows torch.float16"):
                    loss.backward()
                self.assertIsNone(prediction.grad)

    def test_safe_nonzero_half_precision_gradients_remain_supported(self):
        for kind in ("object", "global"):
            for dtype, value in ((torch.float16, 0.1), (torch.bfloat16, 4.0)):
                with self.subTest(kind=kind, dtype=dtype):
                    prediction = torch.full((1, 1, 1, 1), value, dtype=dtype, requires_grad=True)
                    loss = 0.01 * self._single_cell_loss(kind, prediction, torch.zeros_like(prediction))
                    loss.backward()
                    expected = (0.02 * prediction.detach().float() / EPS).to(dtype)
                    torch.testing.assert_close(prediction.grad, expected)
                    self.assertTrue(torch.isfinite(prediction.grad).all())

    def test_fp32_entry_rejects_nonfinite_upstream_gradients(self):
        for kind in ("object", "global"):
            for incoming in (math.inf, math.nan):
                with self.subTest(kind=kind, incoming=incoming):
                    prediction = torch.full((1, 1, 1, 1), 2., requires_grad=True)
                    loss = self._single_cell_loss(kind, prediction, torch.ones_like(prediction))
                    self.assertTrue(torch.isfinite(loss))
                    with self.assertRaisesRegex(FloatingPointError, "before input-dtype conversion"):
                        loss.backward(torch.tensor(incoming))
                    self.assertIsNone(prediction.grad)

    def test_invalid_features_epsilon_shapes_and_reduction_overflow(self):
        x = torch.ones(1, 1, 2, 2)
        for first, second in (([], []), (x, [x]), ([x], []), ([x.long()], [x]),
                              ([x], [torch.ones(1, 2, 2, 2)]), ([x[0]], [x[0]]),
                              ([torch.empty(0, 1, 2, 2)], [torch.empty(0, 1, 2, 2)]),
                              ([x, x], [x, torch.ones(2, 1, 2, 2)]),
                              ([torch.full_like(x, math.nan)], [x]),
                              ([torch.full_like(x, math.inf)], [x]),
                              ([torch.full_like(x, 1e30)], [x]),
                              ([torch.full(x.shape, 1e300, dtype=torch.float64)], [x])):
            with self.subTest(kind="features"), self.assertRaises(ValueError):
                global_reconstruction_loss(first, second)
        for eps in (0, -1, 1e-50, math.nan, math.inf, True, "small"):
            with self.subTest(eps=eps), self.assertRaises(ValueError):
                global_reconstruction_loss([x], [x], eps=eps)
        with self.assertRaises(ValueError):
            global_reconstruction_loss([x], [x], return_details=1)

    def test_invalid_boxes_image_coordinates_and_options(self):
        x = torch.ones(1, 1, 2, 2)
        valid = [torch.tensor([[0., 0., 2., 2.]])]
        for boxes in ([], [torch.ones(4)], [torch.ones(1, 5)], [torch.ones(1, 4, dtype=torch.int64)],
                      [torch.tensor([[0., 0., 0., 1.]])], [torch.tensor([[2., 0., 1., 1.]])],
                      [torch.tensor([[3., 0., 4., 1.]])], [torch.tensor([[0., 0., math.inf, 1.]])],
                      [torch.tensor([[math.nan, 0., 1., 1.]])]):
            with self.subTest(kind="boxes"), self.assertRaises(ValueError):
                object_balanced_reconstruction_loss([x], [x], boxes, (2, 2))
        for image_hw in (None, (0, 2), (2.0, 2), (True, 2), [(2, 2), (2, 2)], (2,)):
            with self.subTest(image_hw=image_hw), self.assertRaises(ValueError):
                object_balanced_reconstruction_loss([x], [x], valid, image_hw)
        for weight in (-1, math.nan, math.inf, True):
            with self.subTest(weight=weight), self.assertRaises(ValueError):
                object_balanced_reconstruction_loss([x], [x], valid, (2, 2), background_weight=weight)


if __name__ == "__main__":
    unittest.main(verbosity=2)
