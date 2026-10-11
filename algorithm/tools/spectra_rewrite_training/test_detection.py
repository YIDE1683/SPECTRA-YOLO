"""Synthetic hand-calculated detection fixtures, not detector accuracy results."""

import math
import unittest

import torch

from detection import decode_training_outputs, detection_loss


torch.set_num_threads(2)


def output(rows=1, cols=1, stride=2, batch=1, classes=1, distance=0.5, logit=0.0):
    return {"class_logits": torch.full((batch, classes, rows, cols), float(logit), requires_grad=True),
            "distances": torch.full((batch, 4, rows, cols), float(distance), requires_grad=True), "stride": stride}


def target(boxes=(), labels=None):
    boxes = torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4)
    if labels is None:
        labels = [0] * len(boxes)
    return {"boxes": boxes, "labels": torch.tensor(labels, dtype=torch.int64)}


class DetectionTests(unittest.TestCase):
    def test_decode_stride_row_major_and_no_training_clamp(self):
        raw = output(rows=2, cols=3, stride=4, distance=1)
        decoded = decode_training_outputs([raw], (8, 12))
        torch.testing.assert_close(decoded["centers"], torch.tensor([[2., 2.], [6., 2.], [10., 2.], [2., 6.], [6., 6.], [10., 6.]]))
        torch.testing.assert_close(decoded["boxes_unclamped"][0, 0], torch.tensor([-2., -2., 6., 6.]))
        self.assertEqual(decoded["class_logits"].shape, (1, 6, 1))
        self.assertTrue(decoded["boxes_unclamped"].requires_grad)

    def test_perfect_box_binary_bce_and_four_side_targets(self):
        raw = output()
        result = detection_loss([raw], [target([[0., 0., 2., 2.]])], (2, 2), return_details=True)
        self.assertEqual(result["num_positive"], 1)
        self.assertAlmostEqual(result["classification"].item(), math.log(2), places=6)
        self.assertAlmostEqual(result["giou"].item(), 0, places=6)
        self.assertAlmostEqual(result["ltrb"].item(), 0, places=6)
        self.assertAlmostEqual(result["loss"].item(), 0.5 * math.log(2), places=6)
        torch.testing.assert_close(result["assignment"]["ltrb_targets"], torch.full((1, 1, 4), 0.5))

    def test_nonperfect_box_giou_ltrb_and_weighted_total_hand_calculation(self):
        # Pred [0.5,0.5,1.5,1.5] inside GT [0,0,2,2]: IoU=GIoU=1/4.
        raw = output(distance=0.25)
        result = detection_loss([raw], [target([[0., 0., 2., 2.]])], (2, 2))
        self.assertAlmostEqual(result["giou"].item(), 0.75, places=6)
        self.assertAlmostEqual(result["ltrb"].item(), 1.0, places=6)
        self.assertAlmostEqual(result["loss"].item(), 0.5 * math.log(2) + 7.5 * .75 + 1, places=5)

    def test_outside_prediction_keeps_nonzero_regression_gradient(self):
        raw = output(distance=1.5)
        result = detection_loss([raw], [target([[0., 0., 2., 2.]])], (2, 2))
        # Unclamped [-2,-2,4,4]: IoU=1/9, not perfect after clipping.
        self.assertAlmostEqual(result["giou"].item(), 8 / 9, places=6)
        result["giou"].backward()
        self.assertTrue(torch.isfinite(raw["distances"].grad).all())
        self.assertGreater(raw["distances"].grad.abs().sum().item(), 0)

    def test_giou_includes_enclosing_area_penalty(self):
        raw = output()
        with torch.no_grad():
            raw["distances"][0, :, 0, 0] = torch.tensor([.25, .25, 1., 1.])
        # Pred [.5,.5,3,3], GT [0,0,2,2]: intersection2.25, union8,
        # enclosing9; GIoU=2.25/8 - (9-8)/9, unlike plain IoU.
        result = detection_loss([raw], [target([[0., 0., 2., 2.]])], (2, 2))
        self.assertAlmostEqual(result["giou"].item(), 1 - (2.25 / 8 - 1 / 9), places=6)
        self.assertAlmostEqual(result["ltrb"].item(), 1.5, places=6)

    def test_batch_global_positive_normalization_and_one_hot_targets(self):
        raw = output(batch=2, classes=2)
        result = detection_loss([raw], [target([[0., 0., 2., 2.]], [1]), target()], (2, 2), return_details=True)
        self.assertEqual(result["stats"]["normalizer"], 1)
        self.assertAlmostEqual(result["classification"].item(), 4 * math.log(2), places=6)
        self.assertEqual(result["assignment"]["binary_class_targets"].tolist(), [[[0., 1.]], [[0., 0.]]])

    def test_duplicate_gt_iou_tie_prefers_original_index(self):
        raw = output(classes=2)
        result = detection_loss([raw], [target([[0., 0., 2., 2.], [0., 0., 2., 2.]], [1, 0])], (2, 2), return_details=True)
        self.assertEqual(result["assignment"]["gt_indices"].tolist(), [[0]])
        self.assertEqual(result["stats"]["totals"]["anchors_multiple_gt_nominations"], 1)
        self.assertEqual(result["assignment"]["per_image"][0]["unmatched_gt_indices"], [1])
        self.assertEqual(result["assignment"]["binary_class_targets"].tolist(), [[[0., 1.]]])

    def test_conflict_higher_iou_beats_earlier_gt(self):
        raw = output(distance=.25)
        result = detection_loss([raw], [target([[0., 0., 2., 2.], [.5, .5, 1.5, 1.5]])], (2, 2), return_details=True)
        self.assertEqual(result["assignment"]["gt_indices"].tolist(), [[1]])

    def test_alignment_topk_ties_use_anchor_order_and_topk_limits_count(self):
        # Zero-area predictions yield zero alignment. Geometric candidates are
        # retained and a tie selects row-major indices rather than inventing GT.
        raw = output(rows=1, cols=12, stride=2, distance=0)
        result = detection_loss([raw], [target([[0., 0., 24., 2.]])], (2, 24), return_details=True)
        self.assertEqual(result["num_positive"], 10)
        self.assertEqual(result["assignment"]["positive_mask"].tolist(), [[True] * 10 + [False] * 2])
        again = detection_loss([raw], [target([[0., 0., 24., 2.]])], (2, 24), return_details=True)
        self.assertTrue(torch.equal(result["assignment"]["gt_indices"], again["assignment"]["gt_indices"]))

    def test_alignment_uses_class_and_iou_factors(self):
        raw = output(rows=1, cols=2, stride=2, distance=0.5)
        with torch.no_grad():
            raw["class_logits"][0, 0, 0] = torch.tensor([-4., 4.])
        result = detection_loss([raw], [target([[0., 0., 4., 2.]])], (2, 4), topk=1, return_details=True)
        self.assertEqual(result["assignment"]["gt_indices"].tolist(), [[-1, 0]])

    def test_iou_power_can_outweigh_higher_class_confidence(self):
        raw = output(rows=1, cols=2, stride=2, distance=.5)
        with torch.no_grad():
            raw["class_logits"][0, 0, 0] = torch.tensor([-2., 2.])
            raw["distances"][0, :, 0, 0] = torch.tensor([.5, .5, 1.5, .5])
        # First IoU1 vs second IoU.5: sqrt(sigmoid(-2)) exceeds
        # sqrt(sigmoid(2))*.5**6 even though its class confidence is lower.
        result = detection_loss([raw], [target([[0., 0., 4., 2.]])], (2, 4), topk=1, return_details=True)
        self.assertEqual(result["assignment"]["gt_indices"].tolist(), [[0, -1]])

    def test_no_gt_and_no_candidate_are_reported_without_fabrication(self):
        raw = output(rows=1, cols=2)
        empty = detection_loss([raw], [target()], (2, 4))
        self.assertEqual(empty["num_positive"], 0)
        self.assertAlmostEqual(empty["classification"].item(), 2 * math.log(2), places=6)
        self.assertEqual(empty["giou"].item(), 0)
        self.assertEqual(empty["ltrb"].item(), 0)
        # First center x=1 is exactly on right edge, so strict GT membership fails.
        missed = detection_loss([raw], [target([[0., 0., 1., 2.]])], (2, 4), return_details=True)
        self.assertEqual(missed["num_positive"], 0)
        self.assertEqual(missed["stats"]["totals"]["gt_without_candidates"], 1)
        self.assertEqual(missed["assignment"]["per_image"][0]["no_candidate_gt_indices"], [0])

    def test_all_ignore_is_zero_and_gradients_zero(self):
        raw = output(rows=2, cols=2, classes=2)
        result = detection_loss([raw], [target([[0., 0., 2., 2.]])], (4, 4),
                                ignore_boxes=[torch.tensor([[0., 0., 4., 4.]])], return_details=True)
        self.assertEqual(result["loss"].item(), 0)
        self.assertEqual(result["stats"]["totals"]["gt_ignored_overlap"], 1)
        self.assertEqual(result["stats"]["totals"]["anchors_classification_eligible"], 0)
        result["loss"].backward()
        self.assertTrue(torch.equal(raw["class_logits"].grad, torch.zeros_like(raw["class_logits"])))
        self.assertTrue(torch.equal(raw["distances"].grad, torch.zeros_like(raw["distances"])))

    def test_ignored_gt_full_rectangle_propagates_beyond_original_ignore(self):
        raw = output(rows=1, cols=4, stride=2)
        result = detection_loss([raw], [target([[0., 0., 6., 2.]])], (2, 8),
                                ignore_boxes=[torch.tensor([[0., 0., 2., 2.]])], return_details=True)
        self.assertEqual(result["assignment"]["classification_eligible_mask"].tolist(), [[False, False, False, True]])
        stats = result["stats"]["totals"]
        self.assertEqual(stats["anchors_original_ignore_in_content"], 1)
        self.assertEqual(stats["anchors_gt_propagation_additional"], 2)
        self.assertEqual(stats["gt_ignored_overlap"], 1)
        self.assertAlmostEqual(result["classification"].item(), math.log(2), places=6)

    def test_padding_mask_excludes_classification_and_gt_crossing_is_rejected(self):
        raw = output(rows=2, cols=2, stride=2)
        content = [torch.tensor([0., 0., 2., 4.])]
        result = detection_loss([raw], [target()], (4, 4), valid_regions=content, return_details=True)
        self.assertEqual(result["assignment"]["classification_eligible_mask"].tolist(), [[True, False, True, False]])
        self.assertEqual(result["stats"]["totals"]["anchors_padding"], 2)
        self.assertAlmostEqual(result["classification"].item(), 2 * math.log(2), places=6)
        with self.assertRaisesRegex(ValueError, "valid content"):
            detection_loss([raw], [target([[0., 0., 3., 3.]])], (4, 4), valid_regions=content)

    def test_ignore_boundary_conservative_and_touching_gt_not_removed(self):
        raw = output(rows=1, cols=2, stride=2)
        result = detection_loss([raw], [target([[2., 0., 4., 2.]])], (2, 4),
                                ignore_boxes=[torch.tensor([[0., 0., 1., 2.]])], return_details=True)
        self.assertEqual(result["assignment"]["classification_eligible_mask"].tolist(), [[False, True]])
        self.assertEqual(result["num_positive"], 1)
        self.assertEqual(result["stats"]["totals"]["gt_ignored_overlap"], 0)

    def test_assignment_detached_only_raw_predictions_receive_gradients(self):
        raw = output(distance=.25, classes=2)
        gt = target([[0., 0., 2., 2.]], [1])
        gt["boxes"].requires_grad_()
        content = torch.tensor([0., 0., 2., 2.], requires_grad=True)
        result = detection_loss([raw], [gt], (2, 2), valid_regions=[content], return_details=True)
        for key, value in result["assignment"].items():
            if isinstance(value, torch.Tensor):
                self.assertFalse(value.requires_grad, key)
                self.assertIsNone(value.grad_fn, key)
        result["loss"].backward()
        self.assertIsNone(gt["boxes"].grad)
        self.assertIsNone(content.grad)
        self.assertTrue(torch.isfinite(raw["class_logits"].grad).all())
        self.assertTrue(torch.isfinite(raw["distances"].grad).all())
        self.assertGreater(raw["class_logits"].grad.abs().sum().item(), 0)
        self.assertGreater(raw["distances"].grad.abs().sum().item(), 0)

    def test_multilevel_stride_applied_once_for_ltrb_target(self):
        first, second = output(rows=2, cols=2, stride=2), output(stride=4)
        result = detection_loss([first, second], [target([[0., 0., 4., 4.]])], (4, 4), return_details=True)
        torch.testing.assert_close(result["assignment"]["strides"], torch.tensor([2., 2., 2., 2., 4.]))
        torch.testing.assert_close(result["assignment"]["ltrb_targets"][0, 0], torch.tensor([.5, .5, 1.5, 1.5]))
        torch.testing.assert_close(result["assignment"]["ltrb_targets"][0, 4], torch.full((4,), .5))

    def test_reject_invalid_predictions_and_configuration(self):
        examples = [None, [], [{}], [output(stride=0)], [output(stride=True)]]
        for field, value in (("class_logits", torch.ones(1, 1, 1, 1, dtype=torch.float16)),
                             ("class_logits", torch.full((1, 1, 1, 1), math.inf)),
                             ("class_logits", torch.ones(1, 1, 1)),
                             ("distances", torch.full((1, 4, 1, 1), math.nan)),
                             ("distances", torch.full((1, 4, 1, 1), -.1)),
                             ("distances", torch.ones(1, 3, 1, 1)),
                             ("distances", torch.full((1, 4, 1, 1), 3e38))):
            raw = output()
            raw[field] = value
            examples.append([raw])
        examples.append([output(), output()])
        for outputs in examples:
            with self.subTest(kind="raw"), self.assertRaises(ValueError):
                detection_loss(outputs, [target()], (2, 2))
        for kwargs in ({"topk": 0}, {"topk": True}, {"class_exponent": 0},
                       {"iou_exponent": math.nan}, {"return_details": 1}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                detection_loss([output()], [target()], (2, 2), **kwargs)
        for size in ((0, 2), (2., 2), (2, 3), (2,), None):
            with self.subTest(size=size), self.assertRaises(ValueError):
                detection_loss([output()], [target()], size)

    def test_reject_invalid_targets_and_regions(self):
        bad_targets = [[], [{}], [target([[0., 0., 2., 2.]], [-1])], [target([[0., 0., 2., 2.]], [1])],
                       [target([[0., 0., 0., 2.]])], [target([[-1., 0., 1., 1.]])],
                       [target([[0., 0., 3., 2.]])], [target([[0., 0., math.nan, 2.]])]]
        wrong_labels = target([[0., 0., 2., 2.]])
        wrong_labels["labels"] = torch.tensor([0.])
        bad_targets.append([wrong_labels])
        for gt in bad_targets:
            with self.subTest(kind="GT"), self.assertRaises(ValueError):
                detection_loss([output()], gt, (2, 2))
        for kwargs in ({"valid_regions": []}, {"valid_regions": [torch.ones(2, 2)]},
                       {"valid_regions": [torch.tensor([0., 0., 0., 1.])]},
                       {"ignore_boxes": []}, {"ignore_boxes": [torch.tensor([[0., 0., 4., 4.]])]},
                       {"ignore_boxes": [torch.tensor([[0., 0., math.inf, 1.]])]}):
            with self.subTest(kind="region"), self.assertRaises(ValueError):
                detection_loss([output()], [target()], (2, 2), **kwargs)

    def test_nonfinite_upstream_gradient_is_rejected(self):
        raw = output(distance=.25)
        result = detection_loss([raw], [target([[0., 0., 2., 2.]])], (2, 2))
        with self.assertRaises(FloatingPointError):
            result["loss"].backward(torch.tensor(math.inf))


if __name__ == "__main__":
    unittest.main(verbosity=2)
