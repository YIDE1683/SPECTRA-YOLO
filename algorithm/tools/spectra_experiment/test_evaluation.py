"""Synthetic analytic/unit fixtures only, not detector accuracy evidence.

Direct reference calls below independently construct COCO records and invoke
COCOeval; they do not reuse wrapper conversion, ordering, matching or averaging.
"""

from contextlib import redirect_stdout
from copy import deepcopy
import io
import json
import unittest

import numpy as np
import torch
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from evaluation import evaluate_predictions, postprocess


torch.set_num_threads(2)


def record(sample_id="a", boxes=(), labels=None, scores=None, **extra):
    boxes = torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4)
    result = {"sample_id": sample_id, "boxes": boxes,
              "labels": torch.tensor([0]*len(boxes) if labels is None else labels, dtype=torch.int64), **extra}
    if scores is not None:
        result["scores"] = torch.tensor(scores, dtype=torch.float32)
    return result


def meta(sample_id="a", original=(8, 8), output=(8, 8), scale=(1., 1.), padding=(0, 0)):
    return {"sample_id": sample_id, "transform": {"original_hw": original, "output_hw": output,
            "scale_x": scale[0], "scale_y": scale[1], "pad_left": padding[0], "pad_top": padding[1]}}


def raw(rows=1, cols=1, stride=8, classes=1, batch=1, distance=.5, logit=0.):
    return {"class_logits": torch.full((batch, classes, rows, cols), float(logit), requires_grad=True),
            "distances": torch.full((batch, 4, rows, cols), float(distance), requires_grad=True), "stride": stride}


def direct_coco(annotations, detections, image_ids, class_ids, cap=300):
    """Independent reference consumes explicit COCO xywh fixtures only."""
    with redirect_stdout(io.StringIO()):
        truth = COCO()
        truth.dataset = {"info": {}, "images": [{"id": n} for n in image_ids],
                         "categories": [{"id": n} for n in class_ids], "annotations": deepcopy(annotations)}
        truth.createIndex()
        if detections:
            predicted = truth.loadRes(deepcopy(detections))
        else:
            predicted = COCO()
            predicted.dataset = {"images": truth.dataset["images"], "categories": truth.dataset["categories"],
                                 "annotations": []}
            predicted.createIndex()
        evaluation = COCOeval(truth, predicted, "bbox")
        evaluation.params.imgIds = image_ids
        evaluation.params.catIds = class_ids
        evaluation.params.iouThrs = np.linspace(.5, .95, 10)
        evaluation.params.recThrs = np.linspace(0, 1, 101)
        evaluation.params.maxDets = [1, 10, cap]
        evaluation.params.areaRng = [[0., float(2**48)]]
        evaluation.params.areaRngLbl = ["all"]
        evaluation.evaluate()
        evaluation.accumulate()
    index = evaluation.params.maxDets.index(cap)
    return evaluation.eval["precision"][:, :, :, 0, index], evaluation.eval["recall"][:, :, 0, index]


class PostprocessTests(unittest.TestCase):
    def test_single_anchor_decode_stride_and_detach(self):
        outputs = [raw()]
        result = postprocess(outputs, [meta()])[0]
        torch.testing.assert_close(result["boxes"], torch.tensor([[0., 0., 8., 8.]]))
        torch.testing.assert_close(result["scores"], torch.tensor([.5]))
        self.assertEqual(result["labels"].tolist(), [0])
        self.assertEqual(result["boxes"].dtype, torch.float32)
        for key in ("boxes", "scores", "labels"):
            self.assertFalse(result[key].requires_grad)
        self.assertIsNone(outputs[0]["class_logits"].grad)
        self.assertIsNone(outputs[0]["distances"].grad)

    def test_actual_axis_scales_inverse_padding_and_clip(self):
        # Original H3 W5 -> resized H5 W8 -> padded H8 W8 (top1,bottom2).
        metadata = meta(original=(3, 5), output=(8, 8), scale=(8/5, 5/3), padding=(0, 1))
        metadata["transform"].update(resized_hw=(5, 8), pad_right=0, pad_bottom=2)
        outputs = [raw()]
        # input-pixel box [1.6,1,6.4,6] must become [1,0,4,3].
        with torch.no_grad():
            outputs[0]["distances"][0, :, 0, 0] = torch.tensor([2.4, 3., 2.4, 2.]) / 8
        result = postprocess(outputs, [metadata])[0]
        torch.testing.assert_close(result["boxes"], torch.tensor([[1., 0., 4., 3.]]), atol=1e-6, rtol=0)
        broad = postprocess([raw(distance=3.)], [metadata])[0]
        torch.testing.assert_close(broad["boxes"], torch.tensor([[0., 0., 5., 3.]]))

    def test_single_highest_class_not_multilabel_and_class_tie_lowest(self):
        outputs = [raw(classes=3)]
        with torch.no_grad():
            outputs[0]["class_logits"][0, :, 0, 0] = torch.tensor([3., 3., 2.])
        result = postprocess(outputs, [meta()])[0]
        self.assertEqual(result["labels"].tolist(), [0])
        self.assertEqual(len(result["scores"]), 1)

    def test_score_threshold_inclusive_and_zero_area_filtered(self):
        self.assertEqual(len(postprocess([raw()], [meta()], score_threshold=.5)[0]["boxes"]), 1)
        self.assertEqual(len(postprocess([raw()], [meta()], score_threshold=.50001)[0]["boxes"]), 0)
        self.assertEqual(len(postprocess([raw(distance=0)], [meta()])[0]["boxes"]), 0)
        # First anchor lies entirely in top padding and vanishes after clip.
        metadata = meta(original=(2, 8), output=(8, 8), padding=(0, 6))
        self.assertEqual(len(postprocess([raw(rows=2, cols=2, stride=4, distance=.1)], [metadata])[0]["boxes"]), 2)

    def test_stable_class_aware_nms_and_global_cap(self):
        outputs = [raw(cols=3, stride=8, classes=2, distance=10)]
        with torch.no_grad():
            outputs[0]["class_logits"][0, 0, 0] = torch.tensor([2., 2., 1.])
            outputs[0]["class_logits"][0, 1, 0] = torch.tensor([1., 1., 2.])
        metadata = [meta(original=(8, 24), output=(8, 24))]
        result = postprocess(outputs, metadata)[0]
        self.assertEqual(result["labels"].tolist(), [0, 1])
        self.assertEqual(result["source_config"]["retained_anchor_indices"], [0, 2])
        limited = postprocess(outputs, metadata, max_detections=1)[0]
        self.assertEqual(limited["source_config"]["retained_anchor_indices"], [0])
        # IoU==1 is retained at nms_iou=1; suppression is strictly greater.
        self.assertEqual(len(postprocess(outputs, metadata, nms_iou=1)[0]["boxes"]), 3)

    def test_cross_level_flattening_row_major_tie_order(self):
        outputs = [raw(rows=2, cols=2, stride=4), raw(stride=8)]
        result = postprocess(outputs, [meta()], nms_iou=1)[0]
        self.assertEqual(result["source_config"]["retained_anchor_indices"], [0, 1, 2, 3, 4])
        torch.testing.assert_close(result["boxes"][:4], torch.tensor([[0., 0., 4., 4.], [4., 0., 8., 4.],
                                                                      [0., 4., 4., 8.], [4., 4., 8., 8.]]))

    def test_default_global_top300_after_score_sort(self):
        outputs = [raw(cols=401, stride=1, distance=.25)]
        with torch.no_grad():
            outputs[0]["class_logits"][0, 0, 0, 400] = 2
        result = postprocess(outputs, [meta(original=(1, 401), output=(1, 401))])[0]
        self.assertEqual(len(result["boxes"]), 300)
        self.assertEqual(result["source_config"]["retained_anchor_indices"], [400] + list(range(299)))

    def test_batch_metadata_and_input_unchanged(self):
        outputs = [raw(batch=2)]
        before = outputs[0]["distances"].detach().clone()
        result = postprocess(outputs, [meta("a"), meta("b", original=(4, 4), scale=(2, 2))])
        self.assertEqual([r["sample_id"] for r in result], ["a", "b"])
        torch.testing.assert_close(result[1]["boxes"], torch.tensor([[0., 0., 4., 4.]]))
        torch.testing.assert_close(outputs[0]["distances"], before)

    def test_invalid_parameters_raw_shapes_and_finite(self):
        for kwargs in ({"score_threshold": float("nan")}, {"score_threshold": -1}, {"score_threshold": True},
                       {"nms_iou": 1.1}, {"max_detections": 0}, {"max_detections": True}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                postprocess([raw()], [meta()], **kwargs)
        for field, value in (("class_logits", torch.zeros(1, 1, 1, 1, dtype=torch.float16)),
                             ("class_logits", torch.full((1, 1, 1, 1), float("nan"))),
                             ("distances", torch.full((1, 4, 1, 1), -1.)),
                             ("distances", torch.full((1, 4, 1, 1), float("inf"))),
                             ("distances", torch.zeros(1, 3, 1, 1)), ("stride", 4)):
            outputs = [raw()]
            outputs[0][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                postprocess(outputs, [meta()])
        with self.assertRaises(ValueError):
            postprocess([raw(), raw()], [meta()])
        with self.assertRaises(ValueError):
            postprocess([raw(distance=torch.finfo(torch.float32).max)], [meta()])

    def test_invalid_transform_and_ids(self):
        for key, value in (("scale_x", 0), ("scale_y", float("inf")), ("pad_left", -1),
                           ("original_hw", (0, 8)), ("resized_hw", (7, 8)), ("pad_right", 3)):
            metadata = meta()
            metadata["transform"][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                postprocess([raw()], [metadata])
        with self.assertRaises(ValueError):
            postprocess([raw(batch=2)], [meta(), meta()])
        with self.assertRaises(ValueError):
            postprocess([raw(batch=2)], [meta()])
        with self.assertRaises(ValueError):
            postprocess([raw()], [{**meta(), "original_hw": (9, 8)}])


class EvaluationTests(unittest.TestCase):
    def evaluate(self, gt, pred, classes=("defect",), **kwargs):
        return evaluate_predictions(gt, pred, list(classes), **kwargs)

    def test_perfect_and_units_provenance(self):
        result = self.evaluate([record(boxes=[[0, 0, 8, 8]])], [record(boxes=[[0, 0, 8, 8]], scores=[.9])])
        for key in ("ap50_95", "ap50", "ap75", "ar_max_detections"):
            self.assertAlmostEqual(result[key], 1.)
        self.assertEqual(result["protocol"]["percentage_point_multiplier"], 100)
        self.assertEqual(result["protocol"]["max_dets_requested"], [1, 10, 300])
        self.assertEqual(result["provenance"]["pycocotools_version"], "2.0.10")
        self.assertFalse(result["provenance"]["summarize_used"])
        self.assertTrue(all(len(value) == 64 for value in result["provenance"]["source_sha256"].values()))
        json.dumps(result, allow_nan=False)

    def test_missed_and_empty_predictions(self):
        result = self.evaluate([record(boxes=[[0, 0, 8, 8]])], [record(scores=[])])
        self.assertEqual(result["ap50_95"], 0.)
        self.assertEqual(result["ar_max_detections"], 0.)
        self.assertEqual(result["image_count"], 1)

    def test_two_gt_images_only_one_detected_101_point_mean(self):
        gt = [record("a", [[0, 0, 8, 8]]), record("b", [[0, 0, 8, 8]])]
        pred = [record("b", scores=[]), record("a", [[0, 0, 8, 8]], scores=[.9])]
        result = self.evaluate(gt, pred)
        self.assertAlmostEqual(result["ap50_95"], 51/101)
        self.assertEqual(result["ar_max_detections"], .5)

    def test_wrong_class_and_macro_not_classagnostic(self):
        gt = [record(boxes=[[0, 0, 8, 8], [10, 0, 18, 8]], labels=[0, 1])]
        pred = [record(boxes=[[0, 0, 8, 8], [10, 0, 18, 8]], labels=[1, 0], scores=[.9, .8])]
        self.assertEqual(self.evaluate(gt, pred, ("first", "second"))["ap50_95"], 0.)
        one_class = [record(boxes=[[0, 0, 8, 8]], labels=[0], scores=[.9])]
        self.assertAlmostEqual(self.evaluate(gt, one_class, ("first", "second"))["ap50_95"], .5)

    def test_macro_weights_classes_not_object_counts(self):
        frequent = [[i*10, 0, i*10+8, 8] for i in range(4)]
        gt = [record(boxes=frequent + [[0, 10, 8, 18]], labels=[0, 0, 0, 0, 1])]
        pred = [record(boxes=frequent, labels=[0]*4, scores=[.9, .8, .7, .6])]
        result = self.evaluate(gt, pred, ("frequent", "rare"))
        self.assertAlmostEqual(result["ap50_95"], .5)  # not micro object recall4/5
        self.assertEqual([r["gt_count"] for r in result["classwise"]], [4, 1])

    def test_fp_rank_and_duplicate_after_tp_interpolation(self):
        gt = [record(boxes=[[0, 0, 8, 8]])]
        boxes = [[12, 12, 20, 20], [0, 0, 8, 8]]
        self.assertAlmostEqual(self.evaluate(gt, [record(boxes=boxes, scores=[.9, .8])])["ap50_95"], .5)
        self.assertAlmostEqual(self.evaluate(gt, [record(boxes=boxes, scores=[.8, .9])])["ap50_95"], 1.)
        self.assertAlmostEqual(self.evaluate(gt, [record(boxes=[[0, 0, 8, 8]]*2, scores=[.9, .8])])["ap50_95"], 1.)

    def test_stable_ties_within_image_and_across_image_list_order(self):
        gt = [record(boxes=[[0, 0, 8, 8]])]
        fp_first = [record(boxes=[[12, 12, 20, 20], [0, 0, 8, 8]], scores=[.5, .5])]
        tp_first = [record(boxes=[[0, 0, 8, 8], [12, 12, 20, 20]], scores=[.5, .5])]
        self.assertAlmostEqual(self.evaluate(gt, fp_first)["ap50_95"], .5)
        self.assertAlmostEqual(self.evaluate(gt, tp_first)["ap50_95"], 1.)
        gt = [record("b", [[0, 0, 8, 8]]), record("a")]
        pred = [record("b", [[0, 0, 8, 8]], scores=[.5]), record("a", [[12, 12, 20, 20]], scores=[.5])]
        first = self.evaluate(gt, pred)
        second = self.evaluate(list(reversed(gt)), list(reversed(pred)))
        self.assertAlmostEqual(first["ap50_95"], .5)
        self.assertEqual(first, second)

    def test_absent_class_null_not_zero_and_prediction_only_image_counts_fp(self):
        gt = [record("a", [[0, 0, 8, 8]]), record("b")]
        pred = [record("a", [[0, 0, 8, 8]], scores=[.8]),
                record("b", [[12, 12, 20, 20]], scores=[.9])]
        result = self.evaluate(gt, pred, ("present", "absent"))
        self.assertAlmostEqual(result["ap50_95"], .5)
        self.assertIsNone(result["classwise"][1]["ap50_95"])
        self.assertTrue((np.asarray(result["precision"])[:, :, 1] == -1).all())

    def test_empty_gt_dataset_and_images_return_undefined(self):
        for gt, pred in (([], []), ([record()], [record(scores=[])]),
                         ([record()], [record(boxes=[[0, 0, 8, 8]], scores=[.9])])):
            result = self.evaluate(gt, pred)
            self.assertIsNone(result["ap50_95"])
            self.assertIsNone(result["classwise"][0]["ar_max_detections"])
            self.assertEqual(result["status"], "undefined_no_ground_truth")
            json.dumps(result, allow_nan=False)

    def test_multiple_iou_thresholds_hand_calculation(self):
        # Intersection60/union100 gives IoU0.6: passes .50,.55,.60 only.
        result = self.evaluate([record(boxes=[[0, 0, 10, 10]])],
                               [record(boxes=[[0, 0, 6, 10]], scores=[.9])])
        self.assertAlmostEqual(result["ap50_95"], .3)
        self.assertAlmostEqual(result["ap50"], 1.)
        self.assertEqual(result["ap75"], 0.)
        self.assertEqual(result["recall"], [[1.], [1.], [1.], [0.], [0.], [0.], [0.], [0.], [0.], [0.]])

    def test_maxdet_300_not_hardcoded_100_and_rank301_discarded(self):
        gt = [record(boxes=[[0, 0, 8, 8]])]
        pred150 = [record(boxes=[[10, 10, 18, 18]]*149 + [[0, 0, 8, 8]], scores=[.9]*149 + [.8])]
        self.assertAlmostEqual(self.evaluate(gt, pred150)["ap50_95"], 1/150)
        pred301 = [record(boxes=[[10, 10, 18, 18]]*300 + [[0, 0, 8, 8]], scores=[.9]*300 + [.8])]
        result = self.evaluate(gt, pred301)
        self.assertEqual(result["ap50_95"], 0.)
        self.assertEqual(result["prediction_count_evaluated"], 300)

    def test_global_not_only_per_class_cap_and_small_custom_limit(self):
        gt = [record(boxes=[[0, 0, 8, 8]], labels=[1])]
        pred = [record(boxes=[[10, 10, 18, 18], [0, 0, 8, 8]], labels=[0, 1], scores=[.9, .8])]
        result = self.evaluate(gt, pred, ("other", "present"), max_detections=1)
        self.assertEqual(result["ap50_95"], 0.)
        self.assertEqual(result["prediction_count_evaluated"], 1)
        self.assertAlmostEqual(self.evaluate(gt, pred, ("other", "present"), max_detections=2)["ap50_95"], 1.)

    def test_all_eligible_areas_including_larger_than_default_coco_bound(self):
        big = [[0, 0, 200000, 200000]]
        result = self.evaluate([record(boxes=big)], [record(boxes=big, scores=[.9])])
        self.assertAlmostEqual(result["ap50_95"], 1.)

    def test_direct_cocoeval_independent_multiclass_fixture_all_tables(self):
        # Two images, class1 GT2, class2 GT1, class3 absent; partial box IoU.6.
        gt = [record("b", [[0, 0, 10, 10]], labels=[0]),
              record("a", [[0, 0, 10, 10], [20, 0, 30, 10]], labels=[0, 1])]
        pred = [record("a", [[40, 0, 50, 10], [0, 0, 10, 10], [20, 0, 30, 10]],
                       labels=[0, 0, 1], scores=[.95, .9, .7]),
                record("b", [[0, 0, 6, 10]], labels=[0], scores=[.8])]
        annotations = [{"id": 1, "image_id": 1, "category_id": 1, "bbox": [0, 0, 10, 10], "area": 100, "iscrowd": 0},
                       {"id": 2, "image_id": 1, "category_id": 2, "bbox": [20, 0, 10, 10], "area": 100, "iscrowd": 0},
                       {"id": 3, "image_id": 2, "category_id": 1, "bbox": [0, 0, 10, 10], "area": 100, "iscrowd": 0}]
        detections = [{"image_id": 1, "category_id": 1, "bbox": [40, 0, 10, 10], "score": float(np.float32(.95))},
                      {"image_id": 1, "category_id": 1, "bbox": [0, 0, 10, 10], "score": float(np.float32(.9))},
                      {"image_id": 1, "category_id": 2, "bbox": [20, 0, 10, 10], "score": float(np.float32(.7))},
                      {"image_id": 2, "category_id": 1, "bbox": [0, 0, 6, 10], "score": float(np.float32(.8))}]
        expected_precision, expected_recall = direct_coco(annotations, detections, [1, 2], [1, 2, 3])
        result = self.evaluate(gt, pred, ("first", "second", "absent"))
        np.testing.assert_array_equal(np.asarray(result["precision"]), expected_precision)
        np.testing.assert_array_equal(np.asarray(result["recall"]), expected_recall)
        self.assertAlmostEqual(result["ap50_95"], expected_precision[expected_precision >= 0].mean())

    def test_direct_cocoeval_empty_fixture_all_tables(self):
        annotations = [{"id": 1, "image_id": 1, "category_id": 1, "bbox": [0, 0, 8, 8], "area": 64, "iscrowd": 0}]
        expected_precision, expected_recall = direct_coco(annotations, [], [1], [1])
        result = self.evaluate([record(boxes=[[0, 0, 8, 8]])], [record(scores=[])])
        np.testing.assert_array_equal(np.asarray(result["precision"]), expected_precision)
        np.testing.assert_array_equal(np.asarray(result["recall"]), expected_recall)

    def test_inputs_unchanged_and_no_grad_graph(self):
        gt, pred = [record(boxes=[[0, 0, 8, 8]])], [record(boxes=[[0, 0, 8, 8]], scores=[.9])]
        pred[0]["boxes"].requires_grad_()
        pred[0]["scores"].requires_grad_()
        before_gt, before_pred = deepcopy(gt), deepcopy(pred)
        self.evaluate(gt, pred)
        self.assertEqual(set(pred[0]), set(before_pred[0]))
        for key in ("boxes", "labels", "scores"):
            torch.testing.assert_close(pred[0][key], before_pred[0][key])
        torch.testing.assert_close(gt[0]["boxes"], before_gt[0]["boxes"])
        self.assertIsNone(pred[0]["boxes"].grad)
        self.assertIsNone(pred[0]["scores"].grad)

    def test_id_matching_and_duplicate_errors(self):
        gt = [record("a", [[0, 0, 8, 8]])]
        for pred in ([], [record("b", scores=[])], [record("a", scores=[])]*2):
            with self.assertRaises(ValueError):
                self.evaluate(gt, pred)
        with self.assertRaises(ValueError):
            self.evaluate(gt*2, [record("a", scores=[])])

    def test_ignore_crowd_difficult_and_visdrone_explicitly_rejected(self):
        for extra in ({"ignore_boxes": [[0, 0, 8, 8]]}, {"ignore_metadata": [{}]}, {"iscrowd": [1]},
                      {"difficult": [1]}, {"ignore": True}, {"dataset": "VisDrone"}, {"ignore": float("nan")}):
            with self.subTest(extra=extra), self.assertRaises(ValueError):
                self.evaluate([record(boxes=[[0, 0, 8, 8]], **extra)], [record(scores=[])])
        allowed = record(boxes=[[0, 0, 8, 8]], ignore_boxes=torch.empty(0, 4), difficult_policy="legacy_include")
        self.assertAlmostEqual(self.evaluate([allowed], [record(boxes=[[0, 0, 8, 8]], scores=[.9])])["ap50_95"], 1.)

    def test_invalid_boxes_scores_labels_shapes_classes_and_dimensions(self):
        gt = [record(boxes=[[0, 0, 8, 8]])]
        good = record(boxes=[[0, 0, 8, 8]], scores=[.9])
        changes = [("boxes", torch.zeros(1, 4)), ("boxes", torch.tensor([[0., 0., float("nan"), 8.]])),
                   ("boxes", torch.tensor([[-1., 0., 8., 8.]])), ("boxes", torch.zeros(1, 3)),
                   ("boxes", torch.zeros(1, 4, dtype=torch.float16)), ("labels", torch.tensor([1])),
                   ("labels", torch.tensor([0.])), ("labels", torch.tensor([0, 0])),
                   ("scores", torch.tensor([float("inf")])), ("scores", torch.tensor([-1.])),
                   ("scores", torch.tensor([1.01])), ("scores", torch.ones(1, 1))]
        for key, value in changes:
            bad = {**good, key: value}
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.evaluate(gt, [bad])
        for classes in ([], ["x", "x"], [""], "x"):
            with self.assertRaises(ValueError):
                evaluate_predictions(gt, [good], classes)
        with self.assertRaises(ValueError):
            self.evaluate([{**gt[0], "original_hw": (4, 4)}], [good])
        with self.assertRaises(ValueError):
            self.evaluate(gt, [{**good, "original_hw": (4, 4)}])
        with self.assertRaises(ValueError):
            self.evaluate(gt, [good], max_detections=False)


if __name__ == "__main__":
    unittest.main(verbosity=2)
