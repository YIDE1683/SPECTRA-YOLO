"""Narrow stdlib-only tests; no model import, training, or device access."""

import json
import unittest

from resource_model import ACTIVATION_BUDGET_BYTES, estimate


class ResourceArithmeticTests(unittest.TestCase):
    def test_all_24_variants_and_exact_sum(self):
        for p2 in (0, 32, 64):
            for neck in (64, 96):
                for reduced in (False, True):
                    for classes in (6, 10):
                        with self.subTest(p2=p2, neck=neck, reduced=reduced, classes=classes):
                            result = estimate(p2, neck, reduced, classes)
                            json.dumps(result, allow_nan=False)
                            self.assertEqual(result["conv_count"], 32 - 2 * reduced + 6 * bool(p2))
                            for key in ("deploy_conv_parameters", "logical_macs", "physical_cycle_proxy"):
                                self.assertEqual(result[key], sum(layer[key] for layer in result["layers"]))
                            for output in result["raw_outputs"]:
                                side = 640 // output["stride"]
                                self.assertEqual(result["tensor_shapes"][output["class_logits"]], [1, classes, side, side])
                                self.assertEqual(result["tensor_shapes"][output["distances"]], [1, 4, side, side])

    def test_p2_delta_and_depth_delta(self):
        for classes in (6, 10):
            baseline = estimate(num_classes=classes)
            for p2, delta in ((32, 1_152_000), (64, 2_201_600)):
                full = estimate(p2, num_classes=classes)
                reduced = estimate(p2, reduced_depth=True, num_classes=classes)
                self.assertEqual(full["physical_cycle_proxy"] - baseline["physical_cycle_proxy"], delta)
                self.assertEqual(full["physical_cycle_proxy"] - reduced["physical_cycle_proxy"], 2_880_000)
                self.assertEqual(full["logical_macs"] - reduced["logical_macs"], 1_474_560_000)
                self.assertEqual(full["deploy_conv_parameters"] - reduced["deploy_conv_parameters"], 479_552)
                self.assertLess(reduced["physical_cycle_proxy"], baseline["physical_cycle_proxy"])

    def test_class_channel_tails_change_macs_but_not_proxy(self):
        six, ten = estimate(p2_channels=32), estimate(p2_channels=32, num_classes=10)
        self.assertEqual(six["physical_cycle_proxy"], ten["physical_cycle_proxy"])
        self.assertEqual(ten["deploy_conv_parameters"] - six["deploy_conv_parameters"], 912)
        self.assertEqual(ten["logical_macs"] - six["logical_macs"], 5_427_200)
        outputs = [layer for layer in ten["layers"] if ".class.output.conv" in layer["name"]]
        self.assertTrue(all(layer["padded_output_channels"] == 32 for layer in outputs))

    def test_rectangular_input(self):
        result = estimate(64, 96, True, 10, (320, 640))
        self.assertEqual(result["tensor_shapes"]["input"], [1, 3, 320, 640])
        for out in result["raw_outputs"]:
            stride = out["stride"]
            self.assertEqual(result["tensor_shapes"][out["class_logits"]], [1, 10, 320 // stride, 640 // stride])

    def test_invalid_inputs(self):
        invalid = (
            {"p2_channels": 16}, {"p2_channels": False}, {"neck_channels": 32},
            {"num_classes": 1}, {"num_classes": 6.0}, {"reduced_depth": 1},
            {"input_size": (0, 640)}, {"input_size": (640, -32)},
            {"input_size": (641, 640)}, {"input_size": (True, 640)},
            {"input_size": (640.0, 640)}, {"input_size": "640"},
            {"input_size": (640,)},
        )
        for kwargs in invalid:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                estimate(**kwargs)

    def test_deploy_bias_and_weight_storage(self):
        result = estimate()
        for layer in result["layers"]:
            self.assertTrue(layer["bias"])
            cin, cout = layer["input_shapes"][0][1], layer["output_shape"][1]
            expected = cin * cout * layer["kernel"] ** 2
            self.assertEqual(layer["weight_parameters"], expected)
            self.assertEqual(layer["deploy_conv_parameters"], expected + cout)
            self.assertEqual(layer["bias_storage_bytes_int32"], cout * 4)


class LivenessTests(unittest.TestCase):
    def test_schedule_no_use_after_release_and_exact_final_outputs(self):
        result = estimate(32, 96, False, 10)
        live = {"input"}
        ever_released = set()
        for step in result["schedule"]:
            self.assertTrue(set(step["inputs"]).issubset(live))
            self.assertFalse(set(step["inputs"]) & ever_released)
            live.add(step["allocated"])
            self.assertEqual(live, set(step["live_buffers_during"]))
            for name in step["released_after_operation"]:
                self.assertIn(name, live)
                live.remove(name)
                ever_released.add(name)
            self.assertEqual(live, set(step["live_buffers_after"]))
        self.assertEqual(live, set(result["retained_raw_output_buffers"]))
        self.assertEqual(len(live), 8)
        self.assertTrue(all(name.startswith("head.") for name in live))

    def test_add_nearest_relu_allocate_real_buffers(self):
        result = estimate(64)
        for step in result["schedule"]:
            if step["kind"] in ("add", "nearest2x", "relu"):
                self.assertGreater(step["allocated_output_bytes"], 0)
                self.assertEqual(step["live_bytes_without_accumulator"],
                                 step["live_bytes_before"] + step["allocated_output_bytes"])
                self.assertEqual(step["accumulator_scratch_bytes_int32"], 0)
            if step["kind"] == "add":
                self.assertEqual(len(step["inputs"]), 2)
                self.assertTrue(set(step["inputs"]).issubset(step["live_buffers_during"]))
                self.assertIn(step["allocated"], step["live_buffers_during"])
                sizes = [step["live_buffers_during"][name] for name in step["inputs"]]
                self.assertEqual(sizes, [step["allocated_output_bytes"]] * 2)

    def test_c2_retained_only_when_p2_needs_it(self):
        disabled, enabled = estimate(), estimate(32)
        c2 = "backbone.s2.rep0.relu"
        for result, p2 in ((disabled, False), (enabled, True)):
            first_s3 = next(step for step in result["schedule"] if step["operation"] == "backbone.s3.down.conv")
            self.assertEqual(c2 in first_s3["live_buffers_after"], p2)
        p2_lateral = next(step for step in enabled["schedule"] if step["operation"] == "neck.p2.lateral.conv")
        self.assertIn(c2, p2_lateral["released_after_operation"])

    def test_retained_skips_and_raw_head_output_lifetime(self):
        result = estimate(32)
        at_s5 = next(step for step in result["schedule"] if step["operation"] == "backbone.s5.down.conv")
        self.assertIn("backbone.s3.rep1.relu", at_s5["live_buffers_during"])
        self.assertIn("backbone.s4.rep1.relu", at_s5["live_buffers_during"])
        first_logits = result["raw_outputs"][0]["class_logits"]
        first_index = next(step["step"] for step in result["schedule"] if step["allocated"] == first_logits)
        for step in result["schedule"][first_index:]:
            self.assertIn(first_logits, step["live_buffers_after"])

    def test_scratch_peak_not_double_counted_and_budget_fails_closed(self):
        result = estimate()
        self.assertEqual(result["activation_storage_bits"], 8)
        self.assertEqual(result["raw_output_storage_bits"], 8)
        for step in result["schedule"]:
            self.assertEqual(step["live_bytes_without_accumulator"], sum(step["live_buffers_during"].values()))
            self.assertEqual(step["live_bytes_with_accumulator"],
                             step["live_bytes_without_accumulator"] + step["accumulator_scratch_bytes_int32"])
        self.assertEqual(result["peak_full_frame_live_activation_bytes_without_accumulator"], 6_553_600)
        self.assertEqual(result["peak_full_frame_live_activation_bytes_with_accumulator"], 17_612_800)
        self.assertGreater(result["peak_full_frame_live_activation_bytes_without_accumulator"], ACTIVATION_BUDGET_BYTES)
        self.assertFalse(result["full_frame_budget_pass"])
        self.assertFalse(result["weight_tile_fit_verified"])
        self.assertIn("does not reject all tiled", result["tile_status"])
        small = estimate(input_size=(32, 32))
        self.assertTrue(small["full_frame_budget_pass"])
        self.assertIsNone(small["board_latency_or_acceptance"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
