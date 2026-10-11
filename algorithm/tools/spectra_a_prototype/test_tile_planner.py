"""Tests for resident capacity screen; no actual tensor or hardware execution."""

from copy import deepcopy
import json
import unittest

from resource_model import estimate
from tile_planner import METADATA_RESERVE_BYTES, plan_graph


class TilePlannerTests(unittest.TestCase):
    def test_all_24_configurations_each_operation_screened(self):
        for p2 in (0, 32, 64):
            for neck in (64, 96):
                for reduced in (False, True):
                    for classes in (6, 10):
                        with self.subTest(p2=p2, neck=neck, reduced=reduced, classes=classes):
                            reference = estimate(p2, neck, reduced, classes)
                            plan = plan_graph(reference)
                            json.dumps(plan, allow_nan=False)
                            self.assertEqual(plan["operation_count"], len(reference["operations"]))
                            self.assertEqual(plan["conv_count"], reference["conv_count"])
                            self.assertTrue(plan["bytes_capacity_pass"])
                            self.assertFalse(plan["all_implementation_gates_pass"])
                            for op in plan["operation_plans"]:
                                self.assertTrue(op["bytes_capacity_pass"])
                                self.assertLessEqual(op["resident_activation_bytes"], 1_048_576)
                                self.assertLessEqual(op["tile_weight_and_bias_bytes"], 262_144)
                                self.assertEqual(op["selected_spatial_block"], 32)
                                self.assertEqual(op["resident_activation_bytes"], sum(op["activation_components"].values()))
                                self.assertEqual(op["tile_weight_and_bias_bytes"], sum(op["weight_components"].values()))

    def test_stride_two_halo_and_rgb_tail(self):
        plan = plan_graph(estimate())
        first = plan["operation_plans"][0]
        self.assertEqual(first["name"], "backbone.s1.down.conv")
        self.assertEqual(first["input_channel_block"], 3)
        self.assertEqual(first["input_channel_blocks"], 1)
        self.assertEqual(first["input_tile_shape_chw_including_halo_and_zero_padding"], [3, 65, 65])
        self.assertEqual(first["activation_components"]["input_pingpong_bytes"], 2 * 65 * 65 * 3)
        self.assertEqual(first["activation_components"]["line_buffer_int8_bytes"], 2 * 65 * 3)
        self.assertIn("integer zero", first["padding_policy"])

    def test_class_and_regression_channel_tails(self):
        for classes in (6, 10):
            plan = plan_graph(estimate(32, num_classes=classes))
            cls = next(op for op in plan["operation_plans"] if op["name"] == "head.p2.class.output.conv")
            reg = next(op for op in plan["operation_plans"] if op["name"] == "head.p2.regression.output.conv")
            self.assertEqual(cls["output_channel_block"], classes)
            self.assertEqual(cls["last_output_channel_block"], classes)
            self.assertEqual(reg["output_channel_block"], 4)
            self.assertEqual(cls["weight_components"]["bias_block_int32_bytes"], classes * 4)
            self.assertNotIn("bias_block_int32_bytes", cls["activation_components"])

    def test_nonconv_add_pingpong_and_nearest_footprint(self):
        plan = plan_graph(estimate(64))
        addition = next(op for op in plan["operation_plans"] if op["name"] == "neck.p2.add")
        self.assertEqual(addition["input_operand_pingpong_bytes"], [2 * 32 * 32 * 32] * 2)
        self.assertEqual(addition["resident_activation_bytes"], 6 * 32 * 32 * 32 + METADATA_RESERVE_BYTES)
        nearest = next(op for op in plan["operation_plans"] if op["name"] == "neck.p2.up")
        self.assertEqual(nearest["input_operand_tile_shapes_chw"], [[32, 17, 17]])
        self.assertFalse(addition["inplace_assumed"])
        self.assertFalse(nearest["inplace_assumed"])

    def test_largest_valid_candidate_and_no_fit(self):
        # Nonconv remains fixed32 and can fail under this tight budget; test
        # the independent conv choice rather than invent whole-graph success.
        plan = plan_graph(estimate(), activation_budget_bytes=100_000)
        rep = next(op for op in plan["operation_plans"] if op["name"] == "backbone.s2.rep0.conv")
        self.assertEqual(rep["selected_spatial_block"], 16)
        self.assertFalse(rep["candidate_screens"][0]["bytes_capacity_pass"])
        self.assertTrue(rep["candidate_screens"][1]["bytes_capacity_pass"])
        too_small = plan_graph(estimate(), activation_budget_bytes=0)
        self.assertFalse(too_small["bytes_capacity_pass"])
        self.assertEqual(len(too_small["failed_operations"]), too_small["operation_count"])
        self.assertTrue(all(op["selected_spatial_block"] is None for op in too_small["operation_plans"]))
        no_weight = plan_graph(estimate(), weight_budget_bytes=0)
        self.assertFalse(no_weight["bytes_capacity_pass"])
        self.assertEqual(len(no_weight["failed_operations"]), no_weight["conv_count"])
        self.assertGreater(too_small["max_resident_activation_bytes"], 0)

    def test_invalid_negative_budgets_fail_closed(self):
        for invalid in (-1, -100, 1.5, True, None):
            for key in ("activation_budget_bytes", "weight_budget_bytes"):
                with self.subTest(key=key, value=invalid), self.assertRaises(ValueError):
                    plan_graph(estimate(), **{key: invalid})

    def test_ddr_peak_and_traffic_are_explicit_not_board_measurements(self):
        reference = estimate(64)
        plan = plan_graph(reference)
        self.assertFalse(reference["full_frame_budget_pass"])
        self.assertTrue(plan["bytes_capacity_pass"])
        self.assertEqual(plan["external_ddr_peak_activation_bytes"], reference["peak_full_frame_live_activation_bytes_without_accumulator"])
        self.assertEqual(plan["external_ddr_peak_activation_plus_all_parameters_bytes"],
                         plan["external_ddr_peak_activation_bytes"] + plan["external_ddr_all_weights_biases_bytes"])
        self.assertEqual(plan["external_ddr_retained_raw_output_bytes"], reference["retained_raw_output_bytes"])
        self.assertEqual(plan["materialized_ddr_transfer_lower_bound_bytes"],
                         sum(op["materialized_ddr_activation_transfer_lower_bound_bytes"]
                             + op["materialized_ddr_parameter_transfer_lower_bound_bytes"] for op in plan["operation_plans"]))
        self.assertFalse(plan["external_ddr_capacity_verified"])
        self.assertFalse(plan["all_implementation_gates_pass"])
        self.assertIsNone(plan["board_latency_or_acceptance"])

    def test_small_and_rectangular_boundary_tiles(self):
        plan = plan_graph(estimate(32, input_size=(32, 64)))
        deep = next(op for op in plan["operation_plans"] if op["name"] == "backbone.s5.down.conv")
        self.assertEqual(deep["output_tile_shape_chw"], [32, 1, 2])
        self.assertEqual(deep["input_tile_shape_chw_including_halo_and_zero_padding"], [16, 3, 5])
        self.assertEqual(deep["spatial_tiles"], 1)
        self.assertTrue(plan["bytes_capacity_pass"])

    def test_malformed_or_unsupported_graph_not_silently_accepted(self):
        with self.assertRaises(ValueError):
            plan_graph({})
        for mutate in (
            lambda r: r["operations"][0].update(kind="unsupported"),
            lambda r: r["operations"][0].update(inputs=["missing"]),
            lambda r: r.update(activation_storage_bits=4),
            lambda r: r["operations"][0].update(output_shape=[1, 32, 319, 320]),
        ):
            bad = deepcopy(estimate())
            mutate(bad)
            with self.assertRaises(ValueError):
                plan_graph(bad)


if __name__ == "__main__":
    unittest.main(verbosity=2)
