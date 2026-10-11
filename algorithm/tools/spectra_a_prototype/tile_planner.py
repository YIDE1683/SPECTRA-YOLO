"""DDR-backed, layer-by-layer resident tile BYTE-CAPACITY screen only.

The full graph's tensors and weights live in external DDR. On-chip accounting
contains explicit input/output ping-pong tiles and scratch. This module does
not implement integer operators, bank/port allocation, DMA, synchronization,
or a measured bandwidth/latency model. Passing bytes never passes deployment.
"""

from __future__ import annotations

from typing import Any


DEFAULT_ACTIVATION_BUDGET_BYTES = 1_048_576
DEFAULT_WEIGHT_BUDGET_BYTES = 262_144
METADATA_RESERVE_BYTES = 4_096
CONV_SPATIAL_CANDIDATES = (32, 16, 8)
NONCONV_SPATIAL_BLOCK = 32


def _ceildiv(numerator: int, denominator: int) -> int:
    return (numerator + denominator - 1) // denominator


def _shape(value: Any) -> tuple[int, int, int, int]:
    if (not isinstance(value, (list, tuple)) or len(value) != 4
            or any(type(n) is not int or n <= 0 for n in value)
            or value[0] != 1):
        raise ValueError("expected positive integer batch-one NCHW shape")
    return tuple(value)


def _elements(shape: tuple[int, int, int, int]) -> int:
    return shape[1] * shape[2] * shape[3]


def _budget(value: Any, name: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer; invalid budgets fail closed")
    return value


def _conv_candidate(op: dict[str, Any], spatial: int) -> dict[str, Any]:
    inp, out = _shape(op["input_shapes"][0]), _shape(op["output_shape"])
    kernel, stride, padding = op["kernel"], op["stride"], op["padding"]
    if (type(kernel) is not int or kernel not in (1, 3)
            or type(stride) is not int or stride not in (1, 2)
            or type(padding) is not int or padding < 0
            or op.get("groups") != 1 or len(op["input_shapes"]) != 1):
        raise ValueError("only the frozen dense 1x1/3x3 convolution contract is supported")
    expected_h = (inp[2] + 2 * padding - kernel) // stride + 1
    expected_w = (inp[3] + 2 * padding - kernel) // stride + 1
    if (out[2], out[3]) != (expected_h, expected_w):
        raise ValueError("convolution output shape disagrees with kernel/stride/padding")
    th, tw = min(spatial, out[2]), min(spatial, out[3])
    ci, co = min(inp[1], 16), min(out[1], 32)
    ih, iw = (th - 1) * stride + kernel, (tw - 1) * stride + kernel
    activation = {
        "input_pingpong_bytes": 2 * ih * iw * ci,
        "output_pingpong_bytes": 2 * th * tw * co,
        "accumulator_int32_bytes": 4 * th * tw * co,
        "line_buffer_int8_bytes": (kernel - 1) * iw * ci,
        "metadata_reserve_bytes": METADATA_RESERVE_BYTES,
    }
    weights = {
        "weights_pingpong_int8_bytes": 2 * kernel * kernel * ci * co,
        "bias_block_int32_bytes": 4 * co,
    }
    return {
        "requested_spatial_block": spatial,
        "output_tile_shape_chw": [co, th, tw],
        "input_tile_shape_chw_including_halo_and_zero_padding": [ci, ih, iw],
        "input_channel_block": ci, "output_channel_block": co,
        "input_channel_blocks": _ceildiv(inp[1], ci),
        "output_channel_blocks": _ceildiv(out[1], co),
        "last_input_channel_block": inp[1] % ci or ci,
        "last_output_channel_block": out[1] % co or co,
        "spatial_tiles": _ceildiv(out[2], th) * _ceildiv(out[3], tw),
        "last_spatial_tile_hw": [out[2] % th or th, out[3] % tw or tw],
        "activation_components": activation,
        "weight_components": weights,
        "resident_activation_bytes": sum(activation.values()),
        "tile_weight_and_bias_bytes": sum(weights.values()),
        "padding_policy": "input origin=(output_origin*stride-padding); out-of-image slots allocated and filled with integer zero",
        "accumulator_policy": "one output-channel/spatial tile remains resident across every input-channel tile; bias added once",
        "loop_order": "output spatial tile -> output channel block -> input channel blocks; final requantized output committed to DDR",
    }


def _nonconv_candidate(op: dict[str, Any]) -> dict[str, Any]:
    out = _shape(op["output_shape"])
    inputs = [_shape(value) for value in op["input_shapes"]]
    kind = op["kind"]
    th, tw = min(NONCONV_SPATIAL_BLOCK, out[2]), min(NONCONV_SPATIAL_BLOCK, out[3])
    channels = min(out[1], 32)
    if kind == "add":
        if len(inputs) != 2 or any(shape != out for shape in inputs):
            raise ValueError("add requires two shape-aligned operands")
        footprints = [[channels, th, tw], [channels, th, tw]]
    elif kind == "relu":
        if len(inputs) != 1 or inputs[0] != out:
            raise ValueError("ReLU must preserve one input shape")
        footprints = [[channels, th, tw]]
    elif kind == "nearest2x":
        if (len(inputs) != 1 or inputs[0][1] != out[1]
                or out[2] != 2 * inputs[0][2] or out[3] != 2 * inputs[0][3]):
            raise ValueError("nearest2x requires a channel-preserving exact 2x shape")
        # Include worst odd-phase output origin, even though aligned 32 tiles
        # usually need only half as many input positions; never underallocate.
        sh = min(inputs[0][2], _ceildiv(th + 1, 2))
        sw = min(inputs[0][3], _ceildiv(tw + 1, 2))
        footprints = [[channels, sh, sw]]
    else:
        raise ValueError(f"unsupported operation: {kind}")
    per_operand = [2 * shape[0] * shape[1] * shape[2] for shape in footprints]
    activation = {
        "input_operands_pingpong_bytes": sum(per_operand),
        "output_pingpong_bytes": 2 * th * tw * channels,
        "metadata_reserve_bytes": METADATA_RESERVE_BYTES,
    }
    return {
        "requested_spatial_block": NONCONV_SPATIAL_BLOCK,
        "output_tile_shape_chw": [channels, th, tw],
        "input_operand_tile_shapes_chw": footprints,
        "input_operand_pingpong_bytes": per_operand,
        "output_channel_block": channels,
        "output_channel_blocks": _ceildiv(out[1], channels),
        "last_output_channel_block": out[1] % channels or channels,
        "spatial_tiles": _ceildiv(out[2], th) * _ceildiv(out[3], tw),
        "last_spatial_tile_hw": [out[2] % th or th, out[3] % tw or tw],
        "activation_components": activation, "weight_components": {},
        "resident_activation_bytes": sum(activation.values()),
        "tile_weight_and_bias_bytes": 0,
        "inplace_assumed": False,
        "scale_alignment": "required for add; integer requantization implementation and extra workspace not verified",
        "source_footprint_policy": "nearest2x includes conservative worst output-origin phase; other ops read full corresponding tile per operand",
        "loop_order": "output spatial tile -> channel block; every logical operand has independent input ping-pong storage",
    }


def plan_graph(estimate_result: dict[str, Any],
               activation_budget_bytes: int = DEFAULT_ACTIVATION_BUDGET_BYTES,
               weight_budget_bytes: int = DEFAULT_WEIGHT_BUDGET_BYTES) -> dict[str, Any]:
    """Screen one explicit DDR-backed schedule; no deployed feasibility claim.

    Invalid (including negative) budgets raise ValueError. Valid but inadequate
    budgets return bytes_capacity_pass=False; no unverified gate is promoted.
    Each convolution chooses the largest fitting requested spatial block from
    32,16,8. Non-convolution blocks use fixed32. Boundary blocks may be smaller.
    """
    act_budget = _budget(activation_budget_bytes, "activation_budget_bytes")
    wgt_budget = _budget(weight_budget_bytes, "weight_budget_bytes")
    if not isinstance(estimate_result, dict):
        raise ValueError("estimate_result must be a resource estimator dictionary")
    operations = estimate_result.get("operations")
    if not isinstance(operations, list) or not operations:
        raise ValueError("estimate_result must contain a nonempty operations list")
    if (estimate_result.get("activation_storage_bits") != 8
            or estimate_result.get("raw_output_storage_bits") != 8):
        raise ValueError("this screen requires declared INT8 feature/raw-output storage")
    required = ("peak_full_frame_live_activation_bytes_without_accumulator",
                "weights_and_biases_total_storage_bytes", "retained_raw_output_bytes")
    for key in required:
        if type(estimate_result.get(key)) is not int or estimate_result[key] < 0:
            raise ValueError(f"missing/invalid authoritative external storage field: {key}")
    plans: list[dict[str, Any]] = []
    known = {"input": _shape(estimate_result["tensor_shapes"]["input"])}
    activation_traffic_lb = 0
    parameter_traffic_lb = 0
    for op in operations:
        if not isinstance(op, dict) or op.get("name") in known:
            raise ValueError("invalid or repeated graph operation")
        sources = op.get("inputs", [])
        shapes = op.get("input_shapes", [])
        if len(sources) != len(shapes) or not sources:
            raise ValueError("operation source/shape cardinality mismatch")
        for source, shape in zip(sources, shapes):
            if source not in known or known[source] != _shape(shape):
                raise ValueError("operation refers to missing or shape-inconsistent producer")
        out_shape = _shape(op["output_shape"])
        if op.get("output") != op["name"]:
            raise ValueError("canonical producer/output identity mismatch")
        known[op["name"]] = out_shape
        if op["kind"] == "conv2d":
            candidates = [_conv_candidate(op, size) for size in CONV_SPATIAL_CANDIDATES]
        else:
            candidates = [_nonconv_candidate(op)]
        for candidate in candidates:
            candidate["bytes_capacity_pass"] = (
                candidate["resident_activation_bytes"] <= act_budget
                and candidate["tile_weight_and_bias_bytes"] <= wgt_budget)
        selected = next((candidate for candidate in candidates if candidate["bytes_capacity_pass"]), None)
        # Keep the smallest attempted footprint visible on failure; do not
        # report an artificial zero-byte plan when no legal candidate fits.
        representative = selected if selected is not None else candidates[-1]
        op_act_lb = sum(_elements(_shape(shape)) for shape in shapes) + _elements(out_shape)
        op_param_lb = int(op.get("weight_storage_bytes_int8", 0)) + int(op.get("bias_storage_bytes_int32", 0))
        activation_traffic_lb += op_act_lb
        parameter_traffic_lb += op_param_lb
        plans.append({
            "name": op["name"], "kind": op["kind"], "inputs": list(sources),
            "output": op["output"], "output_shape": list(out_shape),
            **representative,
            "choice_status": "largest_valid_candidate" if selected is not None else "no_candidate_fits",
            "selected_spatial_block": selected["requested_spatial_block"] if selected else None,
            "candidate_screens": candidates,
            "activation_budget_bytes_design": act_budget,
            "weight_budget_bytes_design": wgt_budget,
            "materialized_ddr_activation_transfer_lower_bound_bytes": op_act_lb,
            "materialized_ddr_parameter_transfer_lower_bound_bytes": op_param_lb,
        })
    capacity_pass = all(plan["bytes_capacity_pass"] for plan in plans)
    external_activations = estimate_result[required[0]]
    external_parameters = estimate_result[required[1]]
    return {
        "schema_version": 1,
        "kind": "DDR_backed_layer_by_layer_resident_byte_capacity_screen",
        "configuration": estimate_result.get("configuration"),
        "operation_plans": plans,
        "operation_count": len(plans),
        "conv_count": sum(plan["kind"] == "conv2d" for plan in plans),
        "activation_budget_bytes_design": act_budget,
        "weight_budget_bytes_design": wgt_budget,
        "max_resident_activation_bytes": max(plan["resident_activation_bytes"] for plan in plans),
        "max_tile_weight_and_bias_bytes": max(plan["tile_weight_and_bias_bytes"] for plan in plans),
        "bytes_capacity_pass": capacity_pass,
        "failed_operations": [plan["name"] for plan in plans if not plan["bytes_capacity_pass"]],
        "all_implementation_gates_pass": False,
        "external_ddr_peak_activation_bytes": external_activations,
        "external_ddr_all_weights_biases_bytes": external_parameters,
        "external_ddr_peak_activation_plus_all_parameters_bytes": external_activations + external_parameters,
        "external_ddr_retained_raw_output_bytes": estimate_result[required[2]],
        "external_ddr_capacity_available_bytes": None,
        "external_ddr_capacity_verified": False,
        "materialized_ddr_activation_transfer_lower_bound_bytes": activation_traffic_lb,
        "materialized_ddr_parameter_transfer_lower_bound_bytes": parameter_traffic_lb,
        "materialized_ddr_transfer_lower_bound_bytes": activation_traffic_lb + parameter_traffic_lb,
        "traffic_bound_scope": "one full read per logical operand and one full output write per materialized op, plus parameter read once per conv; omits tile halo/reloads/burst padding/cache effects and is not measured bandwidth or latency",
        "external_storage_policy": "all complete input/intermediate/skip/raw-output tensors materialized in DDR; peak reuses canonical last-consumer lifetimes; all compact INT8 weights plus INT32 biases resident in external DDR separately",
        "resident_storage_policy": "INT8 input/output ping-pong; separate INT32 tile accumulator; double INT8 weight tile and single INT32 bias block in separate weight budget; conservative line buffer and4KiB metadata",
        "requantization_workspace": "not independently implemented;4KiB metadata is a design reservation, not proof of scale/shift workspace sufficiency",
        "remaining_gates": ["integer_operator_and_tile_execution", "line_window_correctness",
                            "BRAM_banks_ports_fragmentation", "DMA_layout_padding", "DDR_capacity_and_effective_bandwidth",
                            "FIFO_backpressure_and_schedule", "requantization_workspace_and_rounding",
                            "whole_graph_simulation", "synthesis_route_timing_CDC", "authorized_board_measurement"],
        "board_latency_or_acceptance": None,
    }
