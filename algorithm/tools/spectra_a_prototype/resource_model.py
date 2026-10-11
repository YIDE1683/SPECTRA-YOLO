"""Independent, stdlib-only arithmetic and explicit full-frame liveness model.

This is a canonical deployment-graph estimate, not a PyTorch memory profiler,
cycle-accurate simulator, FPGA mapper, or latency benchmark. All persistent
activation buffers, including retained raw head outputs, use a declared INT8
storage contract. A separate conservative full-output INT32 convolution scratch
is reported both included and excluded from the peak. No tiling is implemented.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import json
from typing import Any


ACTIVATION_BUDGET_BYTES = 1_048_576
WEIGHT_TILE_BUDGET_BYTES = 262_144
INPUT_LANES = 16
OUTPUT_LANES = 32


@dataclass(frozen=True)
class Shape:
    channels: int
    height: int
    width: int

    @property
    def elements(self) -> int:
        return self.channels * self.height * self.width

    def nchw(self) -> list[int]:
        return [1, self.channels, self.height, self.width]


def _ceildiv(value: int, divisor: int) -> int:
    return (value + divisor - 1) // divisor


def _validate(
    p2_channels: int,
    neck_channels: int,
    reduced_depth: bool,
    num_classes: int,
    input_size: tuple[int, int],
) -> tuple[int, int]:
    for name, value, options in (
        ("p2_channels", p2_channels, (0, 32, 64)),
        ("neck_channels", neck_channels, (64, 96)),
        ("num_classes", num_classes, (6, 10)),
    ):
        if type(value) is not int or value not in options:
            raise ValueError(f"{name} must be one of {options}")
    if type(reduced_depth) is not bool:
        raise ValueError("reduced_depth must be bool")
    if not isinstance(input_size, (tuple, list)) or len(input_size) != 2:
        raise ValueError("input_size must contain (height, width)")
    if any(type(v) is not int or v <= 0 or v % 32 for v in input_size):
        raise ValueError("input height and width must be positive multiples of 32")
    return input_size[0], input_size[1]


class _Graph:
    def __init__(self, height: int, width: int) -> None:
        self.shapes: dict[str, Shape] = {"input": Shape(3, height, width)}
        self.ops: list[dict[str, Any]] = []

    def _append(self, name: str, kind: str, inputs: list[str], shape: Shape,
                **extra: Any) -> str:
        if name in self.shapes:
            raise ValueError(f"duplicate tensor name: {name}")
        if any(source not in self.shapes for source in inputs):
            raise ValueError("graph edge references an undefined tensor")
        self.shapes[name] = shape
        self.ops.append({
            "name": name, "kind": kind, "inputs": inputs, "output": name,
            "input_shapes": [self.shapes[source].nchw() for source in inputs],
            "output_shape": shape.nchw(), "deploy_conv_parameters": 0,
            "logical_macs": 0, "physical_cycle_proxy": 0,
            "weight_storage_bytes_int8": 0, "bias_storage_bytes_int32": 0,
            **extra,
        })
        return name

    def conv(self, name: str, source: str, cout: int, kernel: int,
             stride: int = 1, padding: int = 0) -> str:
        src = self.shapes[source]
        height = (src.height + 2 * padding - kernel) // stride + 1
        width = (src.width + 2 * padding - kernel) // stride + 1
        if min(height, width) <= 0:
            raise ValueError("convolution has no valid output")
        shape = Shape(cout, height, width)
        weights = cout * src.channels * kernel * kernel
        parameters = weights + cout  # Every BN is folded; deploy bias=True.
        macs = height * width * weights
        cycles = (height * width * _ceildiv(src.channels, INPUT_LANES)
                  * _ceildiv(cout, OUTPUT_LANES) * kernel * kernel)
        return self._append(
            name, "conv2d", [source], shape, kernel=kernel, stride=stride,
            padding=padding, groups=1, bias=True,
            deploy_conv_parameters=parameters, weight_parameters=weights,
            bias_parameters=cout, logical_macs=macs,
            physical_cycle_proxy=cycles, weight_storage_bytes_int8=weights,
            bias_storage_bytes_int32=cout * 4,
            padded_input_channels=_ceildiv(src.channels, INPUT_LANES) * INPUT_LANES,
            padded_output_channels=_ceildiv(cout, OUTPUT_LANES) * OUTPUT_LANES,
            accumulator_scratch_bytes_int32=shape.elements * 4,
        )

    def relu(self, name: str, source: str) -> str:
        return self._append(name, "relu", [source], self.shapes[source])

    def upsample(self, name: str, source: str) -> str:
        src = self.shapes[source]
        return self._append(name, "nearest2x", [source],
                            Shape(src.channels, src.height * 2, src.width * 2))

    def add(self, name: str, lhs: str, rhs: str) -> str:
        if self.shapes[lhs] != self.shapes[rhs]:
            raise ValueError(f"unaligned addition: {lhs}, {rhs}")
        return self._append(name, "add", [lhs, rhs], self.shapes[lhs],
                            scale_alignment="required; requantization not implemented")


def _canonical_graph(p2: int, neck: int, reduced: bool, classes: int,
                     height: int, width: int) -> tuple[_Graph, list[dict[str, Any]]]:
    graph = _Graph(height, width)
    channels = (32, 64, 128, 192, 256)
    reps = (0, 1, 1, 1, 1) if reduced else (0, 1, 2, 2, 1)
    backbone: dict[int, str] = {}
    current = "input"
    for stage, (cout, count) in enumerate(zip(channels, reps), 1):
        prefix = f"backbone.s{stage}"
        current = graph.conv(f"{prefix}.down.conv", current, cout, 3, 2, 1)
        current = graph.relu(f"{prefix}.down.relu", current)
        for index in range(count):
            current = graph.conv(f"{prefix}.rep{index}.conv", current, cout, 3, 1, 1)
            current = graph.relu(f"{prefix}.rep{index}.relu", current)
        backbone[stage] = current

    pyramid: dict[int, str] = {}
    lateral = graph.conv("neck.l5.conv", backbone[5], neck, 1)
    pyramid[5] = graph.relu("neck.p5.relu", lateral)
    for stage in (4, 3):
        lateral = graph.conv(f"neck.l{stage}.conv", backbone[stage], neck, 1)
        top = graph.upsample(f"neck.p{stage}.up", pyramid[stage + 1])
        fusion = graph.add(f"neck.p{stage}.add", lateral, top)
        pyramid[stage] = graph.relu(f"neck.p{stage}.relu", fusion)
    if p2:
        lateral = graph.conv("neck.p2.lateral.conv", backbone[2], p2, 1)
        top = graph.conv("neck.p2.top.conv", pyramid[3], p2, 1)
        top = graph.upsample("neck.p2.up", top)
        fusion = graph.add("neck.p2.add", lateral, top)
        pyramid[2] = graph.relu("neck.p2.relu", fusion)

    outputs: list[dict[str, Any]] = []
    for stage in sorted(pyramid):
        tower_width, tower_depth = (32, 1) if stage == 2 else (64, 2)
        result: dict[str, Any] = {"stride": 2 ** stage}
        for branch, cout in (("class", classes), ("regression", 4)):
            current = pyramid[stage]
            for index in range(tower_depth):
                prefix = f"head.p{stage}.{branch}.tower{index}"
                current = graph.conv(f"{prefix}.conv", current, tower_width, 3, 1, 1)
                current = graph.relu(f"{prefix}.relu", current)
            current = graph.conv(f"head.p{stage}.{branch}.output.conv", current, cout, 1)
            if branch == "regression":
                current = graph.relu(f"head.p{stage}.distances.relu", current)
            result["class_logits" if branch == "class" else "distances"] = current
        outputs.append(result)
    return graph, outputs


def _liveness(graph: _Graph, outputs: list[dict[str, Any]]) -> dict[str, Any]:
    retained = {item[key] for item in outputs for key in ("class_logits", "distances")}
    remaining = Counter(source for op in graph.ops for source in op["inputs"])
    live: dict[str, int] = {"input": graph.shapes["input"].elements}
    schedule: list[dict[str, Any]] = []
    peak_no_acc = sum(live.values())
    peak_with_acc = peak_no_acc
    peak_no_acc_op = "input"
    peak_with_acc_op = "input"
    max_scratch = 0

    for index, op in enumerate(graph.ops):
        if any(source not in live for source in op["inputs"]):
            raise AssertionError("use after release in canonical schedule")
        before = sum(live.values())
        name = op["output"]
        output_bytes = graph.shapes[name].elements
        live[name] = output_bytes  # Out-of-place op: inputs coexist with output.
        during = dict(live)
        no_acc = sum(live.values())
        scratch = op.get("accumulator_scratch_bytes_int32", 0)
        with_acc = no_acc + scratch
        max_scratch = max(max_scratch, scratch)
        if no_acc > peak_no_acc:
            peak_no_acc, peak_no_acc_op = no_acc, op["name"]
        if with_acc > peak_with_acc:
            peak_with_acc, peak_with_acc_op = with_acc, op["name"]
        released: list[str] = []
        for source in op["inputs"]:
            remaining[source] -= 1
            if remaining[source] == 0 and source not in retained and source in live:
                released.append(source)
                del live[source]
        if remaining[name] == 0 and name not in retained:
            released.append(name)
            del live[name]
        schedule.append({
            "step": index, "operation": op["name"], "kind": op["kind"],
            "inputs": list(op["inputs"]), "allocated": name,
            "live_bytes_before": before, "allocated_output_bytes": output_bytes,
            "live_buffers_during": during,
            "live_bytes_without_accumulator": no_acc,
            "accumulator_scratch_bytes_int32": scratch,
            "live_bytes_with_accumulator": with_acc,
            "released_after_operation": released,
            "live_buffers_after": dict(live), "live_bytes_after": sum(live.values()),
        })
    if set(live) != retained:
        raise AssertionError("schedule did not release all non-returned tensors")
    if any(value != 0 for value in remaining.values()):
        raise AssertionError("graph contains unscheduled consumers")
    return {
        "schedule": schedule,
        "peak_full_frame_live_activation_bytes": peak_no_acc,
        "peak_full_frame_live_activation_bytes_without_accumulator": peak_no_acc,
        "peak_full_frame_live_activation_bytes_with_accumulator": peak_with_acc,
        "peak_without_accumulator_operation": peak_no_acc_op,
        "peak_with_accumulator_operation": peak_with_acc_op,
        "maximum_single_full_frame_accumulator_bytes": max_scratch,
        "retained_raw_output_buffers": dict(sorted(live.items())),
        "retained_raw_output_bytes": sum(live.values()),
        "activation_budget_bytes_design": ACTIVATION_BUDGET_BYTES,
        "full_frame_budget_pass_without_accumulator": peak_no_acc <= ACTIVATION_BUDGET_BYTES,
        "full_frame_budget_pass_with_accumulator": peak_with_acc <= ACTIVATION_BUDGET_BYTES,
        "full_frame_budget_pass": peak_with_acc <= ACTIVATION_BUDGET_BYTES,
    }


def estimate(p2_channels: int = 0, neck_channels: int = 64,
             reduced_depth: bool = False, num_classes: int = 6,
             input_size: tuple[int, int] = (640, 640)) -> dict[str, Any]:
    """Estimate the frozen graph; returns JSON-serializable builtin types only.

    logical_macs excludes bias, ReLU, addition, resize and PS decoding. Cycle
    proxy includes padded channel lanes but excludes non-convolution execution.
    full_frame_budget_pass is deliberately conservative, using the peak with
    full-output INT32 scratch; it is NOT tiled feasibility or board acceptance.
    """
    height, width = _validate(p2_channels, neck_channels, reduced_depth,
                             num_classes, input_size)
    graph, outputs = _canonical_graph(p2_channels, neck_channels, reduced_depth,
                                     num_classes, height, width)
    layers = [dict(op) for op in graph.ops if op["kind"] == "conv2d"]
    memory = _liveness(graph, outputs)
    weights = sum(layer["weight_storage_bytes_int8"] for layer in layers)
    biases = sum(layer["bias_storage_bytes_int32"] for layer in layers)
    return {
        "schema_version": 1,
        "kind": "independent_canonical_deploy_graph_arithmetic_and_naive_full_frame_liveness",
        "configuration": {"p2_channels": p2_channels, "neck_channels": neck_channels,
                          "reduced_depth": reduced_depth, "num_classes": num_classes,
                          "input_size": [height, width]},
        "deploy_conv_parameters": sum(layer["deploy_conv_parameters"] for layer in layers),
        "logical_macs": sum(layer["logical_macs"] for layer in layers),
        "physical_cycle_proxy": sum(layer["physical_cycle_proxy"] for layer in layers),
        "conv_count": len(layers), "layers": layers,
        "operations": graph.ops,
        "edges": [{"from": source, "to": op["name"]}
                  for op in graph.ops for source in op["inputs"]],
        "tensor_shapes": {name: shape.nchw() for name, shape in graph.shapes.items()},
        "raw_outputs": outputs,
        "activation_storage_bits": 8,
        "raw_output_storage_bits": 8,
        "accumulator_scratch_bits": 32,
        "weights_total_storage_bytes_int8": weights,
        "biases_total_storage_bytes_int32": biases,
        "weights_and_biases_total_storage_bytes": weights + biases,
        "largest_complete_conv_weight_and_bias_bytes": max(
            layer["weight_storage_bytes_int8"] + layer["bias_storage_bytes_int32"]
            for layer in layers),
        "weight_tile_budget_bytes_design": WEIGHT_TILE_BUDGET_BYTES,
        "weight_tile_fit_verified": False,
        "tile_status": "not implemented; full-frame failure does not reject all tiled schedules",
        "line_buffer_allocation": "none in naive full-frame graph; tiled line/halo/FIFO allocation unimplemented",
        "nonconv_execution_in_cycle_proxy": False,
        "precision_speedup_assumed": False,
        "board_latency_or_acceptance": None,
        "schedule_policy": "backbone; top-down neck; optional P2; ascending-stride heads; class then regression; all ops out-of-place; return outputs retained",
        **memory,
    }


if __name__ == "__main__":
    # Compact diagnostic only; root orchestration owns durable audit artifacts.
    rows = []
    for p2 in (0, 32, 64):
        for neck in (64, 96):
            for reduced in (False, True):
                for classes in (6, 10):
                    result = estimate(p2, neck, reduced, classes)
                    rows.append({key: result[key] for key in (
                        "configuration", "deploy_conv_parameters", "logical_macs",
                        "physical_cycle_proxy", "conv_count",
                        "peak_full_frame_live_activation_bytes_without_accumulator",
                        "peak_full_frame_live_activation_bytes_with_accumulator",
                        "full_frame_budget_pass")})
    print(json.dumps(rows, indent=2))
