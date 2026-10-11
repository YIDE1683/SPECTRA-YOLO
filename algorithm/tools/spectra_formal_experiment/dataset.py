"""Read-only adapter for the preregistered NEU main split CSV.

The adapter consumes the CSV contract only; it does not import or trust the
split generator.  Raw JPG/XML bytes are reverified at every access.
"""
from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass
from io import BytesIO, StringIO
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset

from tools.spectra_experiment.dataset import (
    CLASS_NAMES, DatasetIntegrityError, _LETTERBOX_RGB, _TRANSFORM_BOXES,
    _convert_box, _parse_original_xml, _payload_record, _positive_int,
    _tensor_boxes, _verified_bytes,
)

FROZEN_MANIFEST_SHA256 = "9d5c3cf582990e543bd2d3668a02ebd39f5b6b3ba42aa7d87fee93bf602a73c7"
FROZEN_SPLIT_SHA256 = "f1b5c0fefbce849f91f9e296bbba1185fafc94438ea544298c5e4495f61adfa9"
ROLES = ("train", "tune", "evaluation")


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


@dataclass(frozen=True)
class FrozenSplitContract:
    """Pinned split contract.  Non-main instances exist only for unit fixtures."""
    name: str
    manifest_sha256: str
    split_sha256: str
    role_counts: tuple[int, int, int]
    coordinate_policy: str = "legacy_raw_xyxy"
    difficult_policy: str = "legacy_include"

    def validate(self) -> None:
        if not self.name or any(type(v) is not int or v <= 0 for v in self.role_counts):
            raise DatasetIntegrityError("Contract name and all three role counts must be positive")
        for value in (self.manifest_sha256, self.split_sha256):
            if len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
                raise DatasetIntegrityError("Contract SHA256 values must be lowercase hex")
        if self.coordinate_policy != "legacy_raw_xyxy" or self.difficult_policy != "legacy_include":
            raise DatasetIntegrityError("Formal v1 supports only legacy_raw_xyxy + legacy_include")

    @classmethod
    def neu_main_v1(cls) -> "FrozenSplitContract":
        """Return the code-pinned formal contract; callers cannot self-assert a split digest."""
        return cls("neu_main_v1", FROZEN_MANIFEST_SHA256, FROZEN_SPLIT_SHA256, (1260, 270, 270))


class FrozenSplitDataset(Dataset):
    """One immutable role view of a complete, mutually exclusive split CSV."""
    def __init__(self, manifest_path, raw_root, split_csv, split_role, *, contract: FrozenSplitContract,
                 image_size: int = 640):
        contract.validate()
        if split_role not in ROLES:
            raise DatasetIntegrityError(f"split_role must be one of {ROLES}")
        if type(image_size) is not int or image_size < 64 or image_size % 32:
            raise DatasetIntegrityError("image_size must be >=64 and divisible by32")
        self.manifest_path = Path(manifest_path).resolve(strict=True)
        self.raw_root = Path(raw_root).resolve(strict=True)
        self.split_path = Path(split_csv).resolve(strict=True)
        if not self.raw_root.is_dir():
            raise DatasetIntegrityError("raw_root must be an existing directory")
        manifest_bytes, split_bytes = self.manifest_path.read_bytes(), self.split_path.read_bytes()
        if _sha(manifest_bytes) != contract.manifest_sha256:
            raise DatasetIntegrityError("Pinned manifest SHA256 mismatch")
        if _sha(split_bytes) != contract.split_sha256:
            raise DatasetIntegrityError("Pinned split CSV SHA256 mismatch")
        try:
            manifest = json.loads(manifest_bytes)
            reader = csv.DictReader(StringIO(split_bytes.decode("utf-8-sig"), newline=""))
            rows = list(reader)
        except (UnicodeDecodeError, json.JSONDecodeError, csv.Error) as exc:
            raise DatasetIntegrityError("Malformed manifest or split CSV") from exc
        if reader.fieldnames != ["sample_id", "source_group", "split_role"]:
            raise DatasetIntegrityError("Split CSV fields/order must be sample_id,source_group,split_role")
        declared_root = Path(manifest["provenance"]["source_root"])
        if not declared_root.is_absolute():
            raise DatasetIntegrityError("Manifest source_root must be absolute")
        self.class_names = tuple(manifest["provenance"]["class_names_from_existing_yaml"])
        if self.class_names != CLASS_NAMES:
            raise DatasetIntegrityError("Unexpected class mapping")
        samples = manifest.get("samples", [])
        by_id = {}
        for sample in samples:
            sid = sample.get("sample_id")
            if not isinstance(sid, str) or not sid or sid in by_id:
                raise DatasetIntegrityError("Manifest sample IDs must be unique nonempty strings")
            by_id[sid] = sample
        seen, group_roles, counts = set(), {}, {r: 0 for r in ROLES}
        for row in rows:
            sid, group, role = row["sample_id"], row["source_group"], row["split_role"]
            if not sid or sid in seen or sid not in by_id or role not in ROLES or not group:
                raise DatasetIntegrityError("Unknown/duplicate/blank split row")
            seen.add(sid); counts[role] += 1
            pixel_sha = by_id[sid]["image"].get("decoded_rgb_sha256")
            if group != pixel_sha:
                raise DatasetIntegrityError("source_group must equal the audited decoded RGB identity")
            if group in group_roles and group_roles[group] != role:
                raise DatasetIntegrityError("A source_group crosses split roles")
            group_roles[group] = role
        if seen != set(by_id) or tuple(counts[r] for r in ROLES) != contract.role_counts:
            raise DatasetIntegrityError("Split must cover the entire manifest with exact frozen role counts")
        self.contract, self.split, self.image_size = contract, split_role, image_size
        self.manifest_sha256, self.split_sha256 = _sha(manifest_bytes), _sha(split_bytes)
        self._samples, self.sample_ids = [], []
        for row in rows:
            if row["split_role"] != split_role:
                continue
            sample = by_id[row["sample_id"]]
            expected_wh = (_positive_int(sample["image"].get("width"), "width"),
                           _positive_int(sample["image"].get("height"), "height"))
            if list(expected_wh) != sample.get("decoded_size"):
                raise DatasetIntegrityError("Audited image dimensions disagree")
            self._samples.append({"sample_id": row["sample_id"], "source_group": row["source_group"],
                "image": _payload_record(sample, "image", declared_root, self.raw_root),
                "annotation": _payload_record(sample, "annotation", declared_root, self.raw_root),
                "expected_wh": expected_wh, "decoded_rgb_sha256": row["source_group"]})
            self.sample_ids.append(row["sample_id"])

    def identity(self) -> dict:
        return {"kind": "frozen_split_dataset", "contract": self.contract.name,
                "manifest_sha256": self.manifest_sha256, "split_csv_sha256": self.split_sha256,
                "split_role": self.split, "role_counts": dict(zip(ROLES, self.contract.role_counts)),
                "coordinate_policy": self.contract.coordinate_policy,
                "difficult_policy": self.contract.difficult_policy, "image_size": self.image_size,
                "class_names": list(self.class_names)}

    def __len__(self): return len(self._samples)

    def __getitem__(self, index):
        record = self._samples[index]
        image_bytes = _verified_bytes(record["image"], self.raw_root)
        xml_bytes = _verified_bytes(record["annotation"], self.raw_root)
        try:
            with Image.open(BytesIO(image_bytes)) as source:
                source.load(); image = source.convert("RGB")
        except (OSError, ValueError) as exc:
            raise DatasetIntegrityError("Verified image failed to decode") from exc
        pixel_sha = _sha(f"RGB:{image.width}x{image.height}:".encode() + image.tobytes())
        if image.size != record["expected_wh"] or pixel_sha != record["decoded_rgb_sha256"]:
            raise DatasetIntegrityError("Decoded pixels/dimensions differ from audit")
        objects = _parse_original_xml(xml_bytes, record["sample_id"], image.size, self.class_names)
        boxes, labels = [], []
        for obj in objects:  # legacy_include deliberately includes difficult objects.
            boxes.append(_convert_box(obj["raw_xyxy"], image.size, "legacy_raw_xyxy"))
            labels.append(obj["class_id"])
        original_boxes, original_labels = _tensor_boxes(boxes), torch.tensor(labels, dtype=torch.int64)
        image_tensor, transform = _LETTERBOX_RGB(image, self.image_size)
        left, top = transform["pad_left"], transform["pad_top"]
        rh, rw = transform["resized_hw"]
        empty_boxes, empty_labels = torch.empty((0, 4), dtype=torch.float32), torch.empty((0,), dtype=torch.int64)
        return {"image": image_tensor, "boxes": _TRANSFORM_BOXES(original_boxes, transform),
            "labels": original_labels.clone(), "ignore_boxes": empty_boxes, "ignore_labels": empty_labels,
            "valid_region": torch.tensor([left, top, left + rw, top + rh], dtype=torch.float32),
            "original_boxes": original_boxes, "original_labels": original_labels,
            "original_ignore_boxes": empty_boxes.clone(), "original_ignore_labels": empty_labels.clone(),
            "image_hw": transform["output_hw"], "original_hw": transform["original_hw"],
            "transform": transform, "sample_id": record["sample_id"], "source_group": record["source_group"],
            "raw_objects": objects, "ignore_metadata": [], "has_ignore_regions": False,
            "provenance": {**self.identity(), "image_sha256": record["image"]["sha256"],
                           "annotation_sha256": record["annotation"]["sha256"],
                           "payload_bytes_and_sha256_reverified": True}}
