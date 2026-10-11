"""Pinned full-existing-split NEU loader with mandatory, non-official data policies.

This module only reads originals and transforms copies in memory. It never repairs
the historical split, automatically assigns unassigned files, merges duplicates,
performs augmentation or starts training. XML object/name is authoritative for
classes; directory labels are not substituted. Coordinate/difficult choices are
mandatory protocol inputs and are NOT asserted to be official NEU semantics.

The exact existing letterbox implementation is reused, not its annotation parser
or implicit policies. Targets are CPU tensors with half-open xyxy PIXEL coordinates.
"""
from __future__ import annotations

import hashlib
import importlib.util
from io import BytesIO
import json
from pathlib import Path
import re
import xml.etree.ElementTree as ET

from PIL import Image
import torch
from torch.utils.data import Dataset


_HELPER_PATH = Path(__file__).resolve().parents[1] / 'spectra_rewrite_training/data.py'
_SPEC = importlib.util.spec_from_file_location('_spectra_exp_letterbox_helpers', _HELPER_PATH)
_HELPERS = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_HELPERS)
_LETTERBOX_RGB = _HELPERS.letterbox_rgb
_TRANSFORM_BOXES = _HELPERS.transform_boxes

COORDINATE_POLICIES = ('legacy_raw_xyxy', 'voc1_inclusive_assumption')
DIFFICULT_POLICIES = ('legacy_include', 'explicit_ignore')
CLASS_NAMES = ('crazing', 'inclusion', 'patches', 'pitted_surface', 'rolled-in_scale', 'scratches')


class DatasetIntegrityError(ValueError):
    """Explicit protocol, pinned manifest, path or original payload is inconsistent."""


def _digest(value: str, field: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r'[0-9a-fA-F]{64}', value) is None:
        raise DatasetIntegrityError(f'{field} must be an explicit 64-character SHA256')
    return value.lower()


def _positive_int(value, field):
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise DatasetIntegrityError(f'{field} must be a positive integer')
    return value


def _native_integer(text, field):
    if not isinstance(text, str) or re.fullmatch(r'[+-]?\d+', text.strip()) is None:
        raise DatasetIntegrityError(f'Invalid integer XML {field}: {text!r}')
    return int(text)


def _source_relative_path(source_path: str, declared_source_root: Path, kind: str) -> Path:
    """Map audited source paths to explicit raw_root, without trusting an external absolute path."""
    if not isinstance(source_path, str) or not source_path or '\x00' in source_path:
        raise DatasetIntegrityError('Invalid audited source path')
    original = Path(source_path)
    if not original.is_absolute():
        raise DatasetIntegrityError('Audited original source path must be absolute')
    try:
        relative = original.relative_to(declared_source_root)
    except ValueError as exc:
        raise DatasetIntegrityError('Audited payload path is outside declared source root') from exc
    if any(part in ('', '.', '..') or ':' in part for part in relative.parts):
        raise DatasetIntegrityError('Traversal/alternate-stream path rejected')
    expected_parent = 'images' if kind == 'image' else 'annotations'
    if not relative.parts or relative.parts[0].lower() != expected_parent:
        raise DatasetIntegrityError(f'{kind} must be in the expected original source directory')
    return relative


def _payload_record(sample: dict, kind: str, declared_root: Path, raw_root: Path) -> dict:
    nested = sample.get(kind)
    if not isinstance(nested, dict):
        raise DatasetIntegrityError(f'Missing audited {kind} record')
    source_key = 'source_image' if kind == 'image' else 'source_annotation'
    sha_key = source_key + '_sha256'
    source_path = sample.get(source_key)
    if nested.get('path') != source_path:
        raise DatasetIntegrityError(f'Conflicting flat/nested {kind} source paths')
    expected_sha = _digest(nested.get('sha256'), kind + ' SHA256')
    if _digest(sample.get(sha_key), sha_key) != expected_sha:
        raise DatasetIntegrityError(f'Conflicting flat/nested {kind} SHA256')
    length = nested.get('bytes')
    if isinstance(length, bool) or not isinstance(length, int) or length <= 0:
        raise DatasetIntegrityError(f'Invalid audited {kind} byte count')
    relative = _source_relative_path(source_path, declared_root, kind)
    if relative.stem != sample['sample_id']:
        raise DatasetIntegrityError('Source payload stem differs from sample_id')
    if kind == 'annotation' and relative.suffix.lower() != '.xml':
        raise DatasetIntegrityError('Original NEU annotation must be XML')
    path = (raw_root / relative).resolve(strict=True)
    if not path.is_relative_to(raw_root) or not path.is_file():
        raise DatasetIntegrityError('Resolved payload escapes explicit raw_root')
    return {'path': path, 'relative_path': relative.as_posix(), 'audited_source_path': source_path,
            'sha256': expected_sha, 'bytes': length}


def _verified_bytes(payload: dict, raw_root: Path) -> bytes:
    # Resolve again for each use to detect symlink replacement after construction.
    path = (raw_root / payload['relative_path']).resolve(strict=True)
    if not path.is_relative_to(raw_root) or not path.is_file():
        raise DatasetIntegrityError('Payload path escaped raw_root after construction')
    if path.stat().st_size != payload['bytes']:
        raise DatasetIntegrityError(f'Payload byte-count mismatch: {path}')
    content = path.read_bytes()
    if len(content) != payload['bytes'] or hashlib.sha256(content).hexdigest() != payload['sha256']:
        raise DatasetIntegrityError(f'Payload SHA256 mismatch: {path}')
    return content


def _parse_original_xml(content: bytes, sample_id: str, image_wh: tuple[int, int], class_names) -> list[dict]:
    """Return native numbers and flags; no coordinate or difficult policy is implicit."""
    if b'<!DOCTYPE' in content.upper() or b'<!ENTITY' in content.upper():
        raise DatasetIntegrityError('XML DTD/entity declarations are not allowed')
    try:
        root = ET.fromstring(content)
    except ET.ParseError as exc:
        raise DatasetIntegrityError('Malformed NEU XML') from exc
    if root.tag != 'annotation':
        raise DatasetIntegrityError('Expected annotation XML root')
    filename = root.findtext('filename')
    if not filename or '/' in filename or '\\' in filename or Path(filename).stem != sample_id:
        raise DatasetIntegrityError('XML filename must match the sample stem; absent .jpg suffix is supported')
    xml_wh = (_native_integer(root.findtext('size/width'), 'width'),
              _native_integer(root.findtext('size/height'), 'height'))
    if xml_wh != image_wh:
        raise DatasetIntegrityError('XML size differs from decoded original image')
    records = []
    for index, obj in enumerate(root.findall('object')):
        name = obj.findtext('name')
        if name not in class_names:
            raise DatasetIntegrityError(f'Unknown XML object category {name!r}')
        raw = [_native_integer(obj.findtext(f'bndbox/{key}'), key) for key in ('xmin', 'ymin', 'xmax', 'ymax')]
        difficult = _native_integer(obj.findtext('difficult'), 'difficult')
        truncated = _native_integer(obj.findtext('truncated', '0'), 'truncated')
        if difficult not in (0, 1) or truncated not in (0, 1):
            raise DatasetIntegrityError('Unsupported difficult/truncated value')
        records.append({'object_index': index, 'class_name': name, 'class_id': class_names.index(name),
                        'raw_xyxy': raw, 'difficult': difficult, 'truncated': truncated})
    return records


def _convert_box(raw: list[int], image_wh: tuple[int, int], coordinate_policy: str) -> list[int]:
    x1, y1, x2, y2 = raw
    if coordinate_policy == 'legacy_raw_xyxy':
        box = [x1, y1, x2, y2]
    else:
        box = [x1 - 1, y1 - 1, x2, y2]
    width, height = image_wh
    if not (0 <= box[0] < box[2] <= width and 0 <= box[1] < box[3] <= height):
        raise DatasetIntegrityError(f'Invalid/out-of-range box under explicit {coordinate_policy}: {raw}')
    return box


def _tensor_boxes(values) -> torch.Tensor:
    return torch.tensor(values, dtype=torch.float32, device='cpu').reshape(-1, 4)


def inverse_transform_boxes(boxes, transform):
    """Map network-input pixel boxes back to the selected original-image convention."""
    return _TRANSFORM_BOXES(boxes, transform, inverse=True)


class FullNeuDataset(Dataset):
    """Load ALL audited rows of an existing train or val split; never auto-assign.

    Mandatory policy/pin arguments deliberately have no defaults. A constructor
    validates the pinned audit and safe source mapping. Every __getitem__ re-reads
    original JPG/XML bytes, verifies sizes/SHA256 and re-parses XML before policy
    conversion. Audited raw boxes are not used as substitutes for source labels.
    """
    def __init__(self, manifest_path, raw_root, split, image_size=640, *,
                 coordinate_policy, difficult_policy, expected_manifest_sha256):
        if coordinate_policy not in COORDINATE_POLICIES:
            raise DatasetIntegrityError(f'Unknown explicit coordinate_policy: {coordinate_policy!r}')
        if difficult_policy not in DIFFICULT_POLICIES:
            raise DatasetIntegrityError(f'Unknown explicit difficult_policy: {difficult_policy!r}')
        if split not in ('train', 'val'):
            raise DatasetIntegrityError('Only existing train/val splits are loadable; unassigned is never auto-included')
        expected_manifest_sha256 = _digest(expected_manifest_sha256, 'expected_manifest_sha256')
        self.manifest_path = Path(manifest_path).resolve(strict=True)
        self.raw_root = Path(raw_root).resolve(strict=True)
        if not self.raw_root.is_dir():
            raise DatasetIntegrityError('Explicit raw_root must be an existing directory')
        content = self.manifest_path.read_bytes()
        self.manifest_sha256 = hashlib.sha256(content).hexdigest()
        if self.manifest_sha256 != expected_manifest_sha256:
            raise DatasetIntegrityError('Pinned audit manifest SHA256 mismatch')
        try:
            manifest = json.loads(content)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise DatasetIntegrityError('Malformed pinned audit manifest') from exc
        declared_root = Path(manifest['provenance']['source_root'])
        if not declared_root.is_absolute():
            raise DatasetIntegrityError('Audit source_root must be absolute')
        self.class_names = tuple(manifest['provenance']['class_names_from_existing_yaml'])
        if self.class_names != CLASS_NAMES:
            raise DatasetIntegrityError('Unexpected NEU class mapping in pinned audit')
        self.coordinate_policy, self.difficult_policy, self.split = coordinate_policy, difficult_policy, split
        self.image_size = image_size
        _HELPERS.letterbox_metadata((1, 1), image_size)
        self._samples = []
        self.sample_ids = []
        seen = set()
        for sample in manifest.get('samples', []):
            identity = sample.get('sample_id')
            if not isinstance(identity, str) or not identity or identity in seen:
                raise DatasetIntegrityError('Missing or duplicate sample_id in audit; pixel duplicates remain allowed')
            seen.add(identity)
            source_split = sample.get('existing_split')
            if source_split not in ('train', 'val', 'unassigned') or sample.get('source_split') != source_split:
                raise DatasetIntegrityError('Unknown, overlapping or inconsistent historical split membership')
            memberships = sample.get('existing_split_memberships')
            expected_membership = [] if source_split == 'unassigned' else [source_split]
            if memberships != expected_membership:
                raise DatasetIntegrityError('Split membership list conflicts with explicit audit split')
            if source_split != split:
                continue
            image = _payload_record(sample, 'image', declared_root, self.raw_root)
            annotation = _payload_record(sample, 'annotation', declared_root, self.raw_root)
            expected_wh = (_positive_int(sample['image'].get('width'), 'audited image width'),
                           _positive_int(sample['image'].get('height'), 'audited image height'))
            if list(expected_wh) != sample.get('decoded_size'):
                raise DatasetIntegrityError('Audit dimension aliases disagree')
            pixel_sha = _digest(sample['image'].get('decoded_rgb_sha256'), 'audited decoded RGB SHA256')
            self._samples.append({'sample_id': identity, 'source_class_directory': sample.get('source_class_directory'),
                                  'image': image, 'annotation': annotation, 'expected_wh': expected_wh,
                                  'decoded_rgb_sha256': pixel_sha, 'existing_split': source_split})
            self.sample_ids.append(identity)
        if not self._samples:
            raise DatasetIntegrityError(f'Pinned audit has no samples in existing {split} split')
        self.letterbox_source_sha256 = hashlib.sha256(_HELPER_PATH.read_bytes()).hexdigest()

    def __len__(self):
        return len(self._samples)

    def __getitem__(self, index):
        record = self._samples[index]
        image_content = _verified_bytes(record['image'], self.raw_root)
        annotation_content = _verified_bytes(record['annotation'], self.raw_root)
        try:
            with Image.open(BytesIO(image_content)) as original:
                original.load()
                image = original.convert('RGB')
        except (OSError, ValueError) as exc:
            raise DatasetIntegrityError('Verified image failed to decode') from exc
        if image.size != record['expected_wh']:
            raise DatasetIntegrityError('Decoded dimensions differ from pinned audit')
        pixel_sha = hashlib.sha256(f'RGB:{image.width}x{image.height}:'.encode() + image.tobytes()).hexdigest()
        if pixel_sha != record['decoded_rgb_sha256']:
            raise DatasetIntegrityError('Decoded pixel hash differs from pinned audit')
        raw_objects = _parse_original_xml(annotation_content, record['sample_id'], image.size, self.class_names)
        boxes, labels, ignore_boxes, ignore_labels, ignore_metadata = [], [], [], [], []
        for obj in raw_objects:
            box = _convert_box(obj['raw_xyxy'], image.size, self.coordinate_policy)
            if self.difficult_policy == 'explicit_ignore' and obj['difficult']:
                ignore_boxes.append(box)
                ignore_labels.append(obj['class_id'])
                ignore_metadata.append(dict(obj, reason='explicit_difficult_policy', original_xyxy=box))
            else:
                boxes.append(box)
                labels.append(obj['class_id'])
        image_tensor, transform = _LETTERBOX_RGB(image, self.image_size)
        original_boxes, original_ignore = _tensor_boxes(boxes), _tensor_boxes(ignore_boxes)
        original_labels = torch.tensor(labels, dtype=torch.int64)
        original_ignore_labels = torch.tensor(ignore_labels, dtype=torch.int64)
        resized_height, resized_width = transform['resized_hw']
        left, top = transform['pad_left'], transform['pad_top']
        provenance = {'manifest_path': str(self.manifest_path), 'manifest_sha256': self.manifest_sha256,
                      'raw_root_explicit': str(self.raw_root), 'existing_split': self.split,
                      'image_path': str(record['image']['path']), 'image_sha256': record['image']['sha256'],
                      'image_bytes': record['image']['bytes'], 'annotation_path': str(record['annotation']['path']),
                      'annotation_sha256': record['annotation']['sha256'], 'annotation_bytes': record['annotation']['bytes'],
                      'decoded_rgb_sha256': pixel_sha, 'payload_bytes_and_sha256_reverified': True,
                      'annotation_source': 'reparsed verified original XML bytes, not audit cached boxes',
                      'coordinate_policy': self.coordinate_policy, 'difficult_policy': self.difficult_policy,
                      'policy_status': 'explicit experiment choice/assumption; not a verified official NEU convention',
                      'source_class_directory': record['source_class_directory'],
                      'class_source': 'XML object/name, never directory class',
                      'duplicate_policy': 'preserve every historical split row; no merge/delete/reassignment',
                      'letterbox_source': str(_HELPER_PATH), 'letterbox_source_sha256': self.letterbox_source_sha256,
                      'augmentation': 'none'}
        return {'image': image_tensor, 'boxes': _TRANSFORM_BOXES(original_boxes, transform),
                'labels': original_labels.clone(), 'ignore_boxes': _TRANSFORM_BOXES(original_ignore, transform),
                'ignore_labels': original_ignore_labels.clone(),
                'valid_region': torch.tensor([left, top, left + resized_width, top + resized_height], dtype=torch.float32),
                'original_boxes': original_boxes, 'original_labels': original_labels,
                'original_ignore_boxes': original_ignore, 'original_ignore_labels': original_ignore_labels,
                'image_hw': transform['output_hw'], 'original_hw': transform['original_hw'],
                'transform': transform, 'sample_id': record['sample_id'], 'source_group': pixel_sha,
                'raw_objects': raw_objects, 'ignore_metadata': ignore_metadata,
                'has_ignore_regions': bool(ignore_boxes), 'provenance': provenance}


def collate_fn(batch):
    """Return (stacked_images, targets_list, metadata_list); no device move or policy changes."""
    if not batch:
        raise DatasetIntegrityError('Cannot collate an empty batch')
    shapes = {tuple(item['image'].shape) for item in batch}
    if len(shapes) != 1:
        raise DatasetIntegrityError('All collated images require the same explicit letterbox shape')
    target_keys = ('boxes', 'labels', 'ignore_boxes', 'ignore_labels', 'valid_region', 'image_hw')
    metadata_keys = ('sample_id', 'source_group', 'original_boxes', 'original_labels',
                     'original_ignore_boxes', 'original_ignore_labels', 'original_hw', 'image_hw',
                     'transform', 'provenance', 'raw_objects', 'ignore_metadata', 'has_ignore_regions')
    return (torch.stack([item['image'] for item in batch]),
            [{key: item[key] for key in target_keys} for item in batch],
            [{key: item[key] for key in metadata_keys} for item in batch])
