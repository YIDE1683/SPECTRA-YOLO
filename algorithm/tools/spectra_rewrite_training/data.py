"""Strict CPU loader for DEVELOPMENT-ONLY original samples, not a train/test split.

No model, inference, training, calibration, optimizer step or prediction writing is
performed here. Images/annotations are read as verified bytes and never modified.
The current integration contract is shape/coordinate/gradient smoke checking only:
samples with ignore regions must have loss=None and a recorded reason until a
reviewed ignore-aware training strategy exists. The complement of ignore regions
MUST NOT automatically become negative/background training labels.

image: CPU float32 RGB CHW in [0,1]. boxes/ignore_boxes: CPU float32 Nx4,
0-based HALF-OPEN xyxy PIXELS in the letterboxed image. labels/classes: CPU
int64 dataset-local IDs (NEU 0..5; VisDrone 0..9); namespaces are NOT merged.
Pass image_hw, NOT original_hw, to losses expecting network-input coordinates.
Actual resize scale_x/scale_y and integer padding are retained; nominal aspect
ratio alone is insufficient after pixel rounding. ignore_mask and
background_valid_mask describe geometry only, never validated negative labels.
"""
from __future__ import annotations

from copy import deepcopy
from fractions import Fraction
import hashlib
from io import BytesIO
import json
import math
from pathlib import Path, PurePosixPath, PureWindowsPath
import re
from typing import Any
import xml.etree.ElementTree as ET

from PIL import Image
import torch
from torch.utils.data import Dataset


NEU_CLASS_NAMES = ('crazing', 'inclusion', 'patches', 'pitted_surface', 'rolled-in_scale', 'scratches')
VISDRONE_CLASS_NAMES = ('pedestrian', 'people', 'bicycle', 'car', 'van', 'truck',
                        'tricycle', 'awning-tricycle', 'bus', 'motor')
CLASS_NAMES = {'neu': NEU_CLASS_NAMES, 'visdrone': VISDRONE_CLASS_NAMES}
DEVELOPMENT_ONLY = 'development-exposed design/smoke samples; NOT a training split or formal test set'


class DataIntegrityError(ValueError):
    """A manifest, verified payload, annotation or transform violates its contract."""


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise DataIntegrityError(f'{name} must be a positive integer')
    return value


def _dataset_name(name: str) -> str:
    if not isinstance(name, str) or name.lower() not in CLASS_NAMES:
        raise DataIntegrityError(f'Unsupported dataset: {name!r}')
    return name.lower()


def safe_payload_path(root: Path | str, relative_path: str) -> Path:
    """Reject traversal, absolute/drive paths, alternate streams and symlink escapes."""
    root = Path(root).resolve(strict=True)
    if not isinstance(relative_path, str) or not relative_path or '\x00' in relative_path:
        raise DataIntegrityError('Payload relative_path must be a nonempty safe string')
    normalized = relative_path.replace('\\', '/')
    if (PurePosixPath(normalized).is_absolute() or PureWindowsPath(relative_path).drive
            or PureWindowsPath(relative_path).root or ':' in normalized
            or any(part in ('', '.', '..') for part in normalized.split('/'))):
        raise DataIntegrityError(f'Unsafe payload path: {relative_path!r}')
    try:
        result = (root / normalized).resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise DataIntegrityError(f'Missing or invalid payload: {relative_path!r}') from exc
    if not result.is_relative_to(root) or not result.is_file():
        raise DataIntegrityError(f'Payload escapes manifest root or is not a file: {relative_path!r}')
    return result


def read_verified_payload(root: Path | str, entry: dict) -> tuple[bytes, Path]:
    expected_size = entry.get('bytes')
    if isinstance(expected_size, bool) or not isinstance(expected_size, int) or expected_size < 0:
        raise DataIntegrityError('Payload bytes must be a nonnegative integer')
    expected_sha = entry.get('sha256', '')
    if not isinstance(expected_sha, str) or re.fullmatch('[0-9a-fA-F]{64}', expected_sha) is None:
        raise DataIntegrityError('Payload SHA256 must contain exactly 64 hexadecimal characters')
    path = safe_payload_path(root, entry.get('relative_path'))
    if path.stat().st_size != expected_size:
        raise DataIntegrityError(f'Payload byte-count mismatch: {path}')
    content = path.read_bytes()
    if len(content) != expected_size or hashlib.sha256(content).hexdigest() != expected_sha.lower():
        raise DataIntegrityError(f'Payload SHA256/bytes mismatch: {path}')
    return content, path


def _integer(text: Any, field: str) -> int:
    if not isinstance(text, str) or re.fullmatch(r'[+-]?\d+', text.strip()) is None:
        raise DataIntegrityError(f'Invalid integer annotation field {field}: {text!r}')
    return int(text)


def _check_xyxy(box: list[int | float], width: int, height: int, context: str) -> None:
    if len(box) != 4 or not all(math.isfinite(value) for value in box):
        raise DataIntegrityError(f'{context}: nonfinite or malformed box')
    x1, y1, x2, y2 = box
    if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
        raise DataIntegrityError(f'{context}: box outside image or nonpositive area: {box}')


def _annotation_result(boxes, labels, ignored, raw_records, convention, mapping):
    return {'boxes': boxes, 'labels': labels, 'ignore_boxes': [entry['xyxy'] for entry in ignored],
            'ignore_metadata': ignored, 'annotation_records': raw_records,
            'coordinate_convention': convention, 'category_mapping': mapping}


def parse_neu_voc(content: bytes, image_size: tuple[int, int], sample_id: str | None = None) -> dict:
    """VOC 1-based inclusive -> 0-based half-open. difficult=1 remains explicit ignore."""
    width, height = image_size
    if b'<!DOCTYPE' in content.upper() or b'<!ENTITY' in content.upper():
        raise DataIntegrityError('DTD/entity declarations are not allowed in annotations')
    try:
        root = ET.fromstring(content)
    except ET.ParseError as exc:
        raise DataIntegrityError('Malformed NEU VOC XML') from exc
    if root.tag != 'annotation':
        raise DataIntegrityError('NEU XML root must be annotation')
    if (_integer(root.findtext('size/width'), 'width'), _integer(root.findtext('size/height'), 'height')) != image_size:
        raise DataIntegrityError('NEU XML dimensions do not match decoded image')
    filename = root.findtext('filename')
    if not filename:
        raise DataIntegrityError('NEU XML filename is required')
    # Some authentic NEU XML files omit .jpg; accept matching stem explicitly.
    if sample_id is not None and (Path(filename).stem != sample_id or '/' in filename or '\\' in filename):
        raise DataIntegrityError('NEU XML filename does not match sample stem')
    boxes, labels, ignored, records = [], [], [], []
    mapping = {name: index for index, name in enumerate(NEU_CLASS_NAMES)}
    for index, obj in enumerate(root.findall('object')):
        name = obj.findtext('name')
        if name not in mapping:
            raise DataIntegrityError(f'Unknown NEU category: {name!r}')
        raw = [_integer(obj.findtext(f'bndbox/{key}'), key) for key in ('xmin', 'ymin', 'xmax', 'ymax')]
        if not (1 <= raw[0] <= raw[2] <= width and 1 <= raw[1] <= raw[3] <= height):
            raise DataIntegrityError(f'NEU object {index}: invalid 1-based inclusive box {raw}')
        box = [raw[0] - 1, raw[1] - 1, raw[2], raw[3]]
        _check_xyxy(box, width, height, f'NEU object {index}')
        difficult = _integer(obj.findtext('difficult', '0'), 'difficult')
        truncated = _integer(obj.findtext('truncated', '0'), 'truncated')
        if difficult not in (0, 1) or truncated not in (0, 1):
            raise DataIntegrityError('NEU difficult/truncated flags must be 0 or 1')
        record = {'object_index': index, 'class_name': name, 'local_class_id': mapping[name],
                  'raw_xml_xyxy': raw, 'xyxy': box, 'difficult': difficult, 'truncated': truncated,
                  'is_ignore': bool(difficult)}
        records.append(record)
        if difficult:
            ignored.append(dict(record, reasons=['voc_difficult']))
        else:
            boxes.append(box)
            labels.append(mapping[name])
    return _annotation_result(boxes, labels, ignored, records,
                              'VOC 1-based inclusive xyxy -> [xmin-1,ymin-1,xmax,ymax] 0-based half-open', mapping)


def parse_visdrone_txt(content: bytes, image_size: tuple[int, int]) -> dict:
    """Native rows preserved; category0/11 or score0 are ignore, never background labels."""
    width, height = image_size
    try:
        lines = content.decode('utf-8-sig').splitlines()
    except UnicodeDecodeError as exc:
        raise DataIntegrityError('VisDrone annotation must be UTF-8 TXT') from exc
    boxes, labels, ignored, records = [], [], [], []
    mapping = {category: category - 1 for category in range(1, 11)}
    for line_number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        fields = [field.strip() for field in line.split(',')]
        if fields[-1] == '':  # Explicitly support the source format's optional trailing comma.
            fields.pop()
        if len(fields) != 8 or any(field == '' for field in fields):
            raise DataIntegrityError(f'VisDrone line {line_number}: expected exactly eight integer fields')
        values = [_integer(field, f'line {line_number}') for field in fields]
        x, y, box_width, box_height, score, category, truncation, occlusion = values
        if score not in (0, 1) or category not in range(12) or truncation not in (0, 1, 2) or occlusion not in (0, 1, 2):
            raise DataIntegrityError(f'VisDrone line {line_number}: unsupported score/category/flags')
        box = [x, y, x + box_width, y + box_height]
        _check_xyxy(box, width, height, f'VisDrone line {line_number}')
        reasons = []
        if score == 0:
            reasons.append('annotation_score_zero')
        if category == 0:
            reasons.append('ignored_region_category_0')
        if category == 11:
            reasons.append('non_target_others_category_11')
        record = {'line_number': line_number, 'raw_values': values, 'xyxy': box,
                  'raw_category_id': category, 'local_class_id': mapping.get(category),
                  'class_name': VISDRONE_CLASS_NAMES[category - 1] if category in mapping else ('ignored_region' if category == 0 else 'others'),
                  'is_ignore': bool(reasons)}
        records.append(record)
        if reasons:
            ignored.append(dict(record, reasons=reasons))
        else:
            boxes.append(box)
            labels.append(mapping[category])
    return _annotation_result(boxes, labels, ignored, records,
                              'Native 0-based left,top,width,height -> half-open [left,top,left+width,top+height]', mapping)


def letterbox_metadata(original_size: tuple[int, int], image_size: int | tuple[int, int] = 640) -> dict:
    """original_size is (width,height); tuple image_size is (height,width)."""
    width, height = original_size
    _positive_int(width, 'original width')
    _positive_int(height, 'original height')
    if isinstance(image_size, int) and not isinstance(image_size, bool):
        output_height = output_width = _positive_int(image_size, 'image_size')
    elif isinstance(image_size, (tuple, list)) and len(image_size) == 2:
        output_height, output_width = (_positive_int(v, 'image_size dimension') for v in image_size)
    else:
        raise DataIntegrityError('image_size must be a positive integer or (height,width) pair')
    nominal = min(Fraction(output_width, width), Fraction(output_height, height))

    def half_up(value):
        return (2 * value.numerator + value.denominator) // (2 * value.denominator)

    resized_width = min(output_width, max(1, half_up(width * nominal)))
    resized_height = min(output_height, max(1, half_up(height * nominal)))
    pad_width, pad_height = output_width - resized_width, output_height - resized_height
    left, top = pad_width // 2, pad_height // 2
    return {'original_hw': (height, width), 'resized_hw': (resized_height, resized_width),
            'output_hw': (output_height, output_width),
            'nominal_scale': float(nominal), 'nominal_scale_fraction': (nominal.numerator, nominal.denominator),
            'scale_x': resized_width / width, 'scale_y': resized_height / height,
            'scale_x_fraction': (resized_width, width), 'scale_y_fraction': (resized_height, height),
            'pad_left': left, 'pad_top': top, 'pad_right': pad_width - left, 'pad_bottom': pad_height - top,
            'resize_rounding': 'exact rational round-half-up; integer resized dimensions determine actual axis scales',
            'padding_rgb': (114, 114, 114), 'resize_filter': 'Pillow BILINEAR',
            'coordinate_system': '0-based half-open xyxy pixels'}


def _box_tensor(boxes) -> torch.Tensor:
    tensor = torch.as_tensor(boxes, dtype=torch.float32, device='cpu')
    if tensor.numel() == 0:
        if tensor.ndim > 2 or (tensor.ndim == 2 and tensor.shape[1] != 4):
            raise DataIntegrityError('Empty boxes must still have shape (0,4)')
        return tensor.reshape(0, 4)
    if tensor.ndim != 2 or tensor.shape[1] != 4 or not torch.isfinite(tensor).all():
        raise DataIntegrityError('Expected finite Nx4 boxes')
    return tensor


def transform_boxes(boxes, transform: dict, inverse: bool = False) -> torch.Tensor:
    result = _box_tensor(boxes).clone()
    scale = torch.tensor([transform['scale_x'], transform['scale_y']] * 2, dtype=torch.float32)
    offset = torch.tensor([transform['pad_left'], transform['pad_top']] * 2, dtype=torch.float32)
    return (result - offset) / scale if inverse else result * scale + offset


def letterbox_rgb(image: Image.Image, image_size: int | tuple[int, int] = 640) -> tuple[torch.Tensor, dict]:
    transform = letterbox_metadata(image.size, image_size)
    output_height, output_width = transform['output_hw']
    resized_height, resized_width = transform['resized_hw']
    rgb = image.convert('RGB')
    resized = rgb.resize((resized_width, resized_height), Image.Resampling.BILINEAR)
    canvas = Image.new('RGB', (output_width, output_height), transform['padding_rgb'])
    canvas.paste(resized, (transform['pad_left'], transform['pad_top']))
    # Copy pixels into a writable buffer; tensor owns a CPU float32 copy after conversion.
    tensor = torch.frombuffer(bytearray(canvas.tobytes()), dtype=torch.uint8).reshape(output_height, output_width, 3)
    return tensor.permute(2, 0, 1).contiguous().to(torch.float32).div_(255.0), transform


class DevelopmentSampleDataset(Dataset):
    """Verified development previews only. Default size640; size128 is a CPU smoke option.

    Every indexed payload is verified on construction by default, and each sample
    re-reads/re-verifies its exact bytes on access (no TOCTOU decode gap). Original
    source paths in metadata are NOT opened; only safe relative payloads are read.
    """
    def __init__(self, manifest_path: str | Path, image_size: int | tuple[int, int] = 640,
                 dataset: str | None = None, *, verify_all: bool = True,
                 expected_manifest_sha256: str | None = None):
        self.manifest_path = Path(manifest_path).resolve(strict=True)
        self.root = self.manifest_path.parent
        manifest_bytes = self.manifest_path.read_bytes()
        self.manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
        if expected_manifest_sha256 is not None and self.manifest_sha256 != expected_manifest_sha256.lower():
            raise DataIntegrityError('Manifest SHA256 mismatch')
        try:
            self.manifest = json.loads(manifest_bytes)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise DataIntegrityError('Malformed source manifest JSON') from exc
        if self.manifest.get('schema_version') != 1 or 'development' not in self.manifest.get('purpose', '').lower():
            raise DataIntegrityError('Expected version1 explicitly development-only source manifest')
        selected = _dataset_name(dataset) if dataset is not None else None
        self.image_size = image_size
        letterbox_metadata((1, 1), image_size)  # Validate transform arguments immediately.
        self.samples = []
        seen = set()
        for group in self.manifest.get('datasets', []):
            lane = _dataset_name(group.get('dataset'))
            if group.get('sample_count') != len(group.get('samples', [])):
                raise DataIntegrityError('Dataset sample_count mismatch')
            for sample in group['samples']:
                identity = (lane, sample.get('sample_id'))
                if not isinstance(identity[1], str) or not identity[1] or identity in seen:
                    raise DataIntegrityError('Missing or duplicate sample identity')
                seen.add(identity)
                _positive_int(sample.get('width'), 'sample width')
                _positive_int(sample.get('height'), 'sample height')
                entries = sample.get('files', [])
                if len(entries) != 2 or sorted(entry.get('kind', '') for entry in entries) != ['annotation', 'image']:
                    raise DataIntegrityError('Each sample requires exactly one original image and one annotation')
                for entry in entries:
                    path = safe_payload_path(self.root, entry.get('relative_path'))
                    if path.stem != identity[1]:
                        raise DataIntegrityError('Payload stem does not match sample_id')
                    expected_suffixes = ('.jpg', '.jpeg', '.png') if entry['kind'] == 'image' else (('.xml',) if lane == 'neu' else ('.txt',))
                    if path.suffix.lower() not in expected_suffixes:
                        raise DataIntegrityError('Unexpected original payload file type')
                    if verify_all:
                        read_verified_payload(self.root, entry)
                if selected is None or lane == selected:
                    self.samples.append({'dataset': lane, 'sample': deepcopy(sample),
                                         'prior_split_statement': group.get('prior_split_statement'),
                                         'split_boundary': group.get('split_boundary')})
        if not self.samples:
            raise DataIntegrityError('No development samples selected')

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index: int) -> dict:
        record = self.samples[index]
        sample, lane = record['sample'], record['dataset']
        entries = {entry['kind']: entry for entry in sample['files']}
        image_bytes, image_path = read_verified_payload(self.root, entries['image'])
        annotation_bytes, annotation_path = read_verified_payload(self.root, entries['annotation'])
        try:
            with Image.open(BytesIO(image_bytes)) as opened:
                image = opened.convert('RGB')
                image.load()
        except (OSError, ValueError) as exc:
            raise DataIntegrityError('Original image cannot be decoded') from exc
        if image.size != (sample['width'], sample['height']):
            raise DataIntegrityError('Decoded image dimensions differ from manifest')
        annotation = (parse_neu_voc(annotation_bytes, image.size, sample['sample_id']) if lane == 'neu'
                      else parse_visdrone_txt(annotation_bytes, image.size))
        tensor, transform = letterbox_rgb(image, self.image_size)
        original_boxes = _box_tensor(annotation['boxes'])
        original_ignore = _box_tensor(annotation['ignore_boxes'])
        boxes = transform_boxes(original_boxes, transform)
        ignore_boxes = transform_boxes(original_ignore, transform)
        labels = torch.tensor(annotation['labels'], dtype=torch.int64, device='cpu')
        height, width = transform['output_hw']
        ignore_mask = torch.zeros((height, width), dtype=torch.bool)
        for box in ignore_boxes.tolist():
            x1, y1, x2, y2 = math.floor(box[0]), math.floor(box[1]), math.ceil(box[2]), math.ceil(box[3])
            ignore_mask[max(0, y1):min(height, y2), max(0, x1):min(width, x2)] = True
        content_mask = torch.zeros((height, width), dtype=torch.bool)
        top, left = transform['pad_top'], transform['pad_left']
        resized_height, resized_width = transform['resized_hw']
        content_mask[top:top + resized_height, left:left + resized_width] = True
        has_ignore = bool(annotation['ignore_metadata'])
        return {'image': tensor, 'boxes': boxes, 'labels': labels, 'classes': labels,
                'boxes_original': original_boxes, 'ignore_boxes': ignore_boxes,
                'ignore_boxes_original': original_ignore,
                'ignore_metadata': annotation['ignore_metadata'], 'annotation_records': annotation['annotation_records'],
                'ignore_mask': ignore_mask, 'content_mask': content_mask,
                'background_valid_mask': content_mask & ~ignore_mask,
                'geometric_masks_are_not_training_background_labels': True,
                'has_ignore_regions': has_ignore,
                'current_stage_loss_policy': ('loss=None: ignore-aware training policy not implemented; shape/coordinate checks only'
                                              if has_ignore else 'development-only forward/coordinate/loss-gradient smoke; no optimizer step'),
                'image_hw': transform['output_hw'], 'original_hw': transform['original_hw'], 'transform': transform,
                'dataset': lane, 'sample_id': sample['sample_id'], 'class_names': CLASS_NAMES[lane],
                'category_mapping': annotation['category_mapping'], 'class_namespace': lane,
                'coordinate_convention': annotation['coordinate_convention'],
                'development_only': True, 'usage_boundary': DEVELOPMENT_ONLY,
                'prior_split_statement': record['prior_split_statement'], 'split_boundary': record['split_boundary'],
                'provenance': {'manifest': str(self.manifest_path), 'manifest_sha256': self.manifest_sha256,
                               'image_path': str(image_path), 'image_sha256': entries['image']['sha256'],
                               'annotation_path': str(annotation_path), 'annotation_sha256': entries['annotation']['sha256'],
                               'payload_bytes_verified': True, 'payload_sha256_verified': True}}


def load_development_sample(manifest_path: str | Path, dataset_name: str, sample_id: str,
                            image_size: int | tuple[int, int] = 640) -> dict:
    dataset = DevelopmentSampleDataset(manifest_path, image_size, dataset_name)
    for index, record in enumerate(dataset.samples):
        if record['sample']['sample_id'] == sample_id:
            return dataset[index]
    raise KeyError(f'No development sample {dataset_name}/{sample_id}')
