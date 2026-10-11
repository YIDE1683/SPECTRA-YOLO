"""Read-only full NEU source/split audit; no split repair, coordinate choice or training.

Only the explicitly selected output directory is written. Native XML numbers are
preserved and alternative coordinate interpretations are compared, not assumed.
Decoded-pixel hashes are domain-separated by RGB mode and image dimensions.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
from io import BytesIO
import json
import math
from pathlib import Path
import platform
import re
import sys
import time
import xml.etree.ElementTree as ET

from PIL import Image, __version__ as PILLOW_VERSION

SOURCE = Path(r'D:\workPRO\TARGETdetcet\SAM_T\datasets\NEU-DET')
SPLIT = Path(r'D:\workPRO\TARGETdetcet\SAM_T\yolov8_neu_det\data')
ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / 'paper/ieee_sensors_content_rewrite_20260831/training_stage/neu_audit'
CONVENTIONS = ('raw_zero_based_half_open', 'one_based_inclusive', 'zero_based_inclusive')


def sha_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def file_payload(path: Path) -> tuple[bytes, dict]:
    content = path.read_bytes()
    stat = path.stat()
    if stat.st_size != len(content):
        raise RuntimeError(f'File changed while being read: {path}')
    return content, {'path': str(path), 'bytes': len(content), 'sha256': sha_bytes(content),
                     'mtime_ns': stat.st_mtime_ns}


def inspect_image(path: Path) -> dict:
    content, record = file_payload(path)
    try:
        with Image.open(BytesIO(content)) as image:
            image.load()
            rgb = image.convert('RGB')
            record.update(width=image.width, height=image.height, original_mode=image.mode,
                          decoded_rgb_sha256=sha_bytes(f'RGB:{image.width}x{image.height}:'.encode() + rgb.tobytes()),
                          decoded_native_sha256=sha_bytes(f'{image.mode}:{image.width}x{image.height}:'.encode() + image.tobytes()),
                          decoded_ok=True)
    except (OSError, ValueError) as exc:
        record.update(decoded_ok=False, decode_error=str(exc))
    return record


def numeric(text, name):
    if text is None:
        raise ValueError(f'Missing {name}')
    value = float(text)
    if not math.isfinite(value):
        raise ValueError(f'Nonfinite {name}')
    return int(value) if value.is_integer() else value


def candidate_xyxy(raw, convention):
    x1, y1, x2, y2 = raw
    if convention == 'raw_zero_based_half_open':
        return [x1, y1, x2, y2]
    if convention == 'one_based_inclusive':
        return [x1 - 1, y1 - 1, x2, y2]
    if convention == 'zero_based_inclusive':
        return [x1, y1, x2 + 1, y2 + 1]
    raise ValueError(f'Unknown coordinate candidate {convention}')


def candidate_valid(raw, width, height, convention):
    x1, y1, x2, y2 = candidate_xyxy(raw, convention)
    return 0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height


def candidate_yolo(raw, width, height, convention):
    x1, y1, x2, y2 = candidate_xyxy(raw, convention)
    return [(x1 + x2) / (2 * width), (y1 + y2) / (2 * height),
            (x2 - x1) / width, (y2 - y1) / height]


def parse_voc_audit(content: bytes, image_size=None) -> dict:
    """Do not reject xmin/ymin=0 merely because a candidate uses one-based coordinates."""
    result = {'objects': [], 'parse_errors': []}
    if b'<!DOCTYPE' in content.upper() or b'<!ENTITY' in content.upper():
        result['parse_errors'].append('DTD/entity declarations rejected')
        return result
    try:
        root = ET.fromstring(content)
        if root.tag != 'annotation':
            raise ValueError('Root is not annotation')
        result['filename_field'] = root.findtext('filename')
        result['filename_missing_suffix'] = bool(result['filename_field']) and not Path(result['filename_field']).suffix
        width = numeric(root.findtext('size/width'), 'XML width')
        height = numeric(root.findtext('size/height'), 'XML height')
        result['xml_size'] = [width, height]
        result['xml_size_matches_image'] = image_size is None or list(image_size) == [width, height]
        if width <= 0 or height <= 0:
            raise ValueError('Nonpositive XML dimensions')
        for index, obj in enumerate(root.findall('object')):
            record = {'object_index': index, 'class_name': obj.findtext('name'),
                      'difficult_text': obj.findtext('difficult'), 'truncated_text': obj.findtext('truncated')}
            try:
                raw = [numeric(obj.findtext(f'bndbox/{key}'), key) for key in ('xmin', 'ymin', 'xmax', 'ymax')]
                record['raw_xml_xyxy'] = raw
                record['raw_facts'] = {'xmin_equals_zero': raw[0] == 0, 'ymin_equals_zero': raw[1] == 0,
                                       'xmax_equals_width': raw[2] == width, 'ymax_equals_height': raw[3] == height,
                                       'has_negative_coordinate': any(value < 0 for value in raw),
                                       'has_noninteger_coordinate': any(int(value) != value for value in raw),
                                       'x_inverted': raw[2] < raw[0], 'y_inverted': raw[3] < raw[1],
                                       'x_equal': raw[2] == raw[0], 'y_equal': raw[3] == raw[1]}
                record['candidate_bounds_valid'] = {name: candidate_valid(raw, width, height, name) for name in CONVENTIONS}
            except (ValueError, TypeError) as exc:
                record['parse_error'] = str(exc)
                result['parse_errors'].append(f'object {index}: {exc}')
            result['objects'].append(record)
    except (ET.ParseError, ValueError, TypeError) as exc:
        result['parse_errors'].append(str(exc))
    return result


def parse_class_names(content: str) -> list[str]:
    """Bounded parser for the actual legacy YAML names list; no new YAML dependency."""
    names, inside = [], False
    for line in content.splitlines():
        if line.strip() == 'names:':
            inside = True
            continue
        if not inside or not line.strip() or line.lstrip().startswith('#'):
            continue
        match = re.fullmatch(r'\s*-\s*[\"\']?([^\"\'#]+?)[\"\']?\s*', line)
        if not match:
            break
        names.append(match.group(1).strip())
    if not names or len(set(names)) != len(names):
        raise ValueError('Legacy YAML must contain a nonempty unique names list')
    return names


def parse_yolo_labels(content: bytes, number_of_classes: int) -> dict:
    records, errors = [], []
    try:
        lines = content.decode('utf-8-sig').splitlines()
    except UnicodeDecodeError as exc:
        return {'rows': [], 'errors': [str(exc)]}
    for number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            fields = line.split()
            if len(fields) != 5:
                raise ValueError('Expected five fields')
            category = int(fields[0])
            coords = [float(value) for value in fields[1:]]
            if not 0 <= category < number_of_classes or not all(math.isfinite(value) for value in coords):
                raise ValueError('Invalid category or nonfinite coordinate')
            if not all(0 <= value <= 1 for value in coords) or coords[2] <= 0 or coords[3] <= 0:
                raise ValueError('Invalid normalized YOLO coordinates')
            records.append({'line_number': number, 'class_id': category, 'xywh_normalized': coords})
        except ValueError as exc:
            errors.append(f'line {number}: {exc}')
    return {'rows': records, 'errors': errors}


def compare_yolo_to_xml(annotation, labels, class_names, width, height, tolerance=5.1e-7):
    objects, rows = annotation['objects'], labels['rows']
    result = {'ordered_row_count_match': len(objects) == len(rows), 'tolerance': tolerance,
              'comparison_order': 'original XML object order vs legacy label row order; no reassignment',
              'candidates': {}}
    result['ordered_class_match'] = (len(objects) == len(rows) and all(
        obj.get('class_name') in class_names and class_names.index(obj['class_name']) == row['class_id']
        for obj, row in zip(objects, rows)))
    if annotation['parse_errors'] or labels['errors'] or not result['ordered_row_count_match']:
        result['comparison_available'] = False
        return result
    result['comparison_available'] = True
    for convention in CONVENTIONS:
        differences, row_details = [], []
        for obj, row in zip(objects, rows):
            expected = candidate_yolo(obj['raw_xml_xyxy'], width, height, convention)
            delta = [actual - ref for actual, ref in zip(row['xywh_normalized'], expected)]
            differences.extend(abs(value) for value in delta)
            row_details.append({'object_index': obj['object_index'], 'expected_xywh_normalized': expected,
                                'label_minus_expected': delta})
        max_difference = max(differences, default=0)
        result['candidates'][convention] = {'max_abs_normalized_difference': max_difference,
                                            'mean_abs_normalized_difference': sum(differences) / len(differences) if differences else 0,
                                            'matches_within_tolerance': result['ordered_class_match'] and max_difference <= tolerance,
                                            'rows': row_details}
    return result


def duplicate_groups(samples, field):
    groups = defaultdict(list)
    for sample in samples:
        value = sample.get('image', {}).get(field)
        if value:
            groups[value].append(sample)
    result = []
    for digest, members in sorted(groups.items()):
        if len(members) < 2:
            continue
        classes = sorted(set(member['source_class_directory'] for member in members))
        split_names = sorted({split for member in members for split in member['existing_split_memberships']})
        annotation_signatures = {json.dumps(member.get('raw_boxes', []), sort_keys=True) for member in members}
        result.append({'hash': digest, 'member_count': len(members), 'cross_class': len(classes) > 1,
                       'cross_train_val': 'train' in split_names and 'val' in split_names,
                       'different_annotation_records': len(annotation_signatures) > 1,
                       'classes': classes, 'existing_split_names': split_names,
                       'members': [{'sample_id': member['sample_id'], 'source_class': member['source_class_directory'],
                                    'source_split': member['source_split'], 'image_path': member['image']['path']} for member in members]})
    return result


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def run(source_root: Path, split_root: Path, output_root: Path):
    source_root, split_root, output_root = source_root.resolve(), split_root.resolve(), output_root.resolve()
    if output_root.is_relative_to(source_root) or output_root.is_relative_to(split_root):
        raise ValueError('Audit output must not be inside immutable source or legacy split directories')
    started = time.perf_counter()
    output_root.mkdir(parents=True, exist_ok=True)
    image_root, annotation_root = source_root / 'IMAGES', source_root / 'ANNOTATIONS'
    yaml_path, converter_path = split_root.parent / 'data.yaml', split_root.parent / 'prepare_data.py'
    yaml_bytes, yaml_meta = file_payload(yaml_path)
    converter_bytes, converter_meta = file_payload(converter_path)
    class_names = parse_class_names(yaml_bytes.decode('utf-8-sig'))
    # Compatibility is checked against the already implemented parser, without
    # transforming any images or changing that parser's coordinate assumptions.
    from data import parse_neu_voc as current_strict_voc_parser
    source_images = sorted(path for folder in image_root.iterdir() if folder.is_dir()
                           for path in folder.iterdir() if path.is_file() and path.suffix.lower() in ('.jpg', '.jpeg', '.png', '.bmp'))
    annotations = {path.stem: path for path in annotation_root.glob('*.xml')}
    split_images, split_labels = {}, {}
    for split in ('train', 'val'):
        split_images[split] = defaultdict(list)
        split_labels[split] = defaultdict(list)
        for path in sorted((split_root / 'images' / split).iterdir()):
            if path.is_file() and path.suffix.lower() in ('.jpg', '.jpeg', '.png', '.bmp'):
                split_images[split][path.stem].append(path)
        for path in sorted((split_root / 'labels' / split).glob('*.txt')):
            split_labels[split][path.stem].append(path)
    provenance = {'utc': datetime.now(timezone.utc).isoformat(), 'python': sys.version, 'platform': platform.platform(),
                  'pillow': PILLOW_VERSION, 'audit_script_sha256': sha_bytes(Path(__file__).read_bytes()),
                  'test_script_sha256': sha_bytes(Path(__file__).with_name('test_neu_audit.py').read_bytes()),
                  'current_strict_data_parser_sha256': sha_bytes(Path(__file__).with_name('data.py').read_bytes()),
                  'source_root': str(source_root), 'existing_split_root': str(split_root),
                  'legacy_yaml': yaml_meta, 'legacy_converter': converter_meta,
                  'legacy_yaml_text': yaml_bytes.decode('utf-8-sig'),
                  'class_names_from_existing_yaml': class_names,
                  'pixel_hash_definition': 'SHA256(UTF8(mode:widthxheight:) + decoded pixel bytes); RGB is common cross-mode comparison',
                  'audit_boundary': 'read-only inventory; no coordinate convention selected, no split repair/reassignment, no training'}
    write_json(output_root / 'audit_provenance.json', provenance)
    samples = []
    for index, image_path in enumerate(source_images, 1):
        identity = image_path.stem
        image = inspect_image(image_path)
        memberships = [name for name in ('train', 'val') if identity in split_images[name]]
        record = {'sample_id': identity, 'source_class_directory': image_path.parent.name, 'image': image,
                  'annotation': None, 'paired_by_sample_id': identity in annotations,
                  'existing_split_memberships': memberships,
                  'source_split': memberships[0] if len(memberships) == 1 else ('train_and_val' if memberships else 'unassigned'),
                  'existing_split_records': {}}
        record.update(source_image=str(image_path), source_image_sha256=image['sha256'],
                      source_annotation=str(annotations[identity]) if identity in annotations else None,
                      source_annotation_sha256=None,
                      decoded_size=[image['width'], image['height']] if image['decoded_ok'] else None,
                      existing_split=record['source_split'], raw_boxes=[], strict_parse_compatible=False,
                      strict_parse_boundary='compatibility with existing data.py parser only; NOT selection of an official coordinate convention')
        if identity in annotations:
            content, metadata = file_payload(annotations[identity])
            parsed = parse_voc_audit(content, (image['width'], image['height']) if image['decoded_ok'] else None)
            metadata.update(parsed)
            metadata['filename_stem_matches_sample_id'] = bool(parsed.get('filename_field')) and Path(parsed['filename_field']).stem == identity
            filename = parsed.get('filename_field')
            metadata['legacy_exact_filename_resolves_any_class'] = bool(filename) and any((folder / filename).is_file() for folder in image_root.iterdir() if folder.is_dir())
            record['annotation'] = metadata
            record['source_annotation_sha256'] = metadata['sha256']
            record['raw_boxes'] = [{'raw_xyxy': obj.get('raw_xml_xyxy'), 'class': obj['class_name'],
                                    'class_id': class_names.index(obj['class_name']) if obj['class_name'] in class_names else None,
                                    'difficult': obj['difficult_text']} for obj in parsed['objects']]
            record['coordinate_convention_candidates'] = {
                name: all(obj.get('candidate_bounds_valid', {}).get(name, False) for obj in parsed['objects'])
                and not parsed['parse_errors'] for name in CONVENTIONS}
            if image['decoded_ok']:
                try:
                    current_strict_voc_parser(content, (image['width'], image['height']), identity)
                    record['strict_parse_compatible'] = True
                    record['strict_parse_error'] = None
                except ValueError as exc:
                    record['strict_parse_error'] = str(exc)
        for split in memberships:
            split_record = {'image_copies': [], 'label_files': []}
            for split_image_path in split_images[split][identity]:
                copy = inspect_image(split_image_path)
                copy['byte_hash_matches_original'] = copy['sha256'] == image['sha256']
                copy['pixel_hash_matches_original'] = copy.get('decoded_rgb_sha256') == image.get('decoded_rgb_sha256')
                split_record['image_copies'].append(copy)
            for label_path in split_labels[split].get(identity, []):
                content, metadata = file_payload(label_path)
                parsed = parse_yolo_labels(content, len(class_names))
                metadata.update(parsed)
                if record['annotation'] is not None and image['decoded_ok']:
                    metadata['xml_candidate_comparison'] = compare_yolo_to_xml(record['annotation'], parsed,
                                                                              class_names, image['width'], image['height'])
                split_record['label_files'].append(metadata)
            record['existing_split_records'][split] = split_record
        samples.append(record)
        if index % 100 == 0 or index == len(source_images):
            write_json(output_root / 'status.json', {'status': 'auditing', 'processed_images': index,
                        'total_images': len(source_images), 'elapsed_seconds': time.perf_counter() - started,
                        'last_sample': identity})
            print(f'[NEU audit] {index}/{len(source_images)} elapsed={time.perf_counter()-started:.1f}s', flush=True)
    source_ids = {sample['sample_id'] for sample in samples}
    stats = Counter()
    candidate_bounds = Counter()
    candidate_label_matches = Counter()
    difficult = Counter()
    categories = Counter()
    dimensions = Counter()
    coordinate_min = [math.inf] * 4
    coordinate_max = [-math.inf] * 4
    per_class = defaultdict(Counter)
    disagreements, unavailable_labels = [], []
    for sample in samples:
        per_class[sample['source_class_directory']][sample['source_split']] += 1
        stats[sample['source_split']] += 1
        stats['strict_parse_compatible_' + sample['source_split']] += sample['strict_parse_compatible']
        stats['all_three_candidates_bounds_valid_' + sample['source_split']] += (
            bool(sample.get('coordinate_convention_candidates'))
            and all(sample['coordinate_convention_candidates'].values()))
        stats['images_decoded_ok'] += bool(sample['image']['decoded_ok'])
        if sample['image']['decoded_ok']:
            dimensions[f'{sample["image"]["width"]}x{sample["image"]["height"]}'] += 1
        ann = sample['annotation']
        if ann is None:
            stats['missing_annotation'] += 1
            continue
        stats['xml_parse_error_samples'] += bool(ann['parse_errors'])
        stats['xml_size_mismatch_samples'] += not ann.get('xml_size_matches_image', False)
        stats['filename_missing_suffix'] += bool(ann.get('filename_missing_suffix'))
        stats['filename_stem_mismatch'] += not ann['filename_stem_matches_sample_id']
        stats['legacy_filename_unresolved'] += not ann['legacy_exact_filename_resolves_any_class']
        stats['empty_annotation_samples'] += not ann['objects']
        stats['samples_with_class_unlike_directory_' + sample['source_split']] += any(
            obj['class_name'] != sample['source_class_directory'] for obj in ann['objects'])
        for obj in ann['objects']:
            stats['total_xml_objects'] += 1
            categories[str(obj['class_name'])] += 1
            difficult[str(obj['difficult_text'])] += 1
            stats['difficult_1_objects_' + sample['source_split']] += obj['difficult_text'] == '1'
            stats['object_class_differs_from_directory'] += obj['class_name'] != sample['source_class_directory']
            stats['unknown_class_objects'] += obj['class_name'] not in class_names
            if 'raw_xml_xyxy' in obj:
                for coordinate, value in enumerate(obj['raw_xml_xyxy']):
                    coordinate_min[coordinate] = min(coordinate_min[coordinate], value)
                    coordinate_max[coordinate] = max(coordinate_max[coordinate], value)
                stats.update({key: int(value) for key, value in obj['raw_facts'].items()})
                candidate_bounds.update({key: int(value) for key, value in obj['candidate_bounds_valid'].items()})
        for split, split_record in sample['existing_split_records'].items():
            stats['split_copy_byte_mismatches'] += sum(not copy['byte_hash_matches_original'] for copy in split_record['image_copies'])
            stats['split_copy_pixel_mismatches'] += sum(not copy['pixel_hash_matches_original'] for copy in split_record['image_copies'])
            stats['assigned_sample_missing_label'] += not split_record['label_files']
            for label in split_record['label_files']:
                stats['label_files_examined'] += 1
                comparison = label.get('xml_candidate_comparison', {})
                stats['label_parse_error_files'] += bool(label['errors'])
                if not comparison.get('comparison_available'):
                    unavailable_labels.append({'sample_id': sample['sample_id'], 'split': split, 'label_path': label['path']})
                else:
                    if not comparison['ordered_class_match']:
                        disagreements.append({'sample_id': sample['sample_id'], 'split': split, 'reason': 'class/order mismatch'})
                    for convention, result in comparison['candidates'].items():
                        candidate_label_matches[convention] += result['matches_within_tolerance']
                        if convention == 'raw_zero_based_half_open' and not result['matches_within_tolerance']:
                            disagreements.append({'sample_id': sample['sample_id'], 'split': split,
                                                  'reason': 'raw direct convention mismatch', 'max_abs_error': result['max_abs_normalized_difference']})
    byte_duplicates = duplicate_groups(samples, 'sha256')
    rgb_duplicates = duplicate_groups(samples, 'decoded_rgb_sha256')
    duplicate_summary = {name: {'groups': len(groups), 'members': sum(group['member_count'] for group in groups),
                                'cross_class_groups': sum(group['cross_class'] for group in groups),
                                'cross_train_val_groups': sum(group['cross_train_val'] for group in groups),
                                'different_annotation_record_groups': sum(group['different_annotation_records'] for group in groups)}
                         for name, groups in [('byte', byte_duplicates), ('decoded_rgb', rgb_duplicates)]}
    summary = {'status': 'read_only_full_inventory_complete_not_training_approval',
               'source_image_count': len(source_images), 'source_xml_count': len(annotations),
               'paired_source_images': sum(sample['paired_by_sample_id'] for sample in samples),
               'annotation_only_ids': sorted(set(annotations) - source_ids),
               'duplicate_source_sample_ids': {key: count for key, count in Counter(sample['sample_id'] for sample in samples).items() if count > 1},
               'source_split_counts': {name: stats[name] for name in ('train', 'val', 'unassigned', 'train_and_val')},
               'existing_split_file_counts': {name: {'images': sum(map(len, split_images[name].values())),
                                                  'labels': sum(map(len, split_labels[name].values())),
                                                  'image_ids_not_in_original': sorted(set(split_images[name]) - source_ids),
                                                  'label_ids_without_split_image': sorted(set(split_labels[name]) - set(split_images[name]))}
                                              for name in ('train', 'val')},
               'per_class_existing_membership_counts': {name: dict(value) for name, value in sorted(per_class.items())},
               'dimensions': dict(dimensions), 'raw_annotation_statistics': dict(stats),
               'raw_coordinate_order': ['xmin', 'ymin', 'xmax', 'ymax'],
               'raw_coordinate_minimum': [None if math.isinf(value) else value for value in coordinate_min],
               'raw_coordinate_maximum': [None if math.isinf(value) else value for value in coordinate_max],
               'object_category_counts': dict(categories), 'difficult_field_counts': dict(difficult),
               'candidate_bounds_valid_object_counts': dict(candidate_bounds),
               'candidate_matches_existing_yolo_files': {name: candidate_label_matches[name] for name in CONVENTIONS},
               'yolo_candidate_tolerance_normalized': 5.1e-7,
               'label_comparison_unavailable': unavailable_labels, 'label_comparison_disagreements': disagreements,
               'duplicate_summary': duplicate_summary,
               'unassigned_ids': [sample['sample_id'] for sample in samples if sample['source_split'] == 'unassigned'],
               'limits': ['No split repair, creation or reassignment was performed.',
                          'Unassigned images remain unassigned; they are not automatically training data.',
                          'Matching legacy YOLO conversion does not independently establish official XML coordinate semantics.',
                          'Coordinate candidates were measured, not selected. No data.py changes.',
                          'Pixel-equality duplicates are content identity evidence, not perceptual near-duplicate analysis.',
                          'No training, inference, model loading, package installation or original-file writes.'],
               'elapsed_seconds': time.perf_counter() - started}
    write_json(output_root / 'sample_manifest.json', {'provenance': provenance, 'samples': samples})
    write_json(output_root / 'duplicate_groups.json', {'byte': byte_duplicates, 'decoded_rgb': rgb_duplicates})
    write_json(output_root / 'summary.json', summary)
    write_json(output_root / 'status.json', {'status': 'complete', 'processed_images': len(samples),
                                          'elapsed_seconds': time.perf_counter() - started})
    report = ['# 本机NEU完整原始数据只读审计', '',
              f'原始图像 {len(source_images)} 张，XML {len(annotations)} 份，按stem配对 {summary["paired_source_images"]} 张。', '',
              '本次不修复旧转换、不重划分、不分配unassigned、不选择坐标约定，也没有训练或模型推理。', '',
              '## 当前实体划分', '',
              f'- train={stats["train"]}；val={stats["val"]}；unassigned={stats["unassigned"]}；同时属于train/val={stats["train_and_val"]}。',
              f'- XML filename缺扩展名={stats["filename_missing_suffix"]}；按旧脚本完整filename找不到图={stats["legacy_filename_unresolved"]}。',
              f'- 实体路径：`{split_root}`；旧data.yaml内容及哈希在audit_provenance.json留存。', '',
              '## 原始坐标事实', '',
              f'- 总object={stats["total_xml_objects"]}；xmin=0：{stats["xmin_equals_zero"]}；ymin=0：{stats["ymin_equals_zero"]}。',
              f'- xmax等于图宽：{stats["xmax_equals_width"]}；ymax等于图高：{stats["ymax_equals_height"]}。',
              f'- 各候选范围合法object计数：`{dict(candidate_bounds)}`。',
              f'- 与现有YOLO文件匹配数：`{summary["candidate_matches_existing_yolo_files"]}`（比较文件数{stats["label_files_examined"]}；容差5.1e-7）。',
              '- 对旧YOLO数值的复算一致性只是历史转换证据，不替代对官方标注坐标语义的确认。', '',
              '## 标注策略风险', '',
              f'- difficult=1共{difficult["1"]}个（train={stats["difficult_1_objects_train"]}，val={stats["difficult_1_objects_val"]}）；旧YOLO转换包含这些框，新方法不能默默切换difficult策略。',
              f'- {stats["object_class_differs_from_directory"]}个object的XML类别与图像目录名不同；必须保留object/name，不按目录覆盖。',
              '- 候选坐标范围都合法不等于坐标解释相同；本审计不会为了获得可训练子集而挑选约定或样本。', '',
              '## 文件、像素重复与跨划分风险', '',
              f'- 文件字节重复：`{duplicate_summary["byte"]}`。',
              f'- 解码RGB像素重复：`{duplicate_summary["decoded_rgb"]}`。',
              f'- 现有split复制图字节不一致={stats["split_copy_byte_mismatches"]}；像素不一致={stats["split_copy_pixel_mismatches"]}。',
              '- duplicate_groups.json给出全部成员和跨类/跨现有train-val关系；不自动删除任何重复项。', '',
              '逐样本原始数值、类别、difficult、源哈希/字节/像素哈希、split及YOLO候选差值见sample_manifest.json。', '']
    (output_root / 'README_CN.md').write_text('\n'.join(report), encoding='utf-8')
    print(json.dumps({'status': summary['status'], 'images': len(source_images),
                      'split_counts': summary['source_split_counts'], 'zero_xmin': stats['xmin_equals_zero'],
                      'zero_ymin': stats['ymin_equals_zero'], 'duplicates': duplicate_summary,
                      'summary': str(output_root / 'summary.json')}, ensure_ascii=False), flush=True)
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', type=Path, default=SOURCE)
    parser.add_argument('--split-root', type=Path, default=SPLIT)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    args = parser.parse_args()
    run(args.source_root, args.split_root, args.output)
