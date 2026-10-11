"""Temporary synthetic originals for FullNeuDataset; CPU reads only, no optimizer."""
from copy import deepcopy
import hashlib
from io import BytesIO
import json
from pathlib import Path
import shutil
import tempfile
import unittest

from PIL import Image
import torch

from dataset import (CLASS_NAMES, DatasetIntegrityError, FullNeuDataset,
                     collate_fn, inverse_transform_boxes)


def xml_bytes(stem, boxes, suffix=True):
    objects = ''.join('<object><name>' + name + '</name><difficult>' + str(difficult)
                      + '</difficult><truncated>0</truncated><bndbox>'
                      + ''.join(f'<{key}>{value}</{key}>' for key, value in zip(('xmin', 'ymin', 'xmax', 'ymax'), box))
                      + '</bndbox></object>' for name, box, difficult in boxes)
    filename = stem + '.jpg' if suffix else stem
    return (f'<annotation><filename>{filename}</filename><size><width>10</width><height>6</height>'
            f'</size>{objects}</annotation>').encode()


class FullNeuDatasetTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='full_neu_dataset_test_')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.raw = self.root / 'raw_neu'
        (self.raw / 'IMAGES/crazing').mkdir(parents=True)
        (self.raw / 'ANNOTATIONS').mkdir()
        stream = BytesIO()
        Image.new('L', (10, 6), 90).save(stream, format='JPEG')
        self.jpeg = stream.getvalue()
        with Image.open(BytesIO(self.jpeg)) as opened:
            rgb_bytes = opened.convert('RGB').tobytes()
        pixel_sha = hashlib.sha256(b'RGB:10x6:' + rgb_bytes).hexdigest()
        specs = [('crazing_a', 'train', [('patches', [1, 1, 8, 6], 0), ('inclusion', [2, 2, 5, 5], 1)], False),
                 ('crazing_b', 'train', [('patches', [2, 1, 9, 6], 0)], True),
                 ('crazing_v', 'val', [('crazing', [1, 1, 8, 6], 0)], True),
                 ('crazing_u', 'unassigned', [('crazing', [1, 1, 8, 6], 0)], True)]
        samples = []
        for identity, split, boxes, suffix in specs:
            image_path = self.raw / 'IMAGES/crazing' / f'{identity}.jpg'
            annotation_path = self.raw / 'ANNOTATIONS' / f'{identity}.xml'
            image_path.write_bytes(self.jpeg)
            xml = xml_bytes(identity, boxes, suffix)
            annotation_path.write_bytes(xml)
            image_sha, annotation_sha = hashlib.sha256(self.jpeg).hexdigest(), hashlib.sha256(xml).hexdigest()
            samples.append({'sample_id': identity, 'source_class_directory': 'crazing',
                            'existing_split': split, 'source_split': split,
                            'existing_split_memberships': [] if split == 'unassigned' else [split],
                            'source_image': str(image_path), 'source_image_sha256': image_sha,
                            'source_annotation': str(annotation_path), 'source_annotation_sha256': annotation_sha,
                            'decoded_size': [10, 6],
                            'image': {'path': str(image_path), 'bytes': len(self.jpeg), 'sha256': image_sha,
                                      'width': 10, 'height': 6, 'decoded_rgb_sha256': pixel_sha},
                            'annotation': {'path': str(annotation_path), 'bytes': len(xml), 'sha256': annotation_sha},
                            'raw_boxes': [{'intentionally_untrusted_cached_box': [99, 99, 999, 999]}]})
        self.manifest = {'provenance': {'source_root': str(self.raw),
                                       'class_names_from_existing_yaml': list(CLASS_NAMES)}, 'samples': samples}
        self.manifest_path = self.root / 'sample_manifest.json'
        self.pin = self.save_manifest()

    def save_manifest(self):
        content = json.dumps(self.manifest).encode()
        self.manifest_path.write_bytes(content)
        return hashlib.sha256(content).hexdigest()

    def dataset(self, **overrides):
        kwargs = dict(manifest_path=self.manifest_path, raw_root=self.raw, split='train', image_size=13,
                      coordinate_policy='legacy_raw_xyxy', difficult_policy='legacy_include',
                      expected_manifest_sha256=self.pin)
        kwargs.update(overrides)
        return FullNeuDataset(**kwargs)

    def test_explicit_coordinates_and_difficult_policies(self):
        raw_include = self.dataset()[0]
        torch.testing.assert_close(raw_include['original_boxes'], torch.tensor([[1., 1., 8., 6.], [2., 2., 5., 5.]]))
        self.assertEqual(raw_include['original_labels'].tolist(), [2, 1])
        self.assertEqual(len(raw_include['ignore_boxes']), 0)
        shifted = self.dataset(coordinate_policy='voc1_inclusive_assumption', difficult_policy='explicit_ignore')[0]
        torch.testing.assert_close(shifted['original_boxes'], torch.tensor([[0., 0., 8., 6.]]))
        torch.testing.assert_close(shifted['original_ignore_boxes'], torch.tensor([[1., 1., 5., 5.]]))
        self.assertEqual(shifted['original_ignore_labels'].tolist(), [1])
        self.assertIn('assumption', shifted['provenance']['coordinate_policy'])

    def test_missing_suffix_and_object_name_over_directory(self):
        item = self.dataset()[0]
        self.assertEqual(item['labels'].tolist(), [2, 1])  # Neither object inherits directory crazing=0.
        self.assertEqual(item['raw_objects'][0]['raw_xyxy'], [1, 1, 8, 6])
        self.assertEqual(item['provenance']['annotation_source'], 'reparsed verified original XML bytes, not audit cached boxes')

    def test_split_membership_and_duplicate_rows_preserved(self):
        dataset = self.dataset()
        self.assertEqual(dataset.sample_ids, ['crazing_a', 'crazing_b'])
        self.assertEqual(len(dataset), 2)
        self.assertEqual(dataset[0]['source_group'], dataset[1]['source_group'])
        self.assertNotEqual(dataset[0]['raw_objects'], dataset[1]['raw_objects'])
        self.assertEqual(len(self.dataset(split='val')), 1)
        self.assertNotIn('crazing_u', dataset.sample_ids)

    def test_missing_mandatory_options_and_unknown_policies(self):
        with self.assertRaises(TypeError):
            FullNeuDataset(self.manifest_path, self.raw, 'train')
        for kwargs in ({'split': 'unassigned'}, {'split': 'test'}, {'coordinate_policy': 'auto'},
                       {'difficult_policy': None}, {'expected_manifest_sha256': None}):
            with self.subTest(kwargs=kwargs), self.assertRaises(DatasetIntegrityError):
                self.dataset(**kwargs)

    def test_pin_mismatch_and_tampered_payload(self):
        with self.assertRaises(DatasetIntegrityError):
            self.dataset(expected_manifest_sha256='0' * 64)
        dataset = self.dataset()
        path = Path(self.manifest['samples'][0]['source_image'])
        content = bytearray(path.read_bytes())
        content[-1] ^= 1
        path.write_bytes(content)  # Same length: must fail SHA, not just bytes.
        with self.assertRaises(DatasetIntegrityError):
            dataset[0]

    def test_annotation_byte_count_tampering(self):
        dataset = self.dataset()
        annotation = Path(self.manifest['samples'][0]['source_annotation'])
        annotation.write_bytes(annotation.read_bytes() + b' ')
        with self.assertRaises(DatasetIntegrityError):
            dataset[0]

    def test_path_escape_and_conflicting_aliases(self):
        sample = self.manifest['samples'][0]
        escape = str(self.raw / 'IMAGES' / '..' / '..' / 'outside.jpg')
        sample['source_image'] = escape
        sample['image']['path'] = escape
        pin = self.save_manifest()
        with self.assertRaises(DatasetIntegrityError):
            self.dataset(expected_manifest_sha256=pin)

    def test_explicit_relocated_root_is_used(self):
        relocated = self.root / 'relocated'
        shutil.copytree(self.raw, relocated)
        original_annotation = Path(self.manifest['samples'][0]['source_annotation'])
        original_annotation.write_bytes(b'do not read this obsolete location')
        item = self.dataset(raw_root=relocated)[0]
        self.assertTrue(Path(item['provenance']['image_path']).is_relative_to(relocated.resolve()))
        self.assertEqual(item['labels'].tolist(), [2, 1])

    def test_letterbox_inverse_valid_region_and_tensor_types(self):
        item = self.dataset()[0]
        self.assertEqual(tuple(item['image'].shape), (3, 13, 13))
        self.assertEqual(item['image'].dtype, torch.float32)
        self.assertEqual(item['image'].device.type, 'cpu')
        self.assertEqual(item['labels'].dtype, torch.int64)
        self.assertEqual(item['image_hw'], (13, 13))
        self.assertEqual(item['original_hw'], (6, 10))
        torch.testing.assert_close(item['valid_region'], torch.tensor([0., 2., 13., 10.]))
        self.assertEqual(item['transform']['scale_x'], 1.3)
        self.assertEqual(item['transform']['scale_y'], 8 / 6)
        torch.testing.assert_close(inverse_transform_boxes(item['boxes'], item['transform']), item['original_boxes'])

    def test_collate_contract(self):
        dataset = self.dataset()
        images, targets, metadata = collate_fn([dataset[0], dataset[1]])
        self.assertEqual(tuple(images.shape), (2, 3, 13, 13))
        self.assertEqual(targets[0]['image_hw'], (13, 13))
        self.assertEqual(metadata[1]['sample_id'], 'crazing_b')
        self.assertIn('original_ignore_boxes', metadata[0])
        with self.assertRaises(DatasetIntegrityError):
            collate_fn([])

    def test_audit_split_conflict_and_duplicate_ids_fail(self):
        self.manifest['samples'][0]['existing_split_memberships'] = ['train', 'val']
        pin = self.save_manifest()
        with self.assertRaises(DatasetIntegrityError):
            self.dataset(expected_manifest_sha256=pin)
        self.manifest['samples'][0]['existing_split_memberships'] = ['train']
        self.manifest['samples'].append(deepcopy(self.manifest['samples'][0]))
        pin = self.save_manifest()
        with self.assertRaises(DatasetIntegrityError):
            self.dataset(expected_manifest_sha256=pin)


if __name__ == '__main__':
    unittest.main(verbosity=2)
