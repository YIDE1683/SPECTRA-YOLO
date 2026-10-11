"""CPU-only temporary-fixture tests; no persistent generated datasets or predictions."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from PIL import Image
import torch

from data import (DataIntegrityError, DevelopmentSampleDataset, letterbox_metadata,
                  letterbox_rgb, load_development_sample, parse_neu_voc,
                  parse_visdrone_txt, safe_payload_path, transform_boxes)


def voc_xml(width=10, height=10, boxes=None, filename='sample.jpg'):
    boxes = boxes if boxes is not None else [('crazing', [1, 1, width, height], 0)]
    objects = ''.join('<object><name>' + name + '</name><difficult>' + str(difficult)
                      + '</difficult><bndbox>'
                      + ''.join(f'<{key}>{value}</{key}>' for key, value in zip(('xmin', 'ymin', 'xmax', 'ymax'), box))
                      + '</bndbox></object>' for name, box, difficult in boxes)
    return (f'<annotation><filename>{filename}</filename><size><width>{width}</width>'
            f'<height>{height}</height><depth>1</depth></size>{objects}</annotation>').encode()


class AnnotationTests(unittest.TestCase):
    def test_neu_inclusive_to_half_open_and_mapping(self):
        result = parse_neu_voc(voc_xml(boxes=[('crazing', [1, 1, 10, 10], 0),
                                             ('scratches', [10, 10, 10, 10], 0)]), (10, 10), 'sample')
        self.assertEqual(result['boxes'], [[0, 0, 10, 10], [9, 9, 10, 10]])
        self.assertEqual(result['labels'], [0, 5])

    def test_neu_missing_suffix_explicitly_supported(self):
        self.assertEqual(parse_neu_voc(voc_xml(filename='sample'), (10, 10), 'sample')['labels'], [0])

    def test_neu_difficult_is_ignore_not_background(self):
        result = parse_neu_voc(voc_xml(boxes=[('inclusion', [1, 1, 4, 5], 1)]), (10, 10))
        self.assertEqual(result['boxes'], [])
        self.assertEqual(result['ignore_boxes'], [[0, 0, 4, 5]])
        self.assertEqual(result['ignore_metadata'][0]['reasons'], ['voc_difficult'])

    def test_neu_invalid_xml_size_category_bounds(self):
        for content in (b'<annotation>', voc_xml(width=11), voc_xml(boxes=[('unknown', [1, 1, 2, 2], 0)]),
                        voc_xml(boxes=[('crazing', [0, 1, 2, 2], 0)]),
                        b'<!DOCTYPE annotation><annotation/>'):
            with self.subTest(content=content), self.assertRaises(DataIntegrityError):
                parse_neu_voc(content, (10, 10))

    def test_visdrone_conversion_full_classes_ignore_and_others(self):
        content = (b'0,0,10,10,1,1,0,0\n1,2,3,4,1,10,1,2,\n'
                   b'0,0,4,4,0,0,0,0\n5,5,3,3,1,11,0,0\n2,2,2,2,0,4,0,0\n')
        result = parse_visdrone_txt(content, (10, 10))
        self.assertEqual(result['boxes'], [[0, 0, 10, 10], [1, 2, 4, 6]])
        self.assertEqual(result['labels'], [0, 9])
        self.assertEqual(len(result['ignore_boxes']), 3)
        self.assertEqual(result['ignore_metadata'][1]['reasons'], ['non_target_others_category_11'])
        self.assertEqual(len(result['annotation_records']), 5)

    def test_visdrone_invalid_rows(self):
        for content in (b'1,2,3', b'0,0,0,1,1,4,0,0', b'0,0,11,1,1,4,0,0',
                        b'0,0,1,1,1,12,0,0', b'0,0,1,1,nan,4,0,0', b'0,,1,1,1,4,0,0'):
            with self.subTest(content=content), self.assertRaises(DataIntegrityError):
                parse_visdrone_txt(content, (10, 10))

    def test_empty_annotations(self):
        self.assertEqual(parse_visdrone_txt(b'', (10, 10))['boxes'], [])
        self.assertEqual(parse_neu_voc(voc_xml(boxes=[]), (10, 10))['ignore_boxes'], [])


class LetterboxTests(unittest.TestCase):
    def test_non_square_exact_axis_scale(self):
        transform = letterbox_metadata((7, 4), 10)
        self.assertEqual(transform['resized_hw'], (6, 10))
        self.assertEqual((transform['pad_top'], transform['pad_bottom']), (2, 2))
        self.assertEqual(transform['scale_x'], 10 / 7)
        self.assertEqual(transform['scale_y'], 1.5)
        boxes = transform_boxes([[0, 0, 7, 4]], transform)
        torch.testing.assert_close(boxes, torch.tensor([[0., 2., 10., 8.]]))
        torch.testing.assert_close(transform_boxes(boxes, transform, inverse=True), torch.tensor([[0., 0., 7., 4.]]))

    def test_half_up_odd_pad_and_rectangular_output(self):
        transform = letterbox_metadata((4, 3), 6)
        self.assertEqual(transform['resized_hw'], (5, 6))  # 4.5 rounds UP, not bankers rounding.
        self.assertEqual((transform['pad_top'], transform['pad_bottom']), (0, 1))
        rectangle = letterbox_metadata((4, 3), (10, 8))
        self.assertEqual(rectangle['output_hw'], (10, 8))
        self.assertEqual(rectangle['resized_hw'], (6, 8))

    def test_rgb_cpu_tensor_padding(self):
        tensor, transform = letterbox_rgb(Image.new('L', (4, 3), 255), 6)
        self.assertEqual(tuple(tensor.shape), (3, 6, 6))
        self.assertEqual(tensor.device.type, 'cpu')
        self.assertEqual(tensor.dtype, torch.float32)
        self.assertTrue(torch.equal(tensor[0], tensor[1]))
        torch.testing.assert_close(tensor[:, 5, :], torch.full((3, 6), 114 / 255))

    def test_empty_boxes_and_invalid_transform(self):
        self.assertEqual(tuple(transform_boxes([], letterbox_metadata((1, 1), 128)).shape), (0, 4))
        with self.assertRaises(DataIntegrityError):
            transform_boxes(torch.empty((0, 5)), letterbox_metadata((1, 1), 128))
        for size in (0, -1, True, (4, 0), (1, 2, 3)):
            with self.subTest(size=size), self.assertRaises(DataIntegrityError):
                letterbox_metadata((10, 10), size)


class LoaderTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='spectra_loader_unittest_')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / 'NEU/images').mkdir(parents=True)
        (self.root / 'NEU/annotations').mkdir(parents=True)
        self.image_path = self.root / 'NEU/images/sample.jpg'
        self.annotation_path = self.root / 'NEU/annotations/sample.xml'
        Image.new('L', (10, 10), 127).save(self.image_path)
        self.annotation_path.write_bytes(voc_xml())
        files = []
        for kind, path in [('image', self.image_path), ('annotation', self.annotation_path)]:
            content = path.read_bytes()
            files.append({'kind': kind, 'relative_path': path.relative_to(self.root).as_posix(),
                          'bytes': len(content), 'sha256': hashlib.sha256(content).hexdigest()})
        self.manifest = {'schema_version': 1, 'purpose': 'development-only test fixture',
                         'datasets': [{'dataset': 'NEU', 'sample_count': 1, 'samples': [
                             {'sample_id': 'sample', 'width': 10, 'height': 10, 'files': files}]}]}
        self.manifest_path = self.root / 'source_manifest.json'
        self.write_manifest()

    def write_manifest(self):
        self.manifest_path.write_text(json.dumps(self.manifest), encoding='utf-8')

    def test_single_sample_and_cpu_contract(self):
        item = load_development_sample(self.manifest_path, 'neu', 'sample', image_size=128)
        self.assertEqual(tuple(item['image'].shape), (3, 128, 128))
        self.assertEqual(item['image_hw'], (128, 128))
        self.assertEqual(item['original_hw'], (10, 10))
        torch.testing.assert_close(item['boxes'], torch.tensor([[0., 0., 128., 128.]]))
        self.assertTrue(item['development_only'])
        self.assertFalse(item['has_ignore_regions'])
        self.assertEqual(item['labels'].dtype, torch.int64)
        self.assertTrue(item['provenance']['payload_sha256_verified'])

    def test_hash_mismatch_and_size_mismatch(self):
        original = self.manifest['datasets'][0]['samples'][0]['files'][0]['sha256']
        self.manifest['datasets'][0]['samples'][0]['files'][0]['sha256'] = '0' * 64
        self.write_manifest()
        with self.assertRaises(DataIntegrityError):
            DevelopmentSampleDataset(self.manifest_path)
        self.manifest['datasets'][0]['samples'][0]['files'][0]['sha256'] = original
        self.manifest['datasets'][0]['samples'][0]['files'][0]['bytes'] += 1
        self.write_manifest()
        with self.assertRaises(DataIntegrityError):
            DevelopmentSampleDataset(self.manifest_path)

    def test_mutation_after_initial_verification_detected(self):
        dataset = DevelopmentSampleDataset(self.manifest_path, image_size=128)
        self.annotation_path.write_bytes(self.annotation_path.read_bytes() + b' ')
        with self.assertRaises(DataIntegrityError):
            dataset[0]

    def test_path_traversal_drive_and_alternate_stream(self):
        for path in ('../secret.txt', '/etc/passwd', 'C:\\temp\\x.jpg', 'C:relative.jpg',
                     'NEU/images/sample.jpg:stream', 'NEU/../sample.jpg'):
            with self.subTest(path=path), self.assertRaises(DataIntegrityError):
                safe_payload_path(self.root, path)

    def test_duplicate_identity_missing_payload_and_manifest_hash(self):
        with self.assertRaises(DataIntegrityError):
            DevelopmentSampleDataset(self.manifest_path, expected_manifest_sha256='0' * 64)
        duplicate = deepcopy(self.manifest['datasets'][0]['samples'][0])
        self.manifest['datasets'][0]['samples'].append(duplicate)
        self.manifest['datasets'][0]['sample_count'] = 2
        self.write_manifest()
        with self.assertRaises(DataIntegrityError):
            DevelopmentSampleDataset(self.manifest_path)
        self.manifest['datasets'][0]['samples'].pop()
        self.manifest['datasets'][0]['sample_count'] = 1
        self.manifest['datasets'][0]['samples'][0]['files'][0]['relative_path'] = 'NEU/images/missing.jpg'
        self.write_manifest()
        with self.assertRaises(DataIntegrityError):
            DevelopmentSampleDataset(self.manifest_path)

    def test_ignore_mask_and_current_stage_no_loss_policy(self):
        self.annotation_path.write_bytes(voc_xml(boxes=[('crazing', [1, 1, 5, 5], 1)]))
        entry = self.manifest['datasets'][0]['samples'][0]['files'][1]
        content = self.annotation_path.read_bytes()
        entry.update(bytes=len(content), sha256=hashlib.sha256(content).hexdigest())
        self.write_manifest()
        item = DevelopmentSampleDataset(self.manifest_path, image_size=10)[0]
        self.assertTrue(item['has_ignore_regions'])
        self.assertEqual(tuple(item['boxes'].shape), (0, 4))
        self.assertEqual(int(item['ignore_mask'].sum()), 25)
        self.assertIn('loss=None', item['current_stage_loss_policy'])
        self.assertTrue(item['geometric_masks_are_not_training_background_labels'])


if __name__ == '__main__':
    unittest.main(verbosity=2)
