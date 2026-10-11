"""Temporary synthetic unit tests for the convention-neutral read-only NEU audit."""
from pathlib import Path
import tempfile
import unittest

from PIL import Image

from audit_neu import (candidate_valid, candidate_yolo, compare_yolo_to_xml,
                       duplicate_groups, inspect_image, parse_class_names,
                       parse_voc_audit, parse_yolo_labels)


def xml(raw, filename='crazing_1.jpg', difficult='0'):
    values = ''.join(f'<{tag}>{value}</{tag}>' for tag, value in zip(('xmin', 'ymin', 'xmax', 'ymax'), raw))
    return (f'<annotation><filename>{filename}</filename><size><width>200</width><height>200</height></size>'
            f'<object><name>crazing</name><difficult>{difficult}</difficult><bndbox>{values}</bndbox></object></annotation>').encode()


class NeuAuditTests(unittest.TestCase):
    def test_zero_coordinates_are_facts_not_rejected(self):
        result = parse_voc_audit(xml([0, 0, 200, 200]), (200, 200))
        self.assertEqual(result['parse_errors'], [])
        obj = result['objects'][0]
        self.assertTrue(obj['raw_facts']['xmin_equals_zero'])
        self.assertTrue(obj['raw_facts']['ymin_equals_zero'])
        self.assertTrue(obj['candidate_bounds_valid']['raw_zero_based_half_open'])
        self.assertFalse(obj['candidate_bounds_valid']['one_based_inclusive'])

    def test_missing_suffix_and_difficult_preserved(self):
        result = parse_voc_audit(xml([1, 1, 199, 199], filename='crazing_1', difficult='1'))
        self.assertTrue(result['filename_missing_suffix'])
        self.assertEqual(result['objects'][0]['difficult_text'], '1')

    def test_candidates_and_exact_yolo_difference(self):
        raw = [10, 20, 100, 120]
        direct = candidate_yolo(raw, 200, 200, 'raw_zero_based_half_open')
        voc = candidate_yolo(raw, 200, 200, 'one_based_inclusive')
        self.assertAlmostEqual(direct[0] - voc[0], .0025)
        self.assertAlmostEqual(voc[2] - direct[2], .005)
        label = ('0 ' + ' '.join(f'{value:.6f}' for value in direct)).encode()
        result = compare_yolo_to_xml(parse_voc_audit(xml(raw)), parse_yolo_labels(label, 1), ['crazing'], 200, 200)
        self.assertTrue(result['candidates']['raw_zero_based_half_open']['matches_within_tolerance'])
        self.assertFalse(result['candidates']['one_based_inclusive']['matches_within_tolerance'])

    def test_malformed_xml_and_inversion_recorded(self):
        self.assertTrue(parse_voc_audit(b'<broken>')['parse_errors'])
        obj = parse_voc_audit(xml([10, 20, 5, 15]))['objects'][0]
        self.assertTrue(obj['raw_facts']['x_inverted'])
        self.assertFalse(any(obj['candidate_bounds_valid'].values()))

    def test_yolo_errors_and_count_mismatch(self):
        self.assertTrue(parse_yolo_labels(b'0 nan .2 .3 .4', 1)['errors'])
        self.assertTrue(parse_yolo_labels(b'1 .2 .2 .3 .4', 1)['errors'])
        result = compare_yolo_to_xml(parse_voc_audit(xml([1, 1, 20, 20])), parse_yolo_labels(b'', 1), ['crazing'], 200, 200)
        self.assertFalse(result['comparison_available'])

    def test_yaml_current_list(self):
        self.assertEqual(parse_class_names('path: old\nnames:\n  - "crazing"\n  - "inclusion"\n'), ['crazing', 'inclusion'])

    def test_decoded_pixels_vs_file_bytes(self):
        with tempfile.TemporaryDirectory(prefix='neu_audit_pixels_') as directory:
            root = Path(directory)
            original = Image.new('RGB', (20, 20), (0, 127, 255))
            original.save(root / 'a.png', compress_level=0)
            original.save(root / 'b.png', compress_level=9)
            first, second = inspect_image(root / 'a.png'), inspect_image(root / 'b.png')
            self.assertNotEqual(first['sha256'], second['sha256'])
            self.assertEqual(first['decoded_rgb_sha256'], second['decoded_rgb_sha256'])

    def test_duplicate_cross_class_cross_split(self):
        samples = [{'sample_id': 'a', 'source_class_directory': 'crazing', 'source_split': 'train',
                    'existing_split_memberships': ['train'], 'image': {'sha256': 'same', 'path': 'a'}},
                   {'sample_id': 'b', 'source_class_directory': 'inclusion', 'source_split': 'val',
                    'existing_split_memberships': ['val'], 'image': {'sha256': 'same', 'path': 'b'}}]
        groups = duplicate_groups(samples, 'sha256')
        self.assertEqual(len(groups), 1)
        self.assertTrue(groups[0]['cross_class'])
        self.assertTrue(groups[0]['cross_train_val'])


if __name__ == '__main__':
    unittest.main(verbosity=2)
