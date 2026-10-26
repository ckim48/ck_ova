import copy
import unittest
from gallery import prepare_plan


class GalleryCoverageTest(unittest.TestCase):
    def test_missing_request_image_is_added_without_changing_training(self):
        requests = [dict(id=f'r{i}', burst=i//5, user=i%5, images=[1000+i], required_steps=5) for i in range(45)]
        source = dict(gallery_images=[9, 10, 1000], validation_images=[9, 10],
                      forgotten_images=list(range(1000, 1045)), partitions=[[1, 2], [1000, 1001]])
        before = copy.deepcopy(source)
        fixed, audit = prepare_plan(source, requests)
        self.assertEqual(source, before)
        self.assertEqual(fixed['partitions'], source['partitions'])
        self.assertEqual(audit['previously_missing_count'], 44)
        self.assertEqual(len(fixed['gallery_images']), 47)
        self.assertIn(1044, fixed['gallery_images'])

    def test_changed_request_cohort_is_rejected(self):
        requests = [dict(id=f'r{i}', burst=i//5, user=i%5, images=[1000+i], required_steps=5) for i in range(45)]
        source = dict(gallery_images=[9], validation_images=[9], forgotten_images=[1000], partitions=[])
        with self.assertRaisesRegex(AssertionError, 'cohort'):
            prepare_plan(source, requests)


if __name__ == '__main__':
    unittest.main()
