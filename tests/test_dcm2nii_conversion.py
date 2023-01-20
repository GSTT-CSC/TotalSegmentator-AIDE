import os
import shutil
import unittest
import subprocess
import nibabel as nib
import numpy as np

from pathlib import Path


class TestDcm2NiiConversion(unittest.TestCase):

    def setUp(self):
        self.dcm_ref_path = Path('tests/data/dcm/')
        self.nii_ref_file = Path('tests/data/nii/ct-test-data.nii.gz')
        self.nii_gen_path = Path('tests/data/nii_generated/')

    def tearDown(self):
        shutil.rmtree(self.nii_gen_path)

    def test_dcm2niix(self):
        os.makedirs(self.nii_gen_path, exist_ok=True)
        subprocess.run(["dcm2niix", "-z", "y", "-o", self.nii_gen_path, "-f", "generated-ct-test-data",
                        self.dcm_ref_path])
        img_ref = nib.load(self.nii_ref_file).get_fdata()
        img_gen = nib.load(self.nii_gen_path / 'generated-ct-test-data.nii.gz').get_fdata()
        images_equal = np.array_equal(img_ref, img_gen)
        self.assertTrue(images_equal, "dcm2niix conversion issue – reference and generated nii files not equal")


if __name__ == '__main__':
    unittest.main()
