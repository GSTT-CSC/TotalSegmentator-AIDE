import os
import shutil
import subprocess
import nibabel as nib
import numpy as np

from pathlib import Path


class TestDcm2NiiConversion:
    def setup_method(self):
        self.dcm_ref_path = Path('data/dcm2nii/dcm/')
        self.nii_ref_file = Path('data/dcm2nii/nii/ct-test-data.nii.gz')
        self.nii_gen_path = Path('data/dcm2nii/nii_generated/')

    def test_load_nifti(self):
        nii_obj = nib.load(self.nii_ref_file)
        assert issubclass(type(nii_obj), nib.nifti1.Nifti1Image)

    def test_dcm2niix(self):
        os.makedirs(self.nii_gen_path, exist_ok=True)
        subprocess.run(["dcm2niix", "-z", "y", "-o", self.nii_gen_path, "-f", "generated-ct-test-data",
                        self.dcm_ref_path])
        img_ref = nib.load(self.nii_ref_file).get_fdata()
        img_gen = nib.load(self.nii_gen_path / 'generated-ct-test-data.nii.gz').get_fdata()
        images_equal = np.array_equal(img_ref, img_gen)
        assert images_equal, f"Generated image array not equal to reference image array"

        # remove test generated images
        shutil.rmtree(self.nii_gen_path)