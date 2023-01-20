import os
import shutil
import unittest
import subprocess
import nibabel as nib
import numpy as np
from rt_utils import RTStructBuilder

from pathlib import Path


class TestNii2RTStruct(unittest.TestCase):

    def setUp(self):
        self.nii_ref_path = Path('tests/data/rtstructwriter/nii/liver-test-seg.nii.gz')
        self.rtstruct_ref_file = Path('tests/data/rtstructwriter/dcm/liver-test-rtstruct.dcm')
        self.rtstruct_gen_path = Path('tests/data/rtstructwriter/rtstruct_generated/')
        self.rtstruct_gen_filename = Path('generated-liver-test-rtstruct.dcm')

        # TODO: use/create small FOV/lightweight CT dataset for test purposes (e.g. < 10MB)

    def tearDown(self):
        shutil.rmtree(self.rtstruct_gen_path)

    def test_nii2rtstruct(self):
        os.makedirs(self.rtstruct_gen_path, exist_ok=True)

        # TODO: need CT image .dcm dataset to initialise the rtstruct object
        rtstruct = RTStructBuilder.create_new(dicom_series_path=dcm_input_path)

        nii = nib.load(self.nii_ref_path)
        nii_img = nii.get_fdata().astype("uint16").astype("bool")
        nii_img = np.rot90(nii_img, 1, (0, 1))  # rotate nii to match DICOM orientation
        rtstruct.add_roi(mask=nii_img, name='liver-test-seg')
        rtstruct.save(os.path.join(self.rtstruct_gen_path, self.rtstruct_gen_filename))

        # TODO: break this down into smaller unit tests:
        #  - initialised rtstruct object
        #  - generated nii_img == liver-test-seg nii_img
        #  - generated rtstruct == liver-test-rtstruct.dcm (which I still need to create)
        #  - test generated-rtstruct.dcm is legit DICOM (check tags etc.)


if __name__ == '__main__':
    unittest.main()
