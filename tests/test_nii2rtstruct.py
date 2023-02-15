import os
import glob
import shutil
from pathlib import Path

import unittest
import numpy as np

from rt_utils import RTStructBuilder
from app.operators.rtstructwriter_operator import add_nii_roi_to_rtstruct, list_nii_files


class TestNii2RTStruct(unittest.TestCase):

    def setUp(self):
        # test subset: public chest CT dataset, slices 81-100
        self.dcm_ref_path = Path('tests/data/rtstructwriter/dcm/')
        self.nii_ref_file = Path('tests/data/rtstructwriter/nii/ct-test-data-81-100.nii.gz')
        self.nii_seg_ref_path = Path('tests/data/rtstructwriter/nii_seg')
        self.rtstruct_ref_file = Path('tests/data/rtstructwriter/rtstruct/rt-struct-test-data-81-100.dcm')
        self.rtstruct_gen_path = Path('tests/data/rtstructwriter/rtstruct_generated/')
        self.rtstruct_gen_filename = Path('generated-rtstruct-test-data.dcm')

        os.makedirs(self.rtstruct_gen_path, exist_ok=True)

    def tearDown(self):
        shutil.rmtree(self.rtstruct_gen_path)

    def test_list_nii_files(self):
        nii_seg_files = list_nii_files(self.nii_seg_ref_path)
        num_nii_files_in_test_dir = len(glob.glob(str(self.nii_seg_ref_path) + '/*.nii*'))
        num_files_in_list_nii_files = len(nii_seg_files)
        self.assertEqual(num_nii_files_in_test_dir, num_files_in_list_nii_files,
                         f"Number of NIfTI files in list does not equal number of NIfTI files in directory")

    def test_nii2rtstruct(self):
        nii_seg_files = list_nii_files(self.nii_seg_ref_path)

        # instantiate RT Struct object
        rtstruct_gen = RTStructBuilder.create_new(dicom_series_path=str(self.dcm_ref_path))

        # add segmentations to RT Struct
        for idx, filename in enumerate(nii_seg_files):
            add_nii_roi_to_rtstruct(self.nii_seg_ref_path, filename, rtstruct_gen)

        # save RT Struct
        rtstruct_gen.save(os.path.join(self.rtstruct_gen_path, self.rtstruct_gen_filename))

        # load test RTStruct
        rtstruct_ref = RTStructBuilder.create_from(
            dicom_series_path=str(self.dcm_ref_path),
            rt_struct_path=str(self.rtstruct_ref_file)
        )

        # compare reference and generated contours based on pixel arrays
        for seg_name in ['aorta', 'heart_myocardium', 'lung_lower_lobe_right']:
            seg_ref = rtstruct_ref.get_roi_mask_by_name(seg_name)
            seg_gen = rtstruct_gen.get_roi_mask_by_name(seg_name)
            segs_equal = np.array_equal(seg_ref, seg_gen)
            self.assertTrue(segs_equal, f"{seg_name} contour images not equal")

        # TODO: add in more RTStruct tests, e.g.
        #  - DICOM file tests
        #  - DICOM metadata tests


if __name__ == '__main__':
    unittest.main()
