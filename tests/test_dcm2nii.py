import os
import shutil
import nibabel as nib
import numpy as np
from pathlib import Path

from monai.deploy.operators import DICOMDataLoaderOperator
from monai.deploy.operators import DICOMSeriesSelectorOperator
from app.operators.dcm2nii_operator import Dcm2NiiOperator


class TestDcm2NiiConversion:
    def setup_method(self):
        self.dcm_ref_path = Path('tests/data/dcm2nii/dcm/')
        self.nii_ref_file = Path('tests/data/dcm2nii/nii/ct-test-data.nii.gz')
        self.nii_gen_path = Path('tests/data/dcm2nii/nii_generated/')
        self.nii_gen_filename = "generated-ct-test-data"

        # create monai_workdir as already exists within a MAP
        self.monai_workdir = Path('tests/data/dcm2nii/monai_workdir')
        os.makedirs(self.monai_workdir, exist_ok=True)

        # setup MONAI Deploy loader and selector operators for DICOM testing
        current_file_dir = Path(__file__).parent.resolve()
        self.dcm_ref_path_abs = current_file_dir.joinpath("../").joinpath(self.dcm_ref_path).absolute()  # absolute path
        loader = DICOMDataLoaderOperator()
        selector = DICOMSeriesSelectorOperator(all_matched=True)
        study_list = loader.load_data_to_studies(self.dcm_ref_path_abs)
        study_selected_series_list = selector.filter('', study_list)
        self.study_selected_series = study_selected_series_list[0]  # select first Series in Study

    def test_create_dcm_input_dir(self):
        Dcm2NiiOperator.create_dir(Path(self.monai_workdir / 'dcm_input_dir'))
        assert Path(self.monai_workdir / 'dcm_input_dir').is_dir()

    def test_create_nii_ct_dataset_dir(self):
        Dcm2NiiOperator.create_dir(Path(self.monai_workdir / 'nii_ct_dataset'))
        assert Path(self.monai_workdir / 'nii_ct_dataset').is_dir()

    def test_load_selected_series(self):
        selected_series, num_instances_in_series = Dcm2NiiOperator.load_selected_series(self.study_selected_series)

        # assert assuming number of DICOM instances loaded = number of files in test directory
        num_files_in_dir = len([f for f in os.listdir(self.dcm_ref_path_abs) if not f.startswith('.')])
        assert num_instances_in_series == num_files_in_dir

    def test_copy_dcm_to_workdir(self):
        Dcm2NiiOperator.create_dir(Path(self.monai_workdir / 'dcm_input_dir'))
        selected_series, num_instances_in_series = Dcm2NiiOperator.load_selected_series(self.study_selected_series)
        Dcm2NiiOperator.copy_dcm_to_workdir(selected_series, num_instances_in_series,
                                            Path(self.monai_workdir), 'dcm_input_dir')

        # assert number of DICOM files copied to monai_workdir = number in test dataset
        num_files_in_dir = len([f for f in os.listdir(Path(self.monai_workdir / 'dcm_input_dir'))
                                if not f.startswith('.')])
        assert num_instances_in_series == num_files_in_dir

    def test_dcm2niix(self):
        os.makedirs(self.nii_gen_path, exist_ok=True)
        Dcm2NiiOperator.run_dcm2niix(
            self.dcm_ref_path,
            self.nii_gen_path,
            self.nii_gen_filename
        )
        img_ref = nib.load(self.nii_ref_file).get_fdata()
        img_gen = nib.load((self.nii_gen_path / self.nii_gen_filename).with_suffix('.nii.gz')).get_fdata()
        images_equal = np.array_equal(img_ref, img_gen)
        assert images_equal, f"Generated image array not equal to reference image array"

        # remove test generated images
        shutil.rmtree(self.nii_gen_path)

    def teardown_method(self):
        # remove test monai_workdir structure
        shutil.rmtree(self.monai_workdir)