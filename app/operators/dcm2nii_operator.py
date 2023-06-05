# DICOM to NIfTI conversion using dcm2niix
#
# This operator uses Chris Rorden's dcm2niix (https://github.com/rordenlab/dcm2niix/) to convert a DICOM Series to a
# single .nii.gz file. The MONAI Deploy DICOMSeriesSelectorOperator is used to access the .dcm filepaths in the
# /var/monai/input/ folder within a MAP. The .dcm files are copied to an output operator folder within the
# monai_workdir/. Here, dcm2niix is performed just on the .dcm files contained in this folder. The NIfTI file it output
# in a separate folder called nii_ct_dataset, which can imported by subsequent operators
# (i.e. totalsegmentator_operator.py).
#
# AIDE stores files within a MiniO bucket. The MiniO directory structure is:
# PayloadID (folder)
#   dcm (folder)
#       StudyUID (folder)
#           SeriesUID (folders)
#               InstanceUID (files)
#
# The 'dcm' folder is mounted inside the MAP 'input' folder, e.g.:
# /var/monai/input/ (folder)
#   StudyUID (folder)
#       SeriesUID (folders)
#           InstanceUID (files)
#

import glob
import logging
import os
import shutil
import subprocess
from pathlib import Path
import monai.deploy.core as md
from monai.deploy.core import DataPath, ExecutionContext, InputContext, IOType, Operator, OutputContext
from typing import Union

@md.input("input_files", DataPath, IOType.DISK)
@md.output("nii_ct_dataset", DataPath, IOType.DISK)
@md.output("dcm_input", DataPath, IOType.DISK)
@md.env(pip_packages=["pydicom >= 2.3.0", "highdicom >= 0.18.2"])
class Dcm2NiiOperator(Operator):
    """
    DICOM to NIfTI Operator
    """

    def compute(self, op_input: InputContext, op_output: OutputContext, context: ExecutionContext):

        logging.info(f"Begin {self.compute.__name__}")

        input_path = op_input.get("input_files").path
        workdir = os.getcwd()

        # get list of DICOMs in input directory
        input_files = sorted(os.listdir(str(input_path)))  # assumes .dcm in input/

        # TODO: establish if parse_recursively_dcm_files required
        # This function is a relic of the fetal app, which contained multiple DICOM series.
        # CT data should only be single series, therefore this is probably surplus to requirements.
        # Leaving this as a reminder for removal or integration with future release.
        # input_files = parse_recursively_dcm_files(str(input_path))  # assumes AIDE MinIO structure

        # move input DICOM files into .monai_workdir operators folder
        dcm_input_dir = 'dcm_input'
        if not os.path.exists(dcm_input_dir):
            os.mkdir(dcm_input_dir)

        for f in input_files:
            if str(os.path.join(input_path, f)).lower().endswith('.dcm'):
                shutil.copyfile(os.path.join(input_path, f), os.path.join(workdir, dcm_input_dir, f))

        # create output directory for input-ct-dataset.nii.gz
        nii_ct_dataset_dirname = 'nii_ct_dataset'
        if not os.path.exists(nii_ct_dataset_dirname):
            os.makedirs(nii_ct_dataset_dirname)

        # ---------
        # Local testing - copy input-ct-dataset.nii.gz from another folder, e.g. local_files in repo root
        # (hardcode this yourself)

        # shutil.copyfile('../../local_files/input-ct-dataset.nii.gz',
        #                os.path.join(nii_ct_dataset_dirname, nii_ct_filename)
        # ---------

        nii_ct_filename = 'input-ct-dataset'  # nb: .nii.gz suffix should be omitted for dcm2niix -f option

        # run dcm2niix
        # TODO: check Eq_ files output by dcm2niix
        # See here: https://github.com/rordenlab/dcm2niix/issues/119
        # Potential for CT images to have non-equidistant slices
        subprocess.run(["dcm2niix", "-z", "y", "-b", "n", "-o", nii_ct_dataset_dirname, "-f", nii_ct_filename, dcm_input_dir])

        # set output path for next operator
        op_output.set(DataPath(os.path.join(nii_ct_dataset_dirname, nii_ct_filename + '.nii.gz')), 'nii_ct_dataset')
        op_output.set(DataPath(dcm_input_dir), 'dcm_input')

        logging.info("Performed dcm2niix conversion")
        logging.info(f"End {self.compute.__name__}")


def parse_recursively_dcm_files(input_path):
    """
    Recursively parse Minio folder structure to extract paths to .dcm files
    Minio file structure:
    /var/monai/input
        StudyUID (folder)
            SeriesUID (folders)
                InstanceUID (files)

    :param input_path:
    :return dcm_paths:
    """

    logging.info(f"input_path: {os.getcwd()}")
    logging.info(f"listdir(input_path): {os.listdir(input_path)}")

    for item in os.listdir(input_path):
        item = os.path.join(input_path, item)
        if os.path.isdir(item):
            study_path = item
        else:
            NameError('Exception occurred with study_path')

    logging.info(f"study_path: {study_path}")

    try:
        series_paths = []
        series_dirs = os.listdir(study_path)
        for sd in series_dirs:
            series_paths.append(os.path.join(study_path, sd))
    except:
        print('Exception occurred with series_paths')

    logging.info(f"series_paths: {series_paths}")

    dcm_files = []
    for sp in series_paths:
        series_files = os.listdir(sp)
        for file in series_files:
            if '.dcm' in Path(file).suffix:
                dcm_files.append(file)

    dcm_paths = [os.path.join(a, b) for a, b in zip(series_paths, dcm_files)]

    logging.info(f"dcm_paths: {dcm_paths}")

    return dcm_paths
