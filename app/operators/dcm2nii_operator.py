# DICOM to NIfTI conversion operator

import glob
import logging
import os
import shutil
import subprocess
from pathlib import Path

import monai.deploy.core as md
from monai.deploy.core import DataPath, ExecutionContext, InputContext, IOType, Operator, OutputContext

@md.input("input_files", DataPath, IOType.DISK)
@md.output("input_files", DataPath, IOType.DISK)
@md.env(pip_packages=["pydicom >= 2.3.0", "highdicom >= 0.18.2"])
class Dcm2NiiOperator(Operator):
    """
    DICOM to NIfTI Operator
    """

    def compute(self, op_input: InputContext, op_output: OutputContext, context: ExecutionContext):

        logging.info(f"Begin {self.compute.__name__}")

        input_path = str(op_input.get("input_files").path)

        # Set local_testing = True if doing local testing
        local_testing = True
        if local_testing:
            input_files = sorted(os.listdir(input_path))  # assumes .dcm in input/
        else:
            input_files = parse_recursively_dcm_files(input_path)  # assumes AIDE MinIO structure

        # move input DICOM files for later
        dcm_input_path = os.path.join(input_path, 'dcm_input')
        if not os.path.exists(dcm_input_path):
            os.makedirs(dcm_input_path)

        for f in input_files:
            shutil.move(os.path.join(input_path, f), dcm_input_path)

        # create output directory for input-ct-dataset.nii.gz
        dcm_output_path = os.path.join(input_path, 'nii_ct_output')
        if not os.path.exists(dcm_output_path):
            os.makedirs(dcm_output_path)

        # Run dcm2niix
        subprocess.run(["dcm2niix", "-z", "y", "-o", dcm_output_path, "-f", "input-ct-dataset", dcm_output_path])

        # ---------
        # Local testing - copy input-ct-dataset.nii.gz from another folder, e.g. local_files in repo root
        # (hardcode this yourself)

        # shutil.copyfile('../../local_files/input-ct-dataset.nii.gz',
        #               os.path.join(dcm_output_path, 'input-ct-dataset.nii.gz'))
        # ---------

        # Delete superfluous .json files
        json_files = glob.glob(dcm_output_path + "/*.json")
        for json_file in json_files:
            os.remove(json_file)

        # Set output path for next operator
        op_output.set(DataPath(input_path))

        logging.info(f"Performed dcm2niix conversion")

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
