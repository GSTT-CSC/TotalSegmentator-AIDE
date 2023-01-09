# Perform automatic segmentation of 104 regions on CT imaging data with TotalSegmentator

import logging
import os
import subprocess

import monai.deploy.core as md
from monai.deploy.core import DataPath, ExecutionContext, InputContext, IOType, Operator, OutputContext


@md.input("input_files", DataPath, IOType.DISK)
@md.output("input_files", DataPath, IOType.DISK)
@md.env(pip_packages=["pydicom >= 2.3.0", "highdicom >= 0.18.2"])
class TotalSegmentatorOperator(Operator):
    """
    TotalSegmentator Operator - perform segmentation on CT imaging data saved as NIFTI file
    """

    def compute(self, op_input: InputContext, op_output: OutputContext, context: ExecutionContext):

        logging.info(f"Begin {self.compute.__name__}")

        input_path = op_input.get("input_files").path
        nii_input_file = input_path / "input-ct-dataset.nii.gz"

        if not os.path.exists(nii_input_file):
            NameError('Exception occurred with nii_input_file')
        else:
            logging.info(f"Found nii_input_file: {nii_input_file}")

        # Create TotalSegmentator output directory
        nii_output_path = os.path.join(input_path, "nii_output")
        if not os.path.exists(nii_output_path):
            os.makedirs(nii_output_path)

        # Run TotalSegmentator
        subprocess.run(["TotalSegmentator", "-i", nii_input_file, "-o", nii_output_path])

        logging.info(f"Performed TotalSegmentator processing")

        # Set output path for next operator
        op_output.set(DataPath(input_path))  # cludge to avoid op_output not exist error
        op_output_folder_path = op_output.get().path
        op_output_folder_path.mkdir(parents=True, exist_ok=True)

        logging.info(f"End {self.compute.__name__}")
