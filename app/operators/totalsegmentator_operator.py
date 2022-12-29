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

        nii_path = op_input.get("input_files").path

        input_dir = os.path.dirname(nii_path)

        # TODO: make pathing consistent with previous operator
        # nii_3d_path = os.path.join(input_dir, "nii_3d")
        # if not os.path.exists(nii_3d_path):
        #     os.makedirs(nii_3d_path)
        op_output.set(DataPath(input_dir))  # cludge to avoid op_output not exist error

        op_output_folder_path = op_output.get().path
        op_output_folder_path.mkdir(parents=True, exist_ok=True)

        # TODO: setup TotalSegmentator to execute with subprocess
        # Run TotalSegmentator
        subprocess.run(["TotalSegmentator", "-i", nii_path, "-o", "output"])

        logging.info(f"End {self.compute.__name__}")
