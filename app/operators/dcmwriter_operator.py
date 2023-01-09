# NII to DICOM Writer Operator

import logging

import monai.deploy.core as md
from monai.deploy.core import DataPath, ExecutionContext, InputContext, IOType, Operator, OutputContext


@md.input("input_files", DataPath, IOType.DISK)
@md.output("dicom_files", DataPath, IOType.DISK)
@md.env(pip_packages=["pydicom >= 2.3.0", "highdicom >= 0.18.2"])
class DicomWriterOperator(Operator):
    """
    DICOM writer
    """

    def compute(self, op_input: InputContext, op_output: OutputContext, context: ExecutionContext):

        logging.info(f"Begin {self.compute.__name__}")

        input_path = op_input.get("input_files").path

        nii_output_path = input_path / 'nii_output'

        dcm_output_path = op_output.get().path

        logging.info(f"WIP: Here is where nii2dcm will be performed on TotalSegmentator output.")

        pass

        # TODO:
        #  - implement nii2dcm to convert .nii.gz labels into DICOMs files that match original
        #  - OR: use the existing label tools within MONAI Deploy
        #  - something like nii2dcm(nii_output_path, dcm_output_path, 'CT')

        logging.info(f"End {self.compute.__name__}")