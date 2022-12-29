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

        pass

        # TODO:
        #  - implement nii2dcm to convert .nii.gz labels into DICOMs files that match original
        #  - OR: use the existing label tools within MONAI Deploy

        logging.info(f"End {self.compute.__name__}")