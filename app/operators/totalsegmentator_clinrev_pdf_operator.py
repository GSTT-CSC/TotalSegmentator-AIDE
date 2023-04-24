# Clinical Review PDF generator
from monai.deploy.core import DataPath, ExecutionContext, InputContext, IOType, Operator, OutputContext
import logging
import os.path
from os import listdir
from os.path import isfile, join

@md.intput("dicom_files", DataPath, IOType.DISK)
@md.output("pdf_file"), DataPath, IOType.DISK)
class ClinicalReviewPDFGenerator(Operator):
    """Generates a DICOM encapssulated PDF with Sag/Cor/Ax views for each structure"""

    def compute(self):
        logging.info(f"Begin {self.compute.__name__}")

        # open RTStruct file

        # open image

        #for each structure get largest pro

        logging.info(f"Creating PDF  ...")


        logging.info(f"Dicom Enapcsualted PDF written to {os.path.join(dcm_output_path, rt_struct_output_filename)}")
        logging.info(f"PDF creation complete ...")
        logging.info(f"End {self.compute.__name__}")
        return

