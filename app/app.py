# TotalSegmentator AIDE App
#
# TotalSegmentator is a tool for robust segmentation of 104 important anatomical structures in CT images.
# Website: https://github.com/wasserth/TotalSegmentator
#
# TotalSegmentator is distributed under the Apache 2.0 licence. The code in this app is not created by the original
# creators of TotalSegmentator.
#
# Tom Roberts (tom.roberts@gstt.nhs.uk / t.roberts@kcl.ac.uk)

import logging

from monai.deploy.core import Application

from operators.dcm2nii_operator import Dcm2NiiOperator
from operators.rtstructwriter_operator import RTStructWriterOperator
from operators.totalsegmentator_operator import TotalSegmentatorOperator


class TotalSegmentatorApp(Application):
    """
    TotalSegmentator - segmentation of 104 anatomical structures in CT images.
    """

    name = "totalsegmentator"
    description = "Robust segmentation of 104 anatomical structures in CT images"
    version = "0.1.1"

    def compose(self):
        """Operators go in here
        """

        logging.info(f"Begin {self.compose.__name__}")

        # DICOM to NIfTI operator
        dcm2nii_op = Dcm2NiiOperator()

        # TotalSegmentator segmentation
        totalsegmentator_op = TotalSegmentatorOperator()

        # RT Struct Writer operator
        custom_tags = {"SeriesDescription": "AI generated image, not for clinical use."}
        rtstructwriter_op = RTStructWriterOperator(custom_tags=custom_tags)

        # Operator pipeline
        self.add_flow(dcm2nii_op, totalsegmentator_op, {"input_files": "input_files"})
        self.add_flow(totalsegmentator_op, rtstructwriter_op, {"input_files": "input_files"})

        logging.info(f"End {self.compose.__name__}")


if __name__ == "__main__":
    TotalSegmentatorApp(do_run=True)