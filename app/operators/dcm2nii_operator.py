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

import logging
import os
import shutil
import subprocess
from typing import List
import monai.deploy.core as md
from monai.deploy.core import DataPath, ExecutionContext, InputContext, IOType, Operator, OutputContext
from monai.deploy.core.domain.dicom_series_selection import StudySelectedSeries


@md.input("study_selected_series_list", List[StudySelectedSeries], IOType.IN_MEMORY)
@md.output("nii_ct_dataset", DataPath, IOType.DISK)
@md.output("dcm_input", DataPath, IOType.DISK)
@md.env(pip_packages=["pydicom >= 2.3.0", "highdicom >= 0.18.2"])
class Dcm2NiiOperator(Operator):
    """
    DICOM to NIfTI Operator
    """

    def compute(self, op_input: InputContext, op_output: OutputContext, context: ExecutionContext):

        logging.info(f"Begin {self.compute.__name__}")

        workdir = os.getcwd()

        # Copy .dcm files from input/ to monai_workdir/ for dcm2niix
        # We use dcm2niix to generate the NIfTI files required as input to TotalSegmentator. For robustness and ease of
        # processing with dcm2niix, we copy the .dcm files to a clean folder within the ephemeral monai_workdir/. We use
        # the DICOMSeriesSelectorOperator to return the absolute paths to the input/ .dcm files which are then copied
        # to the monai_workdir/. dcm2niix is performed on the files in the monai_workdir. This has the additional
        # benefit that the input/ directory is not manipulated.

        # create dcm_input_dir within monai_workdir
        dcm_input_dir = 'dcm_input'
        if not os.path.exists(dcm_input_dir):
            os.mkdir(dcm_input_dir)

        # create output directory for input-ct-dataset.nii.gz within monai_workdir
        nii_ct_dataset_dirname = 'nii_ct_dataset'
        if not os.path.exists(nii_ct_dataset_dirname):
            os.makedirs(nii_ct_dataset_dirname)

        # copy across .dcm files - assumption: single DICOM series
        study_selected_series = op_input.get("study_selected_series_list")[0]
        selected_series = study_selected_series.selected_series
        num_instances_in_series = len(selected_series[0].series.get_sop_instances())

        for idx, f in enumerate(selected_series[0].series.get_sop_instances()):
            dcm_filepath = f._sop.filename
            if str(dcm_filepath).lower().endswith('.dcm'):
                logging.info(f"Copying DICOM Instance: {idx+1}/{num_instances_in_series} ...")
                destination_path = shutil.copy2(dcm_filepath, os.path.join(workdir, dcm_input_dir))
                logging.info(f"Copied {dcm_filepath} to {destination_path}")

        # run dcm2niix
        # TODO: check Eq_ files output by dcm2niix
        # See here: https://github.com/rordenlab/dcm2niix/issues/119
        # Potential for CT images to have non-equidistant slices
        nii_ct_filename = 'input-ct-dataset'  # nb: .nii.gz suffix should be omitted for dcm2niix -f option

        logging.info(f"Performing dcm2niix ...")
        subprocess.run(["dcm2niix", "-z", "y", "-b", "n", "-o", nii_ct_dataset_dirname, "-f", nii_ct_filename, dcm_input_dir])

        # set output path for next operator
        op_output.set(DataPath(os.path.join(nii_ct_dataset_dirname, nii_ct_filename + '.nii.gz')), 'nii_ct_dataset')
        op_output.set(DataPath(dcm_input_dir), 'dcm_input')

        logging.info("Performed dcm2niix conversion.")
        logging.info(f"End {self.compute.__name__}")
