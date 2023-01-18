# RT Struct Writer - converts TotalSegmentator NIfTI segmentations to DICOM RT Struct format

import logging
import os.path
from os import listdir
from os.path import isfile, join

import nibabel as nib
import numpy as np
from rt_utils import RTStructBuilder

import monai.deploy.core as md
from monai.deploy.core import DataPath, ExecutionContext, InputContext, IOType, Operator, OutputContext


@md.input("input_files", DataPath, IOType.DISK)
@md.output("dicom_files", DataPath, IOType.DISK)
@md.env(pip_packages=["pydicom >= 2.3.0", "highdicom >= 0.18.2"])
class RTStructWriterOperator(Operator):
    """
    RTStructWriterOperator - converts TotalSegmentator NIfTI segmentations to DICOM RT Struct format
    """

    def compute(self, op_input: InputContext, op_output: OutputContext, context: ExecutionContext):

        logging.info(f"Begin {self.compute.__name__}")

        input_path = op_input.get("input_files").path
        dcm_input_path = input_path / 'dcm_input'
        nii_seg_output_path = input_path / 'nii_seg_output'
        dcm_output_path = op_output.get().path
        rt_struct_output_filename = 'output-rt-struct.dcm'

        logging.info(f"Creating RT Struct ...")

        # create new RT Struct - requires original DICOM
        rtstruct = RTStructBuilder.create_new(dicom_series_path=dcm_input_path)

        nii_seg_files = [f for f in listdir(nii_seg_output_path) if
                         isfile(join(nii_seg_output_path, f)) and '.nii' in f]

        # add segmentations to RT Struct
        for idx, filename in enumerate(nii_seg_files):

            if '.nii.gz' in filename:
                seg_name = filename.replace('.nii.gz', '')
            elif '.nii' in filename:
                seg_name = filename.replace('.nii', '')
            else:
                NameError('Suffix of segmentation file is not .nii or .nii.gz.')

            nii = nib.load(os.path.join(nii_seg_output_path, filename))
            nii_img = nii.get_fdata().astype("uint16").astype("bool")  # match to DICOM datatype, convert to boolean

            # TODO: check if uint16 conversion is always necessary - may cause unwanted issues

            # rotate nii to match DICOM orientation
            nii_img = np.rot90(nii_img, 1, (0, 1))  # rotate segmentation in-plane

            # log empty segmentations
            if np.sum(nii_img) > 0:
                logging.info(f"RT Struct - added segmentation: {seg_name}")
            elif np.sum(nii_img) == 0:
                logging.info(f"Empty segmentation: {seg_name}")

            # add segmentation to RT Struct
            rtstruct.add_roi(
                mask=nii_img,
                name=seg_name
            )

            # TODO: in-plane rotation above not ideal.
            #  Need to test with different datasets to understand extent of issue.
            #  This problem stems from different packages manipulating the imaging data in different ways.
            #  We have:
            #  - original DICOMs
            #  - NIfTI files output by TotalSegmentator
            #  - RTStructBuilder object created by rt_utils
            #  In testing, without the rotation below, the final rtstruct does not align with the original DICOMs.

        # save RT Struct
        rtstruct.save(os.path.join(dcm_output_path, rt_struct_output_filename))
        logging.info(f"RT Struct written to {os.path.join(dcm_output_path, rt_struct_output_filename)}")

        logging.info(f"RT Struct creation complete ...")

        logging.info(f"End {self.compute.__name__}")