# RT Struct Writer operator
#
# This operator converts the TotalSegmentator output NIfTI segmentations to DICOM RTStruct format
#

import logging
import os.path
from os import listdir
from os.path import isfile, join
import nibabel as nib
import numpy as np
from rt_utils import RTStructBuilder
import monai.deploy.core as md
from monai.deploy.core import DataPath, ExecutionContext, InputContext, IOType, Operator, OutputContext


@md.input("nii_seg_output_path", DataPath, IOType.DISK)
@md.input("dcm_input", DataPath, IOType.DISK)
@md.output("dicom_files", DataPath, IOType.DISK)
@md.env(pip_packages=["pydicom >= 2.3.0", "rt-utils >= 1.2.7"])
class RTStructWriterOperator(Operator):
    """
    RTStructWriterOperator - converts TotalSegmentator NIfTI segmentations to DICOM RT Struct format
    """

    def compute(self, op_input: InputContext, op_output: OutputContext, context: ExecutionContext):

        logging.info(f"Begin {self.compute.__name__}")

        # Gather inputs
        dcm_input_path = op_input.get("dcm_input").path

        nii_seg_output_path = op_input.get("nii_seg_output_path").path
        nii_seg_files = list_nii_files(nii_seg_output_path)

        dcm_output_path = op_output.get().path
        rt_struct_output_filename = 'output-rt-struct.dcm'

        # create new RT Struct - requires original DICOM
        logging.info("Creating RT Struct ...")
        rtstruct = RTStructBuilder.create_new(dicom_series_path=dcm_input_path)

        # add TotalSegmentator segmentations to RT Struct
        for idx, filename in enumerate(nii_seg_files):
            add_nii_roi_to_rtstruct(nii_seg_output_path, filename, rtstruct)

        # round RT Struct ContourData to 10 d.p.
        # TODO remove once new PyPI release of rt-utils
        logging.info("Rounding ContourData values to 10 d.p. ...")

        # loop over ROIs in rtstruct (1 per TotalSegmentator region)
        for roi_idx in range(0, len(rtstruct.ds.ROIContourSequence)):
            # if ROI has contours
            if len(rtstruct.ds.ROIContourSequence[roi_idx].ContourSequence) > 0:
                # loop over the contours within the ROI
                for cs_idx in range(0, len(rtstruct.ds.ROIContourSequence[roi_idx].ContourSequence)):
                    # contour_data_list = []  # for debugging
                    # loop over the ContourData list
                    for idx, c in enumerate(rtstruct.ds.ROIContourSequence[roi_idx].ContourSequence[cs_idx].ContourData):
                        # contour_data_list.append(decimal_check(c, 10))  # for debugging
                        if decimal_check(c, 10) is True:
                            rtstruct.ds.ROIContourSequence[roi_idx].ContourSequence[cs_idx].ContourData[idx] = round(c, 10)
        logging.info("Rounding ContourData values complete ...")

        # save RT Struct
        rtstruct.save(os.path.join(dcm_output_path, rt_struct_output_filename))

        # Log off
        logging.info(f"RT Struct written to {os.path.join(dcm_output_path, rt_struct_output_filename)}")
        logging.info("RT Struct creation complete ...")
        logging.info(f"End {self.compute.__name__}")


def add_nii_roi_to_rtstruct(nii_seg_path, nii_filename, rtstruct):
    """
    Add NIfTI segmentation to rt-utils rtstruct object
    :param nii_filename: NIfTI segmentation filename output from TotalSegmentator, e.g. "aorta.nii.gz"
    :param nii_seg_path: Path to folder containing NIfTI segmentations
    :param rtstruct: rt-utils rtstruct object
    :return: rtstruct object updated with segmentation
    """
    if '.nii.gz' in nii_filename:
        seg_name = nii_filename.replace('.nii.gz', '')
    elif '.nii' in nii_filename:
        seg_name = nii_filename.replace('.nii', '')
    else:
        NameError('Suffix of segmentation file is not .nii or .nii.gz.')

    nii = nib.load(os.path.join(nii_seg_path, nii_filename))
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


def decimal_check(num, dec_places):
    """
    Check if number has more than N decimal places
    :param num: Input number to check
    :param dec_places: Number of decimal places to check
    :return: True/False

    # TODO remove once new PyPI release of rt-utils
    """
    dec_len = len(str(num).split(".")[1])
    if dec_len > dec_places:
        return True
    elif dec_len <= dec_places:
        return False


def list_nii_files(nii_seg_output_path):
    """
    Get list of NIfTI filenames from a directory
    :param nii_seg_output_path: directory containing NIfTI files (.nii or .nii.gz)
    :return: nii_filenames – list of NIfTI filenames
    """
    nii_filenames = [f for f in listdir(nii_seg_output_path) if
                     isfile(join(nii_seg_output_path, f)) and '.nii' in f]

    return nii_filenames

