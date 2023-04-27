# Clinical Review PDF generator
import monai.deploy.core as md
import pydicom
from monai.deploy.core import DataPath, ExecutionContext, InputContext, IOType, Operator, OutputContext
import logging
import nilearn
import pathlib
import os.path
from os import listdir
from os.path import isfile, join
import os
import os.path
import monai.deploy.core as md
from monai.deploy.core import DataPath, ExecutionContext, InputContext, IOType, Operator, OutputContext
from monai.deploy.core.domain.dicom_series_selection import StudySelectedSeries
from monai.deploy.core.domain import DICOMStudy
from monai.deploy.core.domain.dicom_sop_instance import DICOMSOPInstance
from pathlib import Path
import SimpleITK as sitk

import matplotlib.pyplot as plt
import numpy as np
import nibabel as nib

from typing import TYPE_CHECKING, Dict, List, Optional, Sequence
import reportlab.platypus as pl # import Table, TableStyle, Image
from reportlab.platypus import Table, TableStyle, Image
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, KeepInFrame
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch, cm
from reportlab.lib import utils

from reportlab.lib import colors
from reportlab.lib.colors import white, black
from reportlab.lib.fonts import tt2ps
from reportlab.rl_config import canvas_basefontname as _baseFontName
_baseFontNameB = tt2ps(_baseFontName, 1, 0)
_baseFontNameI = tt2ps(_baseFontName, 0, 1)
_baseFontNameBI = tt2ps(_baseFontName, 1, 1)


@md.input("input_files", DataPath, IOType.DISK)
@md.output("pdf_file", DataPath, IOType.DISK)
@md.output("study_selected_series", List[StudySelectedSeries], IOType.IN_MEMORY)
@md.env(pip_packages=["pydicom >= 2.3.0", "highdicom >= 0.18.2"])
class ClinicalReviewPDFGenerator(Operator):

    """Generates a PDF with Sag/Cor/Ax views for each structure"""

    def compute(self, op_input: InputContext, op_output: OutputContext, context: ExecutionContext):

        logging.info(f"Begin {self.compute.__name__}")
        # get list of masks
        input_path = op_input.get("input_files").path
        dcm_input_path = input_path / 'dcm_input'
        nii_seg_output_path = dcm_input_path / 'nii_seg_output'
        nii_filenames = [join(nii_seg_output_path, f) for f in listdir(nii_seg_output_path) if
                         isfile(join(nii_seg_output_path, f)) and '.nii' in f]

        ct_nifti = dcm_input_path / 'input-ct-dataset.nii.gz'

        logging.info(f"Creating PDF  ...")

        # Get example CT dicom, get images and generate report
        example_ct_file = [os.path.join(dcm_input_path, f) for f in os.listdir(dcm_input_path) if '.dcm' in f][0]
        dcm_meta = pydicom.dcmread(example_ct_file, stop_before_pixels=True)
        pdf_output_path = input_path / 'pdf'
        Path(pdf_output_path).mkdir(parents=True, exist_ok=True)

        img_path = self.create_images_for_contours(dcm_meta=dcm_meta,
                                                   output_path=pdf_output_path,
                                                   ct_nifti_filename=ct_nifti,
                                                   masks=nii_filenames)

        pdf_filename = self.generate_report_pdf(dcm_meta,
                                                image_path=img_path,
                                                output_path=pdf_output_path,
                                                nii_filenames=nii_filenames)

        logging.info(f"Dicom Encapsulated PDF written to {pdf_filename}")
        logging.info(f"PDF creation complete ...")
        logging.info(f"End {self.compute.__name__}")

        op_output.set(value=DataPath(pdf_filename), label='pdf_file')
        return

    def create_images_for_contours(self, dcm_meta: pydicom.Dataset, output_path : DataPath,
                                   ct_nifti_filename: DataPath, masks : List):

        img = nib.load(ct_nifti_filename)
        a = np.array(img.dataobj)

        # Calculate aspect ratios
        ps = dcm_meta.PixelSpacing
        ss = dcm_meta.SliceThickness

        ax_aspect = ps[0] / ps[1]
        sag_aspect = ss / ps[1]
        cor_aspect = ss / ps[0]

        logging.info(f"ax_aspect : {ax_aspect} \n sag_aspect : {sag_aspect} \n  cor_aspect : {cor_aspect} \n  ")

        # select midline projections
        ax_arr = np.rot90(a[int(a.shape[0] / 2), :, :])
        sag_arr = np.rot90(a[:, int(a.shape[1] / 2), :])
        cor_arr = np.rot90(a[:, :, int(a.shape[2] / 2)])

        # display projections
        a1 = plt.subplot(3, 1, 1)
        plt.axis('off')
        plt.imshow(cor_arr, cmap='gray')
        a1.set_aspect(ax_aspect)

        a2 = plt.subplot(3, 1, 2)
        plt.axis('off')
        plt.imshow(sag_arr, cmap='gray')
        a2.set_aspect(sag_aspect)

        a3 = plt.subplot(3, 1, 3)
        plt.axis('off')
        plt.imshow(ax_arr, cmap='gray')
        a3.set_aspect(cor_aspect)

        # display all contours  ------------------
        # repeats above for all masks

        max_val = len(masks)  # sets upper limit of colour scale on cmap = 'hsv'
        alpha = 0.3  # transparency of contour

        for i, mask in enumerate(masks):
            try:
                logging.info(mask)
                img = nib.load(mask)
                b = np.array(img.dataobj)
                b = b * i  # means each mask is a different value, therefore different colour on cmap = 'hsv'

                c_ax_arr = np.rot90(b[int(b.shape[0] / 2), :, :])
                c_ax_arr_masked = np.ma.masked_where(c_ax_arr == 0, c_ax_arr)
                c_sag_arr = np.rot90(b[:, int(b.shape[1] / 2), :])
                c_sag_arr_masked = np.ma.masked_where(c_sag_arr == 0, c_sag_arr)
                c_cor_arr = np.rot90(b[:, :, int(b.shape[2] / 2)])
                c_cor_arr_masked = np.ma.masked_where(c_cor_arr == 0, c_cor_arr)

                # display on separate images
                a1 = plt.subplot(3, 1, 1)
                plt.imshow(c_cor_arr_masked, cmap='hsv', alpha=alpha, interpolation='none', vmin=0, vmax=max_val)
                a1.set_aspect(ax_aspect)
                a2 = plt.subplot(3, 1, 2)
                plt.imshow(c_sag_arr_masked, cmap='hsv', alpha=alpha, interpolation='none', vmin=0, vmax=max_val)
                a2.set_aspect(sag_aspect)
                a3 = plt.subplot(3, 1, 3)
                plt.imshow(c_ax_arr_masked, cmap='hsv', alpha=alpha, interpolation='none', vmin=0, vmax=max_val)
                a3.set_aspect(cor_aspect)
            except IndexError:
                logging.info(f"failed on {i} {mask}")
                continue
        img_path = output_path / 'images.png'
        plt.savefig(img_path)

        return img_path

    def generate_report_pdf(self, ds_meta: pydicom.Dataset, image_path: DataPath,  output_path: DataPath, nii_filenames):
        """
        --Test Script--
        Generates pdf report of the results. Takes a dicom image to
        extract patient demographics and study information, and dict of results from classifier.
        Generates a PDF document that can be used downstream in the operator:

        - monai.deploy.operators.DICOMEncapsulatedPDFWriterOperator

        TODO: - format pdf for radiologist requirements, add extra info
              - change name of saved document to pat_id_DATE.pdf for use elsewhere if required e.g. email.
              - add meaningful logging statements
        """

        # Get original dicom series image metadata
        patient_name = ds_meta['PatientName'].value
        dob = ds_meta['PatientBirthDate'].value
        pat_id = ds_meta['PatientID'].value
        sex = ds_meta['PatientSex'].value
        consultant = ds_meta['ReferringPhysicianName'].value
        study_description = ds_meta['StudyDescription'].value
        series_description = ds_meta['SeriesDescription'].value
        series_uid = ds_meta['SeriesInstanceUID'].value
        xray_date = ds_meta['SeriesDate'].value
        xray_time = ds_meta['AcquisitionTime'].value
        accession_number = ds_meta['AccessionNumber'].value

        # generate PDF
        logging.info("building pdf - using reportlab platypus flowables")
        story = []
        styles = getSampleStyleSheet()
        styleN = styles['Normal']
        styleH1 = styles['Heading1']
        styleH2 = styles['Heading2']
        styleH3 = ParagraphStyle(name='Heading3',
                                 parent=styles['Normal'],
                                 fontName=_baseFontNameB,
                                 fontSize=12,
                                 leading=14,
                                 spaceBefore=12,
                                 spaceAfter=6,
                                 textColor=white)


        current_dir = os.path.dirname(__file__)

        # Add patient info
        story.append(Paragraph("TotalSegmentator (AIDE) (v0.1)", styleH1))
        story.append(Paragraph("Auto-contouring of CT scans into RT StructureSets for radiotherapy treatment planning.", styleN))
        story.append(Paragraph(f"Patient Name: {patient_name}", styleH2))
        story.append(Paragraph(f"DOB: {dob} ", styleH2))
        story.append(Paragraph(f"PatientID: {pat_id} ", styleH2))
        story.append(Paragraph(f"Sex: {sex} ", styleH2))


        # Add disclaimer
        story.append(
            Paragraph("IMPORTANT DISCLAIMER: automatically generated contours must be reviewed and approved by a "
                      "radiation oncologist before use within RT planning.",
                      styleH2))

        # Add image info
        story.append(Paragraph("Image used for analysis:", styleN))

        img_info_data = [['Referring physician', consultant],
                         ['Study description', study_description],
                         ['Series description', series_description],
                         ['Series UID', series_uid],
                         ['Series date', xray_date],
                         ['Study Time', xray_time],
                         ['Accession Number', accession_number]]


        img_info = pl.Table(img_info_data, None, 7 * [0.2 * inch], spaceBefore=0.1 * inch, spaceAfter=0.1 * inch)
        img_info.setStyle(pl.TableStyle([('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                                         ('INNERGRID', (0, 0), (-1, -1), 0.25, colors.black),
                                         ('BOX', (0, 0), (-1, -1), 0.25, colors.black),
                                         ('FONTSIZE', (0, 0), (2, 6), 8),
                                         ]))

        story.append(img_info)

        # Add image
        temp = utils.ImageReader(image_path)
        width = 20*cm
        iw, ih = temp.getSize()
        aspect = ih / float(iw)

        im = pl.Image(image_path, width=width, height=(width * aspect))
        im.hAlign = 'CENTRE'
        story.append(im)

        #  Build PDF and save
        pdf_path = output_path / "clinical_review.pdf"
        doc = SimpleDocTemplate(filename=str(pdf_path), pagesize=A4)
        doc.build(story)

        return pdf_path




