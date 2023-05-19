# Clinical Review PDF generator

import pydicom
import logging
import os
import os.path
from os import listdir
from os.path import isfile, join
import monai.deploy.core as md
from monai.deploy.core import DataPath, ExecutionContext, InputContext, IOType, Operator, OutputContext
from monai.deploy.core.domain.dicom_series_selection import StudySelectedSeries
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import nibabel as nib
from typing import List
import reportlab.platypus as pl  # import Table, TableStyle, Image
from reportlab.platypus import Table, TableStyle
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch, cm
from reportlab.lib import utils, colors
from reportlab.lib.fonts import tt2ps
from reportlab.rl_config import canvas_basefontname as _baseFontName
_baseFontNameB = tt2ps(_baseFontName, 1, 0)
_baseFontNameI = tt2ps(_baseFontName, 0, 1)
_baseFontNameBI = tt2ps(_baseFontName, 1, 1)


@md.input("nii_seg_output_path", DataPath, IOType.DISK)
@md.input("study_selected_series_list", List[StudySelectedSeries], IOType.IN_MEMORY)
@md.input("nii_ct_dataset", DataPath, IOType.DISK)
@md.output("pdf_file", DataPath, IOType.DISK)
@md.env(pip_packages=["pydicom >= 2.3.0", "highdicom >= 0.18.2"])
class ClinicalReviewPDFGenerator(Operator):
    """
    Generates a PDF with Sag/Cor/Ax views for each structure
    """

    def compute(self, op_input: InputContext, op_output: OutputContext, context: ExecutionContext):

        logging.info(f"Begin {self.compute.__name__}")

        # get list of masks
        nii_seg_output_path = op_input.get("nii_seg_output_path").path
        nii_filenames = [join(nii_seg_output_path, f) for f in listdir(nii_seg_output_path) if
                         isfile(join(nii_seg_output_path, f)) and '.nii' in f]

        # get original ct nifti
        ct_nifti = op_input.get("nii_ct_dataset").path

        # Get CT dicom metadata
        original_image = op_input.get("study_selected_series_list")
        study_selected_series = original_image[0]
        selected_series = study_selected_series.selected_series
        dcm_meta = selected_series[0].series.get_sop_instances()[0].get_native_sop_instance()

        # create output directory

        pdf_dir_name = "pdf"
        if not os.path.exists(pdf_dir_name):
            os.mkdir(pdf_dir_name)

        pdf_filename = "clinical_review.pdf"
        ax_img_path, sag_img_path, cor_img_path = self.create_images_for_contours(dcm_meta=dcm_meta,
                                                                                  ct_nifti_filename=ct_nifti,
                                                                                  masks=nii_filenames)
        logging.info(f"Creating PDF  ...")
        pdf_filename = self.generate_report_pdf(dcm_meta,
                                                output_filename="clinical_review.pdf",
                                                ax_img_path=ax_img_path,
                                                sag_img_path=sag_img_path,
                                                cor_img_path=cor_img_path)

        logging.info(f"DICOM Encapsulated PDF written to {pdf_filename}")
        logging.info(f"PDF creation complete ...")
        logging.info(f"End {self.compute.__name__}")
        op_output.set(value=DataPath(pdf_filename), label='pdf_file')

    def create_images_for_contours(self, dcm_meta: pydicom.Dataset, ct_nifti_filename: DataPath, masks: List):

        ct_img = nib.load(ct_nifti_filename)
        a = np.array(ct_img.dataobj)

        # Calculate aspect ratios
        ps = dcm_meta.PixelSpacing
        ss = dcm_meta.SliceThickness
        ax_aspect = ps[0] / ps[1]
        sag_aspect = ss / ps[1]
        cor_aspect = ss / ps[0]

        # select midline projections
        sag_arr = np.rot90(a[int(a.shape[0] / 2), :, :])
        cor_arr = np.rot90(a[:, int(a.shape[1] / 2), :])
        ax_arr = np.rot90(a[:, :, int(a.shape[2] / 2)])

        ax_masks = []
        sag_masks = []
        cor_masks = []
        logging.info('Generating masks')
        for i, mask in enumerate(masks):
            try:
                img = nib.load(mask)
                b = np.array(img.dataobj)
                b = b * i  # means each mask is a different value, therefore different colour on cmap = 'hsv'

                c_sag_arr = np.rot90(b[int(b.shape[0] / 2), :, :])
                c_sag_arr_masked = np.ma.masked_where(c_sag_arr == 0, c_sag_arr)
                sag_masks.append([i, c_sag_arr_masked])

                c_cor_arr = np.rot90(b[:, int(b.shape[1] / 2), :])
                c_cor_arr_masked = np.ma.masked_where(c_cor_arr == 0, c_cor_arr)
                cor_masks.append([i, c_cor_arr_masked])

                c_ax_arr = np.rot90(b[:, :, int(b.shape[2] / 2)])
                c_ax_arr_masked = np.ma.masked_where(c_ax_arr == 0, c_ax_arr)
                ax_masks.append([i, c_ax_arr_masked])

            except IndexError:
                logging.info(f"failed on {i} {mask}")
                continue

        ax_filename = 'axial_image.png'
        sag_filename = 'sagittal_image.png'
        cor_filename = 'coronal_image.png'
        num_masks = len(ax_masks)
        axial_img_path = self.create_image(mask_arr=ax_masks,
                                           ct_arr=ax_arr,
                                           aspect=ax_aspect,
                                           filename=ax_filename,
                                           num_masks=num_masks)

        sagittal_img_path = self.create_image(mask_arr=sag_masks,
                                              ct_arr=sag_arr,
                                              aspect=sag_aspect,
                                              filename=sag_filename,
                                              num_masks=num_masks)

        coronal_img_path = self.create_image(mask_arr=cor_masks,
                                             ct_arr=cor_arr,
                                             aspect=cor_aspect,
                                             filename=cor_filename,
                                             num_masks=num_masks)

        return axial_img_path, sagittal_img_path, coronal_img_path

    @staticmethod
    def create_image(mask_arr, ct_arr, aspect, filename, num_masks):
        # plot CT in grayscale
        fig1 = plt.figure()
        ax = fig1.add_subplot(111)
        ax.axis('off')
        plt.imshow(ct_arr, cmap='gray')
        ax.set_aspect(aspect)
        alpha = 0.3

        # plot contours in colours
        for i, arr in mask_arr:
            ax = plt.subplot(111)
            plt.imshow(arr, cmap='hsv', alpha=alpha, interpolation='none', vmin=0, vmax=num_masks)
            ax.set_aspect(aspect)

        plt.savefig(filename, bbox_inches="tight")
        del ax
        return filename

    @staticmethod
    def generate_report_pdf(ds_meta: pydicom.Dataset,
                            ax_img_path: DataPath,
                            sag_img_path: DataPath,
                            cor_img_path: DataPath,
                            output_filename: str,
                            ):
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
        styleH2 = styles['Heading2']

        # Add patient info
        story.append(Paragraph("TotalSegmentator (AIDE) (v0.1)", styleH2))
        story.append(Paragraph(" ", styleN))
        story.append(Paragraph("Auto-contouring of CT scans into RT StructureSets for"
                               " radiotherapy treatment planning.", styleN))

        story.append(
            Paragraph("IMPORTANT DISCLAIMER: automatically generated contours must be reviewed and approved by a "
                      "radiation oncologist before use within RT planning.",
                      styleN))

        img_info_data = [['Patient Name', patient_name],
                         ['DOB', dob],
                         ['PatientID', pat_id],
                         ['Sex', sex],
                         ['Referring Consultant', consultant],
                         ['Study description', study_description],
                         ['Series description', series_description],
                         ['Series UID', series_uid],
                         ['Series date', xray_date],
                         ['Study Time', xray_time],
                         ['Accession Number', accession_number]]

        img_info = pl.Table(img_info_data, None, 11 * [0.2 * inch], spaceBefore=0.1 * inch, spaceAfter=0.1 * inch)
        img_info.setStyle(pl.TableStyle([('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                                         ('INNERGRID', (0, 0), (-1, -1), 0.25, colors.black),
                                         ('BOX', (0, 0), (-1, -1), 0.25, colors.black),
                                         ('FONTSIZE', (0, 0), (-1, -1), 8),
                                         ]))
        story.append(img_info)

        # Add axial images
        temp = utils.ImageReader(ax_img_path)
        width = 8*cm
        iw, ih = temp.getSize()
        aspect = ih / float(iw)
        im1 = pl.Image(ax_img_path, width=width, height=(width * aspect))
        im1.hAlign = 'CENTRE'

        # Add coronal images
        temp = utils.ImageReader(cor_img_path)
        iw, ih = temp.getSize()
        aspect = ih / float(iw)
        im2 = pl.Image(cor_img_path, width=width, height=(width * aspect))
        im2.hAlign = 'CENTRE'

        # Add sag images
        temp = utils.ImageReader(sag_img_path)
        iw, ih = temp.getSize()
        aspect = ih / float(iw)
        im3 = pl.Image(sag_img_path, width=width, height=(width * aspect))
        im3.hAlign = 'CENTRE'

        chart_style = TableStyle([('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                                  ('VALIGN', (0, 0), (-1, -1), 'CENTER')])
        story.append(im1)
        story.append(Table([[im2, im3]], style=chart_style))

        #  Build PDF and save
        doc = SimpleDocTemplate(filename=str(output_filename), pagesize=A4)
        doc.build(story)

        return output_filename




