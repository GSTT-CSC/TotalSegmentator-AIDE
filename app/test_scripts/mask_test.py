import pydicom
import os.path
import os
import os.path
import matplotlib.pyplot as plt
import numpy as np
import nibabel as nib
from monai.deploy.core.domain import DataPath
from pathlib import Path

""" Test script to generate view"""

def add_masks():
    # read in ct nifti and create masks list
    ct_nifti_filename = "../../input/dcm_input/input-ct-dataset.nii.gz"
    mask_path_dir = "../../input/nii_seg_output/"
    dcm_meta = pydicom.dcmread("../../input/dcm_input/CT.1.2.246.352.71.3.815247573688.27173.20160407133501.dcm",
                               stop_before_pixels=True)

    masks = [os.path.join(mask_path_dir, f) for f in os.listdir(mask_path_dir) if '.nii' in f]
    img = nib.load(ct_nifti_filename)
    a = np.array(img.dataobj)

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


    for i, mask in enumerate(masks):
        try:
            print(mask)
            img = nib.load(mask)
            b = np.array(img.dataobj)
            b = b * i        # means each mask is a different value, therefore different colour on cmap = 'hsv'

            c_sag_arr = np.rot90(b[int(b.shape[0] / 2), :, :])
            c_sag_arr_masked = np.ma.masked_where(c_sag_arr == 0, c_sag_arr)
            sag_masks.append([i, c_sag_arr_masked])

            c_cor_arr = np.rot90(b[:, int(b.shape[1] / 2), :])
            c_cor_arr_masked = np.ma.masked_where(c_cor_arr == 0, c_cor_arr)
            cor_masks.append([i, c_cor_arr_masked])

            c_ax_arr = np.rot90(b[:, :, int(b.shape[2] / 2)])
            c_ax_arr_masked = np.ma.masked_where(c_ax_arr == 0, c_ax_arr)
            ax_masks.append([i, c_ax_arr_masked])

            # display on separate images
            """ax = plt.subplot(111)
            plt.imshow(c_cor_arr_masked, cmap='hsv', alpha=alpha, interpolation='none', vmin=0, vmax=max_val)
            ax.set_aspect(ax_aspect)
            a2 = plt.subplot(3, 1, 2)
            plt.imshow(c_sag_arr_masked, cmap='hsv', alpha=alpha, interpolation='none', vmin=0, vmax=max_val)
            a2.set_aspect(sag_aspect)
            a3 = plt.subplot(3, 1, 3)
            plt.imshow(c_ax_arr_masked, cmap='hsv', alpha=alpha, interpolation='none', vmin=0, vmax=max_val)
            a3.set_aspect(cor_aspect)"""

        except IndexError:
            print(f"failed on {i} {mask}")
            continue

    def create_image(mask_arr, ct_arr, aspect, filename):
        #plot CT
        fig1 = plt.figure()
        ax = fig1.add_subplot(111)
        ax.axis('off')
        plt.imshow(ct_arr, cmap='gray')
        ax.set_aspect(aspect)

        max_val = len(masks)  # sets upper limit of colour scale on cmap = 'hsv'
        alpha = 0.3
        # plot contours
        for i, arr in mask_arr:
            ax = plt.subplot(111)
            plt.imshow(arr, cmap='hsv', alpha=alpha, interpolation='none', vmin=0, vmax=max_val)
            ax.set_aspect(aspect)

        plt.savefig(filename)
        del ax
        return filename

    data_path = "/Users/anil/Documents/GitHub/TotalSegmentator-AIDE/app/notebook"
    ax_filename = os.path.join(data_path, 'axial_image.png')
    sag_filename = os.path.join(data_path,'sagittal_image.png')
    cor_filename = os.path.join(data_path,'coronal_image.png')

    axial_img_path = create_image(mask_arr=ax_masks,
                                  ct_arr=ax_arr,
                                  aspect=ax_aspect,
                                  filename=ax_filename)

    sagittal_img_path = create_image(mask_arr=sag_masks,
                                     ct_arr=sag_arr,
                                     aspect=sag_aspect,
                                     filename=sag_filename)

    coronal_img_path = create_image(mask_arr=cor_masks,
                                    ct_arr=cor_arr,
                                    aspect=cor_aspect,
                                    filename=cor_filename)

    return axial_img_path, sagittal_img_path, coronal_img_path


if __name__ == "__main__":
    ax, sag, corr = add_masks()
    print(f"ax: {ax} \nsag: {sag} \ncor: {corr}")