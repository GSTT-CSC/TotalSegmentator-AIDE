import pydicom
import os.path
import os
import os.path
import matplotlib.pyplot as plt
import numpy as np
import nibabel as nib

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

    print(f"ax_aspect : {ax_aspect} \n sag_aspect : {sag_aspect} \n  cor_aspect : {cor_aspect} \n  ")

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

    a3 = plt.subplot(3,1, 3)
    plt.axis('off')
    plt.imshow(ax_arr, cmap='gray')
    a3.set_aspect(cor_aspect)

    # display all contours  ------------------
    # repeats above for all masks

    max_val = len(masks)    # sets upper limit of colour scale on cmap = 'hsv'
    alpha = 0.3             # transparency of contour

    for i, mask in enumerate(masks):
        try:
            print(mask)
            img = nib.load(mask)
            b = np.array(img.dataobj)
            b = b * i        # means each mask is a different value, therefore different colour on cmap = 'hsv'

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
            print(f"failed on {i} {mask}")
            continue
    plt.show()
    plt.savefig('images.png')

if __name__ == "__main__":
    add_masks()