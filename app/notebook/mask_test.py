import pydicom
import os.path
import os
import os.path
import matplotlib.pyplot as plt
import numpy as np
import nibabel as nib

def add_masks():
    ct_nifti_filename = "../../input/dcm_input/input-ct-dataset.nii.gz"
    mask_path_dir = "../../input/nii_seg_output/"
    dcm_meta = pydicom.dcmread("../../input/dcm_input/CT.1.2.246.352.71.3.815247573688.27173.20160407133501.dcm",
                               stop_before_pixels=True)

    masks = [os.path.join(mask_path_dir, f) for f in os.listdir(mask_path_dir) if '.nii' in f]

    img = nib.load(ct_nifti_filename)
    a = np.array(img.dataobj)

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

    # display on separate images
    a1 = plt.subplot(3,1, 1)
    plt.axis('off')
    plt.imshow(cor_arr, cmap='gray')
    a1.set_aspect(ax_aspect)

    a2 = plt.subplot(3,1, 2)
    plt.axis('off')
    plt.imshow(sag_arr, cmap='gray')
    a2.set_aspect(sag_aspect)

    a3 = plt.subplot(3,1, 3)
    plt.axis('off')
    plt.imshow(ax_arr, cmap='gray')
    a3.set_aspect(cor_aspect)

    # display all contours  ------------------
    color_list = plt.colormaps()
    max_val = len(masks)
    for i,mask in enumerate(masks):
        try:
            print(mask)
            img = nib.load(mask)
            b = np.array(img.dataobj)
            b= b*i # means each mask is a different value, therefore different colour on cmap = 'hsv'

            pax_arr = np.rot90(b[int(b.shape[0] / 2), :, :])
            pax_arr_masked = np.ma.masked_where(pax_arr == 0, pax_arr)
            psag_arr = np.rot90(b[:, int(b.shape[1] / 2), :])
            psag_arr_masked = np.ma.masked_where(psag_arr == 0, psag_arr)
            pcor_arr = np.rot90(b[:, :, int(b.shape[2] / 2)])
            pcor_arr_masked = np.ma.masked_where(pcor_arr == 0, pcor_arr)

            # display on separate images
            a1 = plt.subplot(3, 1, 1)
            plt.imshow(pcor_arr_masked, cmap='hsv', alpha=0.3, interpolation='none', vmin=0, vmax=max_val)
            a1.set_aspect(ax_aspect)
            a2 = plt.subplot(3,1, 2)
            plt.imshow(psag_arr_masked, cmap='hsv', alpha=0.3, interpolation='none', vmin=0, vmax=max_val)
            a2.set_aspect(sag_aspect)
            a3 = plt.subplot(3,1, 3)
            plt.imshow(pax_arr_masked, cmap='hsv', alpha=0.3, interpolation='none', vmin=0, vmax=max_val)
            a3.set_aspect(cor_aspect)
        except IndexError:
            print(f"failed on {i} {mask}")
            continue
    plt.show()
    plt.savefig('images.png')

if __name__ == "__main__":
    add_masks()