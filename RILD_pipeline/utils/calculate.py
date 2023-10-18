# Functions to be used for preprocessing nifti labels and generating multichannel labels
import numpy as np
import nibabel as nib



def thresholdSegmentation(nibSegIn): ######## 3.2!!!!!!!!!!

    """
    thresholds a segmentation into a binary image
    """

    
    affine = nibSegIn.affine
    header = nibSegIn.header
    seg = nibSegIn.get_fdata()


    seg[seg >= 0.5] = 1
    seg[seg < 0.5] = 0


    binSegOut = nib.Nifti1Image(seg, affine, header)

    return binSegOut


def concatNii4D(lungsNii, airwaysNii, vessNii): ######## 3.3!!!!!!

    """
    Concatenates 3 3D images along the 4th dimension and
    saves as a nifti file in the coordinate system of the
    1st input.
    All images must have the same size.
    """

    shape = lungsNii.get_fdata().shape
    stack = np.zeros((shape[0], shape[1], shape[2], 3))
    stack[:, :, :, 0] = lungsNii.get_fdata()
    stack[:, :, :, 1] = airwaysNii.get_fdata()
    stack[:, :, :, 2] = vessNii.get_fdata()

    stackNii = nib.Nifti1Image(stack, lungsNii.affine)

    return stackNii


def cropNii(nibImIn, x_start, x_stop, y_start, y_stop, z_start, z_stop):
    """
    Crops 3D nifti volume
    :param nibImIn: 3D nifti vol
    :param x_start: new min x
    :param x_stop: new max x
    :param y_start: new min y
    :param y_stop: new max y
    :param z_start: new min z
    :param z_stop: new max z
    :return: cropped 3D nifti vol
    """

    image = nibImIn.get_fdata()
    affine = nibImIn.affine
    header = nibImIn.header

    if x_start < 0: x_start = 0
    if y_start < 0: y_start = 0
    if z_start < 0: z_start = 0
    if x_stop > int(header["dim"][1]): x_stop = int(header["dim"][1])
    if y_stop > int(header["dim"][2]): y_stop = int(header["dim"][2])
    if z_stop > int(header["dim"][3]): z_stop = int(header["dim"][3])


    cropped_image = image[x_start:x_stop,
                          y_start:y_stop,
                          z_start:z_stop]

    image_origin = np.array([[x_start], [y_start], [z_start], [1]])
    world_origin = np.matmul(affine, image_origin)

    affine[:, 3] = np.squeeze(world_origin)

    croppedNibIm = nib.Nifti1Image(cropped_image, affine)
    return(croppedNibIm)


def padNii(nibImIn, x_width, y_width, z_width, padding=0):
    """
    pads 3D nifti volume up to desired size, centered only
    :param nibImIn: 3D nifti vol
    :param x_width: desired resulting size along x
    :param y_width: desired resulting size along y
    :param z_width: desired resulting size along z
    :return: centered padded nifti volume of specified size
    """

    image = nibImIn.get_fdata()
    affine = nibImIn.affine

    dimensions = nibImIn.header["dim"][1:4]

    x_pad = x_width - dimensions[0]
    y_pad = y_width - dimensions[1]
    z_pad = z_width - dimensions[2]

    if x_pad % 2 == 0:
        x_low = x_pad/2
    else:
        x_low = (x_pad - 1)/2

    if y_pad % 2 == 0:
        y_low = y_pad/2
    else:
        y_low = (y_pad - 1) / 2

    if z_pad % 2 == 0:
        z_low = z_pad/2
    else:
        z_low = (z_pad - 1) / 2

    offsets = [int(x_low), int(y_low), int(z_low)]

    result = np.empty(np.array([x_width, y_width, z_width]))
    result[:] = padding
    insertHere = [slice(offsets[dim], offsets[dim] + image.shape[dim]) for dim in range(image.ndim)]
    result[insertHere] = image

    image_origin = np.array([[-x_low], [-y_low], [-z_low], [1]])
    world_origin = np.matmul(affine, image_origin)

    affine[:, 3] = np.squeeze(world_origin)

    paddedNibIm = nib.Nifti1Image(result, affine)
    return(paddedNibIm)


################################################################################
# Image Intensities ############################################################
################################################################################


def clip_intensities(niftiImgIn, minInt=-1000, maxInt=300):
    """
    Clip intensities of nifti image to set range
    """

    img = niftiImgIn.get_fdata()
    affine = niftiImgIn.affine
    hdr = niftiImgIn.header

    img[img <= minInt] = minInt
    img[img >= maxInt] = maxInt

    niftiImgOut = nib.Nifti1Image(img, affine, hdr)
    return (niftiImgOut)


def rescale_intensities(niftiImgIn, newMinInt=0, newMaxInt=1):
    """
    Rescale intensities of nifti image between set boundaries
    """

    img = niftiImgIn.get_fdata()
    affine = niftiImgIn.affine
    hdr = niftiImgIn.header

    minIn = np.amin(img)
    maxIn = np.amax(img)

    rescaledImg = (((img - minIn)/(maxIn - minIn)) * (newMaxInt - newMinInt)) + newMinInt

    niftiImgOut = nib.Nifti1Image(rescaledImg, affine, hdr)
    return niftiImgOut


