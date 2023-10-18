# Functions for smoothing 3D volumes
import numpy as np
import scipy.ndimage
import SimpleITK as sITK


def calcSmoothingSigma(sourceVox, targetVox):
    """
    Calculates appropriate smoothing sigma for smoothing image to remove high frequencies
    from image
    :param sourceVox: size of source voxel
    :param targetVox: size of target voxel (metric must match)
    :return: smoothing sigma
    """

    return np.sqrt(np.multiply((targetVox ** 2 - sourceVox ** 2), (2 * np.sqrt(2 * np.log(2))) ** (-2)))


def calc1DGaussian(vals, sigma):  # called by gaussConv3D function below

    """
    Calculates 1D Gaussian from sigma
    :param vals: a vector containing the values of x where the Gaussian will be calculated
    :param sigma: a scalar specifying the standard deviation of the Gaussian
    :return: a vector containing the values of the Gaussian for each of the vals
    """

    # calculate constant terms
    const1 = sigma * np.sqrt(2 * np.pi)
    const2 = 2 * sigma ** 2

    # calculate Gaussian
    return np.exp(-np.square(vals) / const2) / const1


def gaussConv3D(volume, sigma_x, sigma_y, sigma_z):
    """
    Convolves 3D volume with a different kernel along each axis
    :param volume: 3D array
    :param sigma_x: standard deviation of Gaussian along axis 0 (determines level of smoothing)
    :param sigma_y: standard deviation of Gaussian along axis 1 (determines level of smoothing)
    :param sigma_z: standard deviation of Gaussian along axis 2 (determines level of smoothing)
    :return: convolved 3D volume
    """

    if sigma_x == 0:  # if sigma is 0 then use "do nothing" kernel

        Gx = np.array([0, 0, 1, 0, 0])

    else:  # calculate kernel for each dimension based on input sigma

        xs = np.arange(np.floor(-2 * sigma_x), np.ceil(2 * sigma_x) + 1)
        Gx = calc1DGaussian(xs, sigma_x)
        Gx = Gx / np.sum(Gx)

    if sigma_y == 0:

        Gy = np.array([0, 0, 1, 0, 0])

    else:

        ys = np.arange(np.floor(-2 * sigma_y), np.ceil(2 * sigma_y) + 1)
        Gy = calc1DGaussian(ys, sigma_y)
        Gy = Gy / np.sum(Gy)

    if sigma_z == 0:

        Gz = np.array([0, 0, 1, 0, 0])

    else:

        zs = np.arange(np.floor(-2 * sigma_z), np.ceil(2 * sigma_z) + 1)
        Gz = calc1DGaussian(zs, sigma_z)
        Gz = Gz / np.sum(Gz)

    x_kernel = Gx.reshape((Gx.shape[0], 1, 1))
    y_kernel = Gy.reshape((1, Gy.shape[0], 1))
    z_kernel = Gz.reshape((1, 1, Gz.shape[0]))

    # convolve padded array in all 3 axes separately
    Vs = scipy.ndimage.convolve(volume, x_kernel, mode='nearest')
    Vs = scipy.ndimage.convolve(Vs, y_kernel, mode='nearest')
    Vs = scipy.ndimage.convolve(Vs, z_kernel, mode='nearest')

    return Vs


# 3.1 gaussian_resample ########################


def resampleToNewResolution(sitkImageIn,
                            resolutionInMM):  ###### updated version of previous IsotropicResampling function, 3.1????????

    """
    Resample the given input image to the given resolution. Keep the origin, spacing and direction as in the given
    image. For now 3d input images and nearest-neighbour interpolation will be assumed.
    :param sitkImageIn: The image that will be resampled.
    :param resolutionInMM: The final resolution of the returned image
    :return: Resampled SITK image object
    """

    # Get the image size and spacing of the input image
    origSize = sitkImageIn.GetSize()
    origSpacing = sitkImageIn.GetSpacing()

    # Define the output spacing and size
    outSpacing = resolutionInMM
    outSize = [int(np.ceil(origSize[0] * origSpacing[0] / outSpacing[0])),
               int(np.ceil(origSize[1] * origSpacing[1] / outSpacing[1])),
               int(np.ceil(origSize[2] * origSpacing[2] / outSpacing[2]))]

    # Generate a simple transform that will be set to identity
    idTransform = sITK.Euler3DTransform()
    idTransform.SetIdentity()

    # Resample
    resampledImg = sITK.Resample(sitkImageIn, outSize, idTransform,
                                 sITK.sitkLinear,
                                 sitkImageIn.GetOrigin(), outSpacing, sitkImageIn.GetDirection())

    return resampledImg

