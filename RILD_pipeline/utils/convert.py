import numpy as np
import nibabel as nib
import SimpleITK as sITK
import pydicom as dc

'''
Collection of methods to convert from nibabel to simpleITK and vice versa. Note: Only applicable for 3d images for
now.
'''


def sitkImageFromNib(nibImageIn):
    '''
    Convert a nibabel image into an SITK object.
    @param nibImageIn: Image object read with nibabel library
    '''

    # Currently only 3d images are supported
    if (nibImageIn.header['dim'][0] != 3):
        print("WARNING: This class is currently only intended for 3D images")

    # Generate an sitk image object from the nibabel image array,
    # Note that the order of the axes is reverted
    nibImageIn_np = nibImageIn.get_fdata()
    sitkImage = sITK.GetImageFromArray(np.transpose(nibImageIn_np, [2, 1, 0]))

    # Set the image geometry
    # - origin
    sitkImage.SetOrigin(nibImageIn.affine[:3, 3] * np.array([-1, -1, 1]))

    # - spacing
    sitkImage.SetSpacing(nibImageIn.header['pixdim'][1:4].astype(np.double))

    # - direction
    dirMatrix = nibImageIn.affine[:3, :3].copy()
    dirMatrix[:, 0] = dirMatrix[:, 0] / np.linalg.norm(dirMatrix[:, 0])
    dirMatrix[:, 1] = dirMatrix[:, 1] / np.linalg.norm(dirMatrix[:, 1])
    dirMatrix[:, 2] = dirMatrix[:, 2] / np.linalg.norm(dirMatrix[:, 2])
    dirMatrix[:2, :] = dirMatrix[:2, :] * (-1)

    sitkImage.SetDirection(dirMatrix.reshape(-1))

    return sitkImage



def nibImageFromSITK( sITKImageIn ):

    '''
    Generate a new nifti image from a given SITK image.
    @param sITKImageIn: THe simple ITK image object to be converted. Note, only 3D images supported at the moment.
    '''


    # Currently only 3D images supported.
    if (sITKImageIn.GetDimension() != 3):
        print("WARNING: This class is currently only intended for 3D images")
        
    affineMatrix = np.eye(4)

    # Create the matrix according to itkSoftware guide
    affineMatrix[:3,:3] = np.dot( np.diag(sITKImageIn.GetSpacing()), np.array(sITKImageIn.GetDirection()).reshape([3,-1]) )
    affineMatrix[:3,3] = sITKImageIn.GetOrigin()


    # Account for change in geometry dicom/ITK vs. nifti
    affineMatrix[:2, :] = (-1) * affineMatrix[:2, :]
    

    return nib.Nifti1Image( np.transpose( sITK.GetArrayFromImage( sITKImageIn ), [2,1,0]), affineMatrix )


def correct_dose_header(dicom_dose, nifti_dose):
    
    dcm = dc.read_file(dicom_dose, force=True)
    nii = nib.load(nifti_dose)

    slice_thickness = dcm.GridFrameOffsetVector[1] - dcm.GridFrameOffsetVector[0]
    affine = nii.affine
    affine[2,2] = slice_thickness

    corrected = nib.Nifti1Image(nii.get_fdata(), affine=affine)
    nib.save(corrected, nifti_dose)

def scale_dose(dicom_dose, nifti_dose):
    dcm = dc.read_file(dicom_dose, force=True)
    nii = nib.load(nifti_dose)

    scale_factor = dcm.DoseGridScaling
    scaled_nii = nii.get_fdata() * scale_factor
    scaled_nib = nib.Nifti1Image(scaled_nii, affine=nii.affine)
    nib.save(scaled_nib, nifti_dose)
