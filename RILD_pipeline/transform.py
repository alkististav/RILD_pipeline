from ast import Continue
import sys
import numpy as np
import nibabel as nib
import os
import glob
from RILD_pipeline.utils import resample, convert, calculate, distance_transform


###TODO DOCUMENTATION#####



def transform(patient, labels = ["lungs", "airways", "vesselness"]):
    
    patient.log(patient.patient_name + 'UPDATE: ' +' Initialising transform for each feature.')
    
    #Setting parameters
    timepoints = patient.ct_timepoints
    pixdim = np.zeros((3, len(timepoints)))  # initialise matrix of pixel dims 
    
    
    ###### seg/image/stack_path may need changing depending on final output directory
    seg_path = patient.dirs['aligned_seg_image']  
    image_path = patient.dirs['aligned_base_image']

    stack_path = patient.dirs['aligned_stack_path']

    ####### Common resoltuion based on biggest voxel size
    image_list = []
    

    for tp, timepoint in enumerate(timepoints):  # loop through image type and timepoint

        image_file = glob.glob(os.path.join(image_path,  timepoint + ".nii*"))[0]
        image = nib.load(image_file)
        image_list.append(image)
        dims = image.header["pixdim"][1:4]
        pixdim[:, tp] = dims
        
    max_pix = np.max(pixdim, axis=1)  # maximum voxel size for all five timepoints which becomes common resolution

    
    for tp, timepoint in enumerate(timepoints):

        concat_list = []
        patient.log(patient.patient_name + 'UPDATE: ' + "Processing at timepoint " + timepoint)

        for label in labels:
            patient.log(patient.patient_name + 'UPDATE: ' + "Processing for " + label)
            seg_file = glob.glob(os.path.join(seg_path, timepoint + "_" + label + ".nii*"))[0]
            

           #for now we will limit try except to only check whether the segmented image exists#
            try: 
                seg_image = nib.load(seg_file)
            except:
                patient.log(patient.patient_name + 'UPDATE: ' + "Segmented file for " + label + " not found. Cannot transform the image. Proceeding. Please upload/check that file exists for future re-run. Continuing to other segmented file.")
                continue

            # calc sigma in plane and out of plane
            sigma_plane = resample.calcSmoothingSigma(pixdim[0, tp], max_pix[0])
            sigma_z = resample.calcSmoothingSigma(pixdim[2, tp], max_pix[2])

            # Gaussian smoothing
            vol = seg_image.get_fdata()
            smoothVol = resample.gaussConv3D(vol, sigma_plane, sigma_plane, sigma_z)
            smoothNii = nib.Nifti1Image(smoothVol, seg_image.affine, seg_image.header)
            smoothSITK = convert.sitkImageFromNib(smoothNii)

            # Resample image to new isotropic grid
            resamp = resample.resampleToNewResolution(smoothSITK, max_pix)
            resampNii = convert.nibImageFromSITK(resamp)

            # Generate distance transforms for lung and airway label
            if label == "lungs" or label == "airways":
                seg_image = calculate.thresholdSegmentation(resampNii)
                seg_image = distance_transform.DistanceTransforms(seg_image)
                dt = distance_transform.DistanceTransforms.minMaxDropoffSDT(seg_image)

                concat_list.append(dt)

            # If vesselness do nothing
            if label == "vesselness":
                concat_list.append(resampNii)

        # Concatenate segmentations
        concatNii = calculate.concatNii4D(concat_list[0], concat_list[1], concat_list[2])

        # Correct affine in stack
        common = np.concatenate((max_pix, np.array([1])))
        dims = np.concatenate((seg_image.header["pixdim"][1:4], np.array([1])))
        vector = common / dims
        scale = np.eye(4)
        scale[0, 0] = vector[0]
        scale[1, 1] = vector[1]
        scale[2, 2] = vector[2]
        aff = np.matmul(seg_image.affine, scale)

        nibim = nib.Nifti1Image(concatNii.get_fdata(), aff)
        output_stack_path = os.path.join(stack_path, timepoint + ".nii.gz") #in saving, we stick to zipped nifti convention for better space use
        nib.save(nibim, output_stack_path)

if __name__ == '__transform__':
    transform(sys.argv[0])