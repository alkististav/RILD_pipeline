from paramiko import SSHClient
from scp import SCPClient

import dicom2nifti
import os
import glob
from shutil import copyfile
import shutil
import cv2
import nibabel as nib
import numpy as np
import torch
import pickle
from PIL import Image, ImageOps
import PIL as PIL
import scipy.ndimage as ndimage 
import skimage.measure
import skimage.morphology
# from itertools import chain
import random

from utils import crop_and_mask

from nets_arch import UNet, UNet_small, UNet_smaller, UNet_deeper
import SimpleITK as sitk

######################## inference function ####################

def net_inference(image, model):
    image[image > 1] = 1
    image[image < 0] = 0
    mean = 0.5
    std = 0.35

    image = (image - mean) / std
    images = torch.from_numpy(image).float()

    images = images.unsqueeze(dim=0).unsqueeze(dim=0)
    images = images.to(device=device, dtype=torch.float32)
    output = model(images)

    return output


def lung_segment(patient, parameters):
    out_size = (288, 384)
    margin = 10
    margin_z = 10  # margin in mm
    max_intensity = 300
    min_intensity = -1000


    #all paths

    output_dir = patient.dirs['orig_seg_image'] 
    image_dir = patient.dirs['orig_base_image']
    model_dir = patient.dirs['model_image']
    model_one = parameters.seg_models['lung'][0]  
    model_two = parameters.seg_models['lung'][0]  

    model_one_path = os.path.join(model_dir, model_one)
    model_two_path = os.path.join(model_dir, model_two)

    timepoints = patient.ct_timepoints

    #############RUNNING MODEL#################################################################################################

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #load models

    model1 = torch.load(model_one_path)
    model1.to(device)
    model1.eval()

    model2 = torch.load(model_two_path)
    model2.to(device)
    model2.eval()


    for followup in patient.ct_timepoints[1:]:  

        ct_im_path = glob.glob(os.path.join(patient.dirs['orig_base_image'], followup + ".nii.gz"))[0]
        ct_im_nii = nib.load(ct_im_path)
        ct_im = np.asarray(ct_im_nii.get_data())

        mask_im = simple_bodymask(ct_im)
        ct_im_header = ct_im_nii.header
        pixdim = ct_im_header['pixdim']

        label_out = np.zeros_like(ct_im)

                    #######################################################
                    ## Cropping data to the lungs size plus margin
        sum_slice = np.squeeze(np.nansum(mask_im, axis=2))
        sum_slice[sum_slice > 1] = 1

        sum_slice_z = np.squeeze(np.nansum(mask_im, axis=0))
        sum_z = np.squeeze(np.nansum(sum_slice_z, axis=0))
        sum_z[sum_z > 1] = 1

                    #  measures in the xy plane
        sum_x = np.squeeze(np.nansum(sum_slice, axis=1))
        sum_y = np.squeeze(np.nansum(sum_slice, axis=0))
        sum_x[sum_x > 1] = 1
        sum_y[sum_y > 1] = 1

        temp_x = np.argwhere(sum_x > 0)
        temp_y = np.argwhere(sum_y > 0)
        temp_z = np.argwhere(sum_z > 0)
        smallest_x = max(0, temp_x.min() - int(
                        margin / pixdim[1]))  # This is to add some margin before cropping later on
        smallest_y = max(0, temp_y.min() - int(margin / pixdim[2]))
        smallest_z = max(0, temp_z.min() - int(margin_z / pixdim[3]))
        largest_x = min(mask_im.shape[0], temp_x.max() + int(margin / pixdim[1]))
        largest_y = min(mask_im.shape[1], temp_y.max() + int(margin / pixdim[2]))
        largest_z = min(mask_im.shape[2], temp_z.max() + int(margin_z / pixdim[3]))

        length_x = (largest_x - smallest_x)
        mid_x = int(smallest_x + largest_x) / 2
        length_y = (largest_y - smallest_y)
        mid_y = int(smallest_y + largest_y) / 2

        if (length_x < out_size[1]):
            smallest_x = max(0, int((mid_x) - (out_size[1] / 2)))
            largest_x = smallest_x + out_size[1]

        if (length_y < out_size[0]):
            smallest_y = max(0, int((mid_y) - (out_size[0] / 2)))
            largest_y = smallest_y + out_size[0]

        ct_4_net = ct_im[smallest_x:largest_x, smallest_y:largest_y, smallest_z:largest_z]
        mask_im = mask_im[smallest_x:largest_x, smallest_y:largest_y, smallest_z:largest_z]

        ct_4_net[ct_4_net < min_intensity] = min_intensity #CT range for intensities -- crop the images between min max
        ct_4_net[ct_4_net > max_intensity] = max_intensity

        ct_4_net = ct_4_net + max_intensity # standardisation
        ct_4_net = ct_4_net / (abs(min_intensity) + abs(max_intensity))

        label_temp_out = np.zeros_like(ct_4_net)

        for slice_nr in range(0, ct_4_net.shape[2]):

            temp_mask = np.squeeze(mask_im[:, :, slice_nr])

            if (out_size[0] < temp_mask.shape[1]) or (out_size[1] < temp_mask.shape[0]): 

                image = np.rot90(np.array(Image.fromarray(np.squeeze(ct_4_net[:, :, slice_nr])).resize(
                                    out_size, resample=PIL.Image.BICUBIC)), 3).astype('float32')

                output_1 = net_inference(image, model1)
                output_2 = net_inference(image, model2)

                output = output_1 + output_2

                output = torch.argmax(output, dim=1)

                output = output.squeeze(dim=0).cpu().data.numpy()

                label_temp_out[:, :, slice_nr] = np.rot90(np.array(Image.fromarray(output.astype('float')).resize((ct_4_net.shape[0], ct_4_net.shape[1]),
                                                                                   resample=PIL.Image.NEAREST)).astype(
                                    'float32'))

            else:  # Cropp to the asked size of the image
                            # print('I am cropping slice to the fixed size')
                image = np.rot90(
                                crop_and_mask.center_crop(np.squeeze(ct_4_net[:, :, slice_nr]), new_width=out_size[0],
                                               new_height=out_size[1]), 3).astype('float32')

                output_1 = net_inference(image, model1)
                output_2 = net_inference(image, model2)
                            # output_3 = net_inference(image, model3)

                output = output_1 + output_2

                output = torch.argmax(output, dim=1)

                output = output.squeeze(dim=0).cpu().data.numpy()

                label_temp_out[:, :, slice_nr] = np.rot90(crop_and_mask.center_uncrop(output,new_width=ct_4_net.shape[0],new_height=ct_4_net.shape[1]), 1)

                label_out[smallest_x:largest_x, smallest_y:largest_y, smallest_z:largest_z] = label_temp_out

                label_out = crop_and_mask.sep_lungs_from_mask(label_out, ct_im)

                label_out_path = nifti_dir + case_nr + '/' + timepoint + '/' + output_name
                label_out_nii = nib.Nifti1Image(label_out, ct_im_nii.affine, ct_im_nii.header)
                nib.save(label_out_nii, label_out_path)

            print(followup + ' done')

    print('Finito')

