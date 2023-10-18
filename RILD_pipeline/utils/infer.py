

#import dicom2nifti
import os
import glob
from shutil import copyfile
import shutil
#import cv2
import nibabel as nib
import numpy as np
import torch
import pickle
from PIL import Image, ImageOps
import PIL as PIL
import matplotlib.pyplot as plt
from RILD_pipeline.utils import calculate, crop_and_mask, nets_arch

import random
#from util import suplementary_functions as sf
from torchvision import transforms
#from nets_arch import UNet, UNet_small, UNet_smaller, UNet_deeper



def one_net_pred(image, model, device):

    # proper norm
    mean = 0.2
    std = 0.19

    images = torch.from_numpy(image).float()
    # masks = torch.from_numpy(mask).float()
    # images = images * masks
    images = images.unsqueeze(dim=0)
    transforms.Normalize([mean], [std])(images)
    images = images.unsqueeze(dim=0)
    images = images.to(device=device, dtype=torch.float32)

    # output = torch.argmax(model(images), dim=1)
    #
    # output = output.squeeze(dim=0).cpu().data.numpy()
    output = model(images)

    return output

def net_inference(patient, saved_model_1_path, saved_model_2_path, saved_model_3_path, saved_model_4_path, saved_model_5_path, saved_model_6_path, out_size, nii = ".nii*"):
    
    #all parameters here 
    margin = 10
    margin_z = 5  # margin in mm
    max_intensity = 300
    min_intensity = -1000
    #

    seg_path = patient.dirs['aligned_seg_image']
    timepoints=patient.ct_timepoints


    #timepoints = ['baseline_data', 'follow_up_03_data', 'follow_up_06_data', 'follow_up_12_data', 'follow_up_24_data']
    #ct_names = ['baseline', 'follow_up_03', 'follow_up_06', 'follow_up_12', 'follow_up_24']


    ########################################################################################################################

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    if not (saved_model_1_path == 'None'):
        model_1 = torch.load((saved_model_1_path), map_location='cpu')
        model_1.to(device)
        model_1.eval()
    else:
        model_1 = 'None'

    if not (saved_model_2_path == 'None'):
        model_2 = torch.load((saved_model_2_path), map_location='cpu')
        model_2.to(device)
        model_2.eval()
    else:
        model_2 = 'None'

    if not (saved_model_3_path == 'None'):
        model_3 = torch.load((saved_model_3_path), map_location='cpu')
        model_3.to(device)
        model_3.eval()
    else:
        model_3 = 'None'

    if not (saved_model_4_path == 'None'):
        model_4 = torch.load((saved_model_4_path), map_location='cpu')
        model_4.to(device)
        model_4.eval()
    else:
        model_4 = 'None'

    if not (saved_model_5_path == 'None'):
        model_5 = torch.load((saved_model_5_path), map_location='cpu')
        model_5.to(device)
        model_5.eval()
    else:
        model_5 = 'None'

    if not (saved_model_6_path == 'None'):
        model_6 = torch.load((saved_model_6_path), map_location='cpu')
        model_6.to(device)
        model_6.eval()
    else:
        model_6 = 'None'



    for timepoint_nr in range(0, len(timepoints)):

        timepoint = timepoints[timepoint_nr]
        #ct_name = ct_names[timepoint_nr]
        print()
        patient.log(patient.patient_name + ' UPDATE: ' +' Starting the image preparation for classification process for ' + timepoint)

        mask_im_path = glob.glob(patient.dirs['aligned_seg_image'] + '/' +  timepoint + '_lungs' + nii)[0]
        mask_im_nii = nib.load(mask_im_path)
        mask_im = np.asarray(mask_im_nii.get_data())
        mask_im[mask_im > 0] = 1

        ct_im_path = glob.glob(patient.dirs['aligned_base_image'] + '/' + timepoint + nii)[0]


        ct_im_nii = nib.load(ct_im_path)
        ct_im = np.asarray(ct_im_nii.get_data())

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

        ct_4_net[ct_4_net < min_intensity] = min_intensity
        ct_4_net[ct_4_net > max_intensity] = max_intensity
        ct_4_net = ct_4_net + abs(min_intensity)
        ct_4_net = ct_4_net / (abs(min_intensity) + abs(max_intensity))
        ct_4_net = ct_4_net * mask_im

        label_temp_out = np.zeros_like(ct_4_net)

        patient.log(patient.patient_name + ' UPDATE: ' +' Image preparation is done. Starting the classification process for ' + timepoint)
        for slice_nr in range(0, ct_4_net.shape[2]):

            temp_mask = np.squeeze(mask_im[:, :, slice_nr])
            # start_x, start_y, stop_x, stop_y = sf.find_bounding_box(temp_mask)

            ## Notice swap of x and y due to rotation duriing nifti image reading in
            ## It is corrected by the rotation before saving the images

            if (out_size[0] < temp_mask.shape[1]) or (out_size[1] < temp_mask.shape[0]):  # if the image is too small to large to cropp, resize it
                # print('I am resizing slice to the fixed size')

                image = np.rot90(np.array(Image.fromarray(np.squeeze(ct_4_net[:, :, slice_nr])).resize(
                                    out_size, resample=PIL.Image.BICUBIC)), 3).astype('float32')

                output = one_net_pred(image, model_1, device)

                if not (model_2 == 'None'):
                    output = output + one_net_pred(image, model_2, device)

                if not (model_3 == 'None'):
                    output = output + one_net_pred(image, model_3, device)

                if not (model_4 == 'None'):
                    output = output + one_net_pred(image, model_4, device)

                if not (model_5 == 'None'):
                    output = output + one_net_pred(image, model_5, device)

                if not (model_6 == 'None'):
                    output = output + one_net_pred(image, model_6, device)

                output = torch.argmax(output, dim=1)

                output = output.squeeze(dim=0).cpu().data.numpy()

                label_temp_out[:, :, slice_nr] = np.rot90(np.array(
                    Image.fromarray(output.astype('float')).resize((ct_4_net.shape[0], ct_4_net.shape[1]),
                        resample=PIL.Image.NEAREST)).astype('float32'))

            else:  # Cropp to the asked size of the image
                # print('I am cropping slice to the fixed size')
                image = np.rot90(
                    crop_and_mask.center_crop(np.squeeze(ct_4_net[:, :, slice_nr]), new_width=out_size[0],
                                new_height=out_size[1]), 3).astype('float32') #FIND CENTER_CROP FROM SF

                output = one_net_pred(image, model_1, device)

                if not (model_2 == 'None'):
                    output = output + one_net_pred(image, model_2, device)

                if not (model_3 == 'None'):
                    output = output + one_net_pred(image, model_3, device)

                if not (model_4 == 'None'):
                    output = output + one_net_pred(image, model_4, device)

                if not (model_5 == 'None'):
                    output = output + one_net_pred(image, model_5, device)

                if not (model_6 == 'None'):
                    output = output + one_net_pred(image, model_6, device)

                output = torch.argmax(output, dim=1)

                output = output.squeeze(dim=0).cpu().data.numpy()

                label_temp_out[:, :, slice_nr] = np.rot90(crop_and_mask.center_uncrop(output,
                                new_width=ct_4_net.shape[0], new_height=ct_4_net.shape[1]), 1) #FIND CENTER_UNCROP FROM SF

        label_temp_out = label_temp_out * mask_im

        # if number_of_class == 4:
        #     label_temp_out[label_temp_out == 3] = 4
        #     label_temp_out[label_temp_out == 2] = 7
        #
        # elif number_of_class == 5:
        #     label_temp_out[label_temp_out == 2] = 7
        #     label_temp_out[label_temp_out == 4] = 6
        #     label_temp_out[label_temp_out == 3] = 4
        #
        # elif number_of_class == 6:
        #     # print('Correct that')
        #     label_temp_out[label_temp_out == 4] = 7
        #     label_temp_out[label_temp_out == 5] = 6
        #     label_temp_out[label_temp_out == 3] = 4
        #     label_temp_out[label_temp_out == 2] = 3

        label_out[smallest_x:largest_x, smallest_y:largest_y, smallest_z:largest_z] = label_temp_out

        label_out_path = os.path.join(patient.dirs['tissue_classes'] , timepoint + '.nii.gz')
        label_out_nii = nib.Nifti1Image(label_out, ct_im_nii.affine, ct_im_nii.header)
        nib.save(label_out_nii, label_out_path)
        patient.log(patient.patient_name + ' UPDATE: ' +' Classification process done for ' + timepoint)


#             except:
#                     print('Generating training data from case: ', patient.patient_name, ' at time point: ', timepoint, ' failed.')
#     except:
#             print('Generating training data from case: ', patient.patient_name, ' failed.')

    return