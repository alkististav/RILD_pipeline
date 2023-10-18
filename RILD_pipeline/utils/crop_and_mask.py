
import os
import glob
from shutil import copyfile
import shutil
import nibabel as nib
import numpy as np
import torch
from PIL import Image, ImageOps
import PIL as PIL
import scipy.ndimage as ndimage 
import skimage.measure
import skimage.morphology

#from unet import UNet  #, UNet_small, UNet_smaller, UNet_deeper
import SimpleITK as sitk

def simple_bodymask(nibimg, airwaynib):
    """
    function to generate a simple body mask from a chest ct
    """
    img = nibimg.get_fdata()
    header = nibimg.header

    airwayimg = airwaynib.get_fdata()
    img[airwayimg > 0] = 300
    maskthreshold = -500
    oshape = img.shape
    img = ndimage.zoom(img, 128/np.asarray(img.shape), order=0)
    bodymask = img > maskthreshold
    bodymask = ndimage.binary_closing(bodymask) #close small holes
    bodymask = ndimage.binary_fill_holes(bodymask, structure=np.ones((2, 2, 2))).astype(int) #fill holes with square element
    bodymask = ndimage.binary_erosion(bodymask, iterations=2) #erode
    bodymask = skimage.measure.label(bodymask.astype(int), connectivity=1)
    regions = skimage.measure.regionprops(bodymask.astype(int)) #count labels
    if len(regions) > 0:
        max_region = np.argmax(list(map(lambda x: x.area, regions))) + 1
        bodymask = bodymask == max_region
        bodymask = ndimage.binary_dilation(bodymask, iterations=2)
    real_scaling = np.asarray(oshape)/128
    return ndimage.zoom(bodymask, real_scaling, order=0)

def keep_two_largest_components(seg):
    """
    function to keep the two largest connected components of lung segmentation
    and remove the second if it is much smaller than the first (for the case that the lungs are connected)
    """
    # find largest connected component
    labels = skimage.measure.label(seg, return_num=False)
    maxCC_nobcg = labels == np.argmax(np.bincount(labels.flat, weights=seg.flat))
    # remove it from segmentation
    seg[maxCC_nobcg == 1] = 0
    # find second largest connected component
    labels = skimage.measure.label(seg, return_num=False)
    secondCC_nobcg = labels == np.argmax(np.bincount(labels.flat, weights=seg.flat))
    # check ratio of largest to second-largest connected component
    ratio = np.sum(maxCC_nobcg) / np.sum(secondCC_nobcg)
    # print(ratio)
    if ratio < 10:
        # add the two largest connected components
        two = np.uint8(maxCC_nobcg) + np.uint8(secondCC_nobcg)
    else:
        # add only the largest connected component
        two = np.uint8(maxCC_nobcg)
    return two

def center_crop(img, new_width=None, new_height=None):

    width = img.shape[1]
    height = img.shape[0]

    if new_width is None:
        new_width = min(width, height)
    elif new_width > width:
        print('Error in width image cropping!')

    if new_height is None:
        new_height = min(width, height)
    elif new_height > height:
        print('Error in height image cropping!')

    left = int(np.ceil((width - new_width) / 2))
    right = width - int(np.floor((width - new_width) / 2))

    top = int(np.ceil((height - new_height) / 2))
    bottom = height - int(np.floor((height - new_height) / 2))

    if len(img.shape) == 2:
        center_cropped_img = img[top:bottom, left:right]
    else:
        center_cropped_img = img[top:bottom, left:right, ...]

    return center_cropped_img

def sep_lungs_from_mask(init_lung_segment, ct_im):
    
    #######This function can segments the left/right lungs and does not return them -- not used in output atm but good to flag for future dev ########

    lungs_im = np.zeros(ct_im.shape)
    seed_im = np.zeros(lungs_im.shape)


    init_lung_segment = ndimage.binary_fill_holes(init_lung_segment, structure=np.ones((5, 5, 3)))
    temp_vol = np.ones(ct_im.shape)
    temp_vol[:int(ct_im.shape[0] / 2), :, :] = 1
    temp_vol[int(ct_im.shape[0] / 2):, :, :] = 2
    lungs_im = init_lung_segment * temp_vol

    # Check if seeds are inside the lungs
    right_seed_1 = ndimage.measurements.center_of_mass(lungs_im == 1)
    right_seed_2 = deepcopy(right_seed_1)
    print('Right lung seed point: ', right_seed_1)
    print(ct_im[int(right_seed_1[0]), int(right_seed_1[1]), int(right_seed_1[2])])
    while ct_im[int(right_seed_1[0]), int(right_seed_1[1]), int(right_seed_1[2])] > -600 or \
            ct_im[int(right_seed_2[0]), int(right_seed_2[1]), int(right_seed_2[2])] > -600:

        if ct_im[int(right_seed_1[0]), int(right_seed_1[1]), int(right_seed_1[2])] > -600:
            right_seed_1 = list(right_seed_1)
            right_seed_1[2] = right_seed_1[2] + 1
            right_seed_1 = tuple(right_seed_1)

        if ct_im[int(right_seed_2[0]), int(right_seed_2[1]), int(right_seed_2[2])] > -600:
            right_seed_2 = list(right_seed_2)
            right_seed_2[2] = right_seed_2[2] - 1
            right_seed_2 = tuple(right_seed_2)

        print('Right lung seed point: ', right_seed_1)
        print(ct_im[int(right_seed_1[0]), int(right_seed_1[1]), int(right_seed_1[2])])

    #####
    left_seed_1 = ndimage.measurements.center_of_mass(lungs_im == 2)
    left_seed_2 = deepcopy(left_seed_1)
    print('Left lung seed point: ', left_seed_1)
    print(ct_im[int(left_seed_1[0]), int(left_seed_1[1]), int(left_seed_1[2])])
    while ct_im[int(left_seed_1[0]), int(left_seed_1[1]), int(left_seed_1[2])] > -600 or \
            ct_im[int(left_seed_2[0]), int(left_seed_2[1]), int(left_seed_2[2])] > -600:

        if ct_im[int(left_seed_1[0]), int(left_seed_1[1]), int(left_seed_1[2])] > -600:
            left_seed_1 = list(left_seed_1)
            left_seed_1[2] = left_seed_1[2] + 1
            left_seed_1 = tuple(left_seed_1)

        if ct_im[int(left_seed_2[0]), int(left_seed_2[1]), int(left_seed_2[2])] > -600:
            left_seed_2 = list(left_seed_2)
            left_seed_2[2] = left_seed_2[2] - 1
            left_seed_2 = tuple(left_seed_2)

        print('Left lung seed point: ', left_seed_1)
        print(ct_im[int(left_seed_1[0]), int(left_seed_1[1]), int(left_seed_1[2])])

    seed_im[int(right_seed_1[0]), int(right_seed_1[1]), int(right_seed_1[2])] = 1
    seed_im[int(right_seed_2[0]), int(right_seed_2[1]), int(right_seed_2[2])] = 1
    seed_im[int(left_seed_1[0]), int(left_seed_1[1]), int(left_seed_1[2])] = 2
    seed_im[int(left_seed_2[0]), int(left_seed_2[1]), int(left_seed_2[2])] = 2

    lungs_4_ws = ct_im * init_lung_segment

    # TODO Check the size of the lungs after the segmentation, if one is much bigger than the other repeat point localization
    # TODO put more than one seed point in each of the lungs

    seed_im_itk = sitk.GetImageFromArray(seed_im.astype('int'))
    lungs_4_ws_itk = sitk.GetImageFromArray(lungs_4_ws.astype('int'))

    ws_itk = sitk.MorphologicalWatershedFromMarkers(lungs_4_ws_itk, seed_im_itk, markWatershedLine=True,
                                                    fullyConnected=False)
    ws = sitk.GetArrayFromImage(ws_itk)
    lungs_separated = ws * init_lung_segment

    # Morphological operations as post-processing
    lungs_separated_left = np.zeros(lungs_separated.shape)
    lungs_separated_right = np.zeros(lungs_separated.shape)

    lungs_separated_right[lungs_separated == 1] = 1
    lungs_separated_left[lungs_separated == 2] = 1

    lungs_separated_right = get_largest_CC(lungs_separated_right)
    lungs_separated_left = get_largest_CC(lungs_separated_left)

    lungs_separated_right[lungs_separated_right > 0] = 1
    lungs_separated_left[lungs_separated_left > 0] = 1

    lungs_separated = lungs_separated_right + lungs_separated_left*2

    return lungs_separated

def center_uncrop(img, new_width=None, new_height=None):

    width = img.shape[1]
    height = img.shape[0]

    center_uncropped_img = np.zeros((new_height, new_width))

    if new_width is None:
        new_width = max(width, height)
    elif new_width < width:
        print('Error in width image uncropping!')

    if new_height is None:
        new_height = max(width, height)
    elif new_height < height:
        print('Error in height image uncropping!')

    left = int(np.ceil((new_width - width) / 2))
    right = new_width - int(np.floor((new_width - width) / 2))

    top = int(np.ceil((new_height - height) / 2))
    bottom = new_height - int(np.floor((new_height - height) / 2))

    if len(img.shape) == 2:
        center_uncropped_img[top:bottom, left:right] = img
    else:
        center_uncropped_img[top:bottom, left:right, ...] = img

    return center_uncropped_img


