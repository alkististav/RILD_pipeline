import os
import SimpleITK as sitk
import nibabel as nib
import glob
import numpy as np
import itertools
import sys
import DicomRTTool as rt

from RILD_pipeline.utils import calculate, crop_and_mask, convert
from RILD_pipeline.operations import Patient, DB, threading
from nipype.interfaces.dcm2nii import Dcm2niix
from nipype.interfaces import niftyreg


import glob
import os
import shutil
import tempfile
import nibabel as nib
import numpy as np


import torch
from monai.transforms import LoadImage
from monai.apps import download_and_extract
from monai.config import print_config
from monai.utils import set_determinism, GridSampleMode, GridSamplePadMode
from monai.networks.nets import SegResNet
from monai.data.nifti_saver import NiftiSaver
from monai.inferers import sliding_window_inference
from monai.losses import DiceLoss, DiceCELoss
from monai.metrics import DiceMetric
from monai.networks.layers import Norm
from monai.networks.nets import UNet
from monai.data import CacheDataset, DataLoader, Dataset, decollate_batch
from monai.transforms import (
    AsDiscreted,
    SaveImaged,
    Invertd,
    AddChanneld,
    Compose,
    LoadImaged,
    Orientationd,
    ScaleIntensityRanged,
    ToTensord,
    KeepLargestConnectedComponentd,
)
from monai.handlers.utils import from_engine
from monai.utils import first, set_determinism

import glob
import os
import shutil
import tempfile
import nibabel as nib
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import argparse
import nibabel
from PIL import Image 

def format(patient, parameters):
    
    patient_name = patient.patient_name
    timepoints = patient.ct_timepoints
    patient.log('Starting formatting process for patient ' + patient_name)
    
    # FIRST PART: 1.1
    # DICOM 2 NIFTI CONVERSION -- if nifti_formatting is False, call dcm2nii script
    if patient.nifti_formatting == True:
        patient.log('Images in nifti format exist for patient ' + patient_name)
    else:
        patient.log('Images in nifti do not exist for patient ' + patient_name + '/n Calling dcm2niix for DICOM to nifti conversion')
        convert_dcm2nii(patient, parameters)
    
    patient.log('Starting main tissue segmentation for patient ' + patient_name)
    for tissue in ["airways", "lung", "tissue"]:
        patient.log('Initialising ' + tissue + ' segmentation')
        segment(patient, parameters, tissue)
        patient.log(tissue  + ' segmentation done')
    
    # SECOND PART: 1.2 
    patient.log('Vessel segmentation is initialising for patient ' + patient_name)
    M = parameters.format['M']
    sigma = parameters.format['sigma']
    
    #technically you first need lungs here
    segment_vessels(patient, M, sigma) #this makes the function better used standalone
    
    #3RD PART
    register_bones(patient, parameters)
    

def convert_dcm2nii(patient, parameter):

        for timepoint in patient.ct_timepoints:
            print(timepoint)
            dicom_file_loc = os.path.join(patient.dirs['orig_image'], timepoint)
            # dicom_files = glob.glob(dicom_file_loc + '/*.dcm*')
            output_file_loc = os.path.join(patient.dirs['orig_base_image'])
            print('entering converter')
            dcm2nii_converter_run(dicom_file_loc, output_file_loc)
            print('done')
            
        patient.log('{}Nifti conversion is complete for timepoints, moving on to planning data '  + patient.patient_name)
        
        planning_data_path = os.path.join(patient.dirs['orig_image'], 'planning')
        # print(planning_data_path)
        
        #planning data
        dicom_file_loc = os.path.join(planning_data_path, 'planning')
        print(dicom_file_loc)
        output_file_loc = os.path.join(patient.dirs['orig_base_image'])
        dcm2nii_converter_run(dicom_file_loc, output_file_loc)
        
        #dose data
        
        dicom_file_loc = os.path.join(planning_data_path, 'dose')
        output_file_loc = os.path.join(patient.dirs['orig_radio_image'])
        dcm2nii_converter_run(dicom_file_loc, output_file_loc)
        
        # correct the header
        dicom_dose = glob.glob(dicom_file_loc +  '/dose.dcm')[0]
        nii_dose = glob.glob(patient.dirs['orig_radio_image'] +  '/dose.nii.gz')[0]
        convert.correct_dose_header(dicom_dose, nii_dose)
        print("Dose Header Corrected")
        convert.scale_dose(dicom_dose, nii_dose)
        print("Dose Scaled")

        
        #plan data
        
        #dicom_file_loc = os.path.join(planning_data_path, 'plan')
        #output_file_loc = os.path.join(patient.dirs['orig_radio_image'])
        #dcm2nii_converter_run(dicom_file_loc, output_file_loc)
        #os.rename(os.path.join(patient.dirs['orig_radio_image'], 'plan.nii.gz'), os.path.join(patient.dirs['orig_radio_image'], 'PTV.nii.gz')) #CHANGE THE FILENAME -- dcm2niix doesnt support naming output file 
        
        
        patient.nifti_formatting = True
        patient.log('Nifti conversion is done for patient '  + patient.patient_name)
        
        #what should we do about DICOMs I dont know

def dcm2nii_converter_run(dicom_files_loc, output_file_loc, merge_image = True, out_filename_tag = '%f'):
            
        converter = Dcm2niix()  
        converter.inputs.source_dir = dicom_files_loc
            #converter.inputs.gzip_output = False
        converter.inputs.output_dir = output_file_loc
        converter.inputs.out_filename = out_filename_tag
        converter.inputs.merge_imgs = merge_image
        converter.run()
        
def convert_rt(patient):
    """
    Converts the structures in the RTSTRUCT file to NIFTI binary masks.
    :param patient: patient object
    """

    # set contour names and associations
    contour_names = ['spinal_canal', 'PTV']
    SCassociations = {'SpinalCanal': 'spinal_canal', 'spinal_canal': 'spinal_canal', 'spinalcanal': 'spinal_canal'}
    ptv_nums = np.arange(0,100)
    ptv_keys = ['PTV' + str(ptv_num) for ptv_num in ptv_nums]
    ptv_keys.append('PTV')
    ptv_values = ['PTV' for ptv_num in ptv_nums + 1]
    PTVassociations = dict(zip(ptv_keys, ptv_values))
    association_dictionaries = [SCassociations, PTVassociations]

    for k, contour_name in enumerate(contour_names):
        # initialise DicomReader and find dicom planning folder
        DicomReader = rt.DicomReaderWriter()
        DicomReader.walk_through_folders(os.path.join(patient.dirs['orig_image'], 'planning'))

        # set contour names and associations
        DicomReader.set_contour_names_and_associations(Contour_Names=[contour_name],
                                                       associations=association_dictionaries[k])
        # find the file with the contours
        indices = DicomReader.which_indexes_have_all_rois()
        DicomReader.set_index(indices[0])  # there is only one struct file
        DicomReader.get_images_and_mask()
        # load mask
        mask = DicomReader.mask
        mask_sitk_handle = DicomReader.annotation_handle
        sitk.WriteImage(mask_sitk_handle, os.path.join(patient.dirs['orig_radio_image'], contour_name + '.nii.gz'))
        
            
def segment_vessels(patient, M = 1, sigma = 1, ct_timepoint = all):

        mask_path = patient.dirs['orig_seg_image'] 
        scan_path = patient.dirs['orig_base_image']
        
        if ct_timepoint == all:
            for timepoint in patient.ct_timepoints:
                try:
                    mask_image = glob.glob(mask_path + '/' + timepoint + '_lungs.nii.gz')[0]
                    scan_image = glob.glob(scan_path + '/' + timepoint + '.nii.gz')[0]
                except: 
                    patient.log('Images required for vessel segmentation not found for ' + timepoint + ' continuing for other timepoints')
                    continue

                img = nib.load(scan_image)
                img = calculate.clip_intensities(img)
                img_array_nii = img.get_fdata()
                mask = nib.load(mask_image)
                mask_array_nii = mask.get_fdata()

                sitk_img = sitk.GetImageFromArray(img_array_nii)

                # Apply smoothing filter
                gaussian = sitk.SmoothingRecursiveGaussianImageFilter()
                gaussian.SetSigma(float(sigma))
                smooth_img = gaussian.Execute(sitk_img)

                # Compute the vesselness
                vessel = sitk.ObjectnessMeasureImageFilter()
                vessel.SetObjectDimension(M)
                vesselness = vessel.Execute(smooth_img)
                arr = sitk.GetArrayFromImage(vesselness)/4
                arr[mask_array_nii == 0] = float(np.amin(arr))

                vesselness_file=nib.Nifti1Image(arr, affine=img.affine, header=img.header)
                nib.save(vesselness_file, os.path.join(mask_path,  timepoint +'_vesselness.nii.gz'))

        patient.log('Finished vessel segmentation for for patient ' + patient.patient_name)


def register_bones(patient, parameters, bone_reg = True):

        segmentations = ['lungs', 'airways', 'vesselness']
        
        seg_file_path = os.path.join(patient.dirs['orig_seg_image'])
        image_file_path = patient.dirs['orig_base_image']
        aff_path = patient.dirs['aff_path']
        radio_path = patient.dirs['orig_radio_image']
        
       #planning first#
        baseline_im = glob.glob(patient.dirs['orig_base_image'] + '/' + 'baseline.nii*')[0]
        plan_im = glob.glob(patient.dirs['orig_base_image'] + '/' + 'planning.nii*')[0]
        plan_res = os.path.join(patient.dirs['aligned_base_image'], "planning.nii.gz")
        plan_aff = os.path.join(aff_path, "psc_planning2baseline.txt")
        
        # register#
        node = niftyreg.RegAladin()
        node.inputs.ref_file = baseline_im
        node.inputs.flo_file = plan_im
        node.inputs.rig_only_flag = True
        node.inputs.ln_val = 5
        node.inputs.lp_val = 4
        node.inputs.omp_core_val = 40
        node.inputs.v_val = 100
        node.inputs.aff_file = plan_aff
        node.inputs.res_file = plan_res

        patient.log(patient.patient_name + ' UPDATE: ' + 'Niftyreg command to run: ' + node.cmdline)
        node.run()

        patient.log(
            patient.patient_name + ' UPDATE: ' + 'Registration of planning scan to baseline scan in patient is done.') 
        
        
        # transform dose
        aff_res = glob.glob(plan_aff)[0]
        orig = glob.glob(patient.dirs['orig_radio_image'] + '/dose.nii.*')[0]

        node = niftyreg.RegTransform()
        node.inputs.upd_s_form_input = orig
        node.inputs.out_file = patient.dirs['aligned_radio_image'] + '/dose.nii.gz'
        node.inputs.upd_s_form_input2 = aff_res
        patient.log(patient.patient_name + ' UPDATE: ' + 'Niftyreg command to run: ' + node.cmdline)
        node.run()

        patient.log(
            patient.patient_name + ' UPDATE: ' + 'Transform of dose to baseline space in patient ' + patient.patient_name + ' is done.')


        # now for the PTV
        ptv_file_path = patient.dirs['orig_radio_image'] + '/PTV.nii.gz'
        if os.path.isfile(ptv_file_path) is False:
            patient.log(
                patient.patient_name + ' UPDATE: ' + 'PTV of patient ' + patient.patient_name + ' not found. Continuing without PTV.')
        elif not os.path.isfile(patient.dirs['orig_radio_image'] + '/PTV.nii'):
            patient.log(
                patient.patient_name + ' UPDATE: ' + 'PTV of patient ' + patient.patient_name + ' not found. Continuing without PTV.')
        else:
            aff_res = glob.glob(plan_aff)[0]
            orig = glob.glob(patient.dirs['orig_radio_image'] + '/PTV.nii*')[0]

            node = niftyreg.RegTransform()
            node.inputs.upd_s_form_input = orig
            node.inputs.out_file = patient.dirs['aligned_radio_image'] + '/PTV.nii.gz'
            node.inputs.upd_s_form_input2 = aff_res
            patient.log(patient.patient_name + ' UPDATE: ' + 'Niftyreg command to run: ' + node.cmdline)
            node.run()

            patient.log(
                patient.patient_name + ' UPDATE: ' + 'Transform of PTV to baseline space in patient ' + patient.patient_name + ' is done.')

        ##THIS DETERMINES WHETHER BONE IMAGES ARE USED FOR REGISTRATION OR NOT###

        if bone_reg == True:
            log_tag = 'bone'
        else:
            log_tag = 'CT'
        
        if bone_reg == True:
            ref_file = glob.glob(patient.dirs['orig_seg_image'] + '/' + 'baseline_bones.nii*')[0]
            shutil.copy(ref_file, patient.dirs['aligned_seg_image'] + '/' + 'baseline_bones.nii.gz')
            shutil.copy(baseline_im, patient.dirs['aligned_base_image'] + '/' + 'baseline.nii.gz')
        elif bone_reg == False:
            ref_file = baseline_im
            shutil.copy(baseline_im, patient.dirs['aligned_base_image'] + '/' + 'baseline.nii.gz')
            

        for seg in segmentations:
            shutil.copy(seg_file_path + '/' + 'baseline_' + seg + '.nii.gz', patient.dirs['aligned_seg_image'] + '/' + 'baseline_' + seg + '.nii.gz')

        for followup in patient.ct_timepoints[1:]:

            aff_res = os.path.join(aff_path, "psc_" + followup + "2baseline.txt")

            if bone_reg == True:
                flo_path = glob.glob(os.path.join(seg_file_path, followup + "_bones.nii*"))[0]
                res_path = os.path.join(patient.dirs['aligned_seg_image'],  followup + "_bones.nii.gz")
            else:
                flo_path = glob.glob(os.path.join(image_file_path, followup + ".nii*"))[0]
                res_path = os.path.join(patient.dirs['aligned_base_image'], followup + ".nii.gz")

            # register#
            node = niftyreg.RegAladin()
            node.inputs.ref_file = ref_file
            node.inputs.flo_file = flo_path
            node.inputs.rig_only_flag = True
            node.inputs.ln_val = 5
            node.inputs.lp_val = 4
            node.inputs.omp_core_val = 40
            node.inputs.v_val = 100
            node.inputs.aff_file = aff_res
            node.inputs.res_file = res_path
            
            patient.log(patient.patient_name + ' UPDATE: ' + 'Niftyreg command to run: ' + node.cmdline)
            node.run()

            patient.log(
                patient.patient_name + ' UPDATE: ' + 'Registration of ' + followup + ' ' + log_tag + ' to baseline ' + log_tag + ' in patient is done.')      

            # transform#

            try:
                aff_res = glob.glob(aff_path + '/' + "psc_" + followup + "2baseline.txt")[0]
            except:
                patient.log(
                    patient.patient_name + ' UPDATE: ' + 'Affine required for transformation not found for ' + followup + ' continuing for other timepoints.')
                continue

            if bone_reg == True:  # loop through all segmentations AND CT
                flo_im_path = glob.glob(os.path.join(image_file_path, followup + ".nii*"))[0]
                ct_res = os.path.join(patient.dirs['aligned_base_image'], followup + ".nii.gz")

                node = niftyreg.RegTransform()
                node.inputs.upd_s_form_input = flo_im_path
                node.inputs.out_file = ct_res
                node.inputs.upd_s_form_input2 = aff_res
                patient.log(patient.patient_name + ' UPDATE: ' + 'Niftyreg command to run: ' + node.cmdline)
                node.run()

                patient.log(
                    patient.patient_name + ' UPDATE: ' + 'Transform of ' + followup + ' CT to baseline space in patient ' + patient.patient_name + ' is done.')

                
            else:  # loop through segmentations only
                pass

            for seg in segmentations:
                flo_path = glob.glob(os.path.join(seg_file_path,   followup + '_' + seg + ".nii*"))[0]
                res_path = os.path.join(patient.dirs['aligned_seg_image'], followup + '_' + seg + ".nii.gz")

                node = niftyreg.RegTransform()
                node.inputs.upd_s_form_input = flo_path
                node.inputs.out_file = res_path
                node.inputs.upd_s_form_input2 = aff_res
                patient.log(patient.patient_name + ' UPDATE: ' + 'Niftyreg command to run: ' + node.cmdline)
                node.run()

                patient.log(
                    patient.patient_name + 'UPDATE: ' + 'Transform of ' + followup + ' ' + seg + ' to baseline space in patient is done.')


def segment(patient, parameters, class_name, post_process=True):
    
    '''
    class_name : str, options: "airways" "lungs" "bones"
    '''
    patient.log('Initialising the {} segmentation'.format(class_name))
    
#     if class_name == 'lung':
#         lung_segment(patient, parameters)
    
#     else:
    monai_segment(patient, parameters, class_name, post_process)
        
    patient.log('{} segmentation is done'.format(class_name))
    
def monai_segment(patient, parameters, class_name, post_process):
    
    output_dir = patient.dirs['orig_seg_image'] 
    image_dir = patient.dirs['orig_base_image']
    model_dir = patient.dirs['models_path']
    
    out_channels = 2

    classes_names = [class_name]
    model_name = parameters.models['format.segment'][class_name]



    # Load test data
    timepoint_files = [timepoint + ".nii*" for timepoint in patient.ct_timepoints] 
    images = sorted(list(itertools.chain.from_iterable([glob.glob(f'{image_dir}/**/{f}', recursive=True) for f in timepoint_files])))


    image_dict = [
        {"image": image_name}
        for image_name in images]




    """## Set deterministic training for reproducibility"""

    set_determinism(seed=0)

    '''
    # 0 - Background
    # 1 - airways
    # 2 - lungs
    # 3 - Skeleton/Bones
    '''
    
    if class_name == "airways":
        test_transforms = Compose(
        [
            LoadImaged(keys=["image"]),
            AddChanneld(keys=["image"]),
            Orientationd(keys=["image"], axcodes="RAS"),
            ScaleIntensityRanged(
                keys=["image"], a_min=-1000, a_max=300, b_min=0.0, b_max=1.0, clip=True,
            ),
            ToTensord(keys=["image"]),
        ]
    )
        post_transforms = Compose([
            Invertd(
                keys="pred",
                transform=test_transforms,
                orig_keys="image",
                meta_keys="pred_meta_dict",
                orig_meta_keys="image_meta_dict",
                meta_key_postfix="meta_dict",
                nearest_interp=False,
                to_tensor=True,
            ),
            AsDiscreted(keys="pred", argmax=True),
            KeepLargestConnectedComponentd(keys="pred", applied_labels=[1]),
            SaveImaged(keys="pred", meta_keys="pred_meta_dict", output_postfix=class_name, resample=False, separate_folder=False,
                       output_dir=output_dir),
        ])
    elif class_name == "lungs":
        
        test_transforms = Compose(
        [
            LoadImaged(keys=["image"]),
            AddChanneld(keys=["image"]),
            Orientationd(keys=["image"], axcodes="RAS"),
            ScaleIntensityRanged(
                keys=["image"], a_min=-1000, a_max=300, b_min=0.0, b_max=1.0, clip=True,
            ),
            ToTensord(keys=["image"]),
        ])
            
        post_transforms = Compose([
            Invertd(
                keys="pred",
                transform=test_transforms,
                orig_keys="image",
                meta_keys="pred_meta_dict",
                orig_meta_keys="image_meta_dict",
                meta_key_postfix="meta_dict",
                nearest_interp=False,
                to_tensor=True,
            ),
            AsDiscreted(keys="pred", argmax=True),
            SaveImaged(keys="pred", meta_keys="pred_meta_dict", output_postfix=class_name, resample=False, separate_folder=False,
                       output_dir=output_dir),
        ])
        
    elif class_name == "bones":
        test_transforms = Compose(
        [
            LoadImaged(keys=["image"]),
            AddChanneld(keys=["image"]),
            Orientationd(keys=["image"], axcodes="RAS"),
            ScaleIntensityRanged(
                keys=["image"], a_min=-1000, a_max=1000, b_min=0.0, b_max=1.0, clip=True,
            ),
            ToTensord(keys=["image"]),
        ]
    )
        post_transforms = Compose([
            Invertd(
                keys="pred",
                transform=test_transforms,
                orig_keys="image",
                meta_keys="pred_meta_dict",
                orig_meta_keys="image_meta_dict",
                meta_key_postfix="meta_dict",
                nearest_interp=False,
                to_tensor=True,
            ),
            AsDiscreted(keys="pred", argmax=True),
            SaveImaged(keys="pred", meta_keys="pred_meta_dict", output_postfix=class_name, resample=False, separate_folder=False,
                       output_dir=output_dir),
        ])
        
        
    test_ds = CacheDataset(data=image_dict, transform=test_transforms, cache_rate=1.0, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=1, num_workers=0)
        
    dice_metric = DiceMetric(include_background=False, reduction="mean")


    # standard PyTorch program style: create UNet, DiceLoss and Adam optimizer
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = UNet(
        dimensions=3,
        in_channels=1,
        out_channels=out_channels,  # channels, 1 for each organ more background
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2,
        norm=Norm.BATCH,
    ).to(device)

    model.load_state_dict(torch.load(os.path.join(model_dir, model_name), map_location='cpu')) 
    model.eval()

    loader = LoadImage()
    with torch.no_grad():
        for test_data in test_loader:
            test_inputs = test_data["image"].to(device)
            roi_size = (96, 96, 96)
            sw_batch_size = 4
            test_data["pred"] = sliding_window_inference(
                test_inputs, roi_size, sw_batch_size, model)

            test_data = [post_transforms(i) for i in decollate_batch(test_data)]
            test_outputs, test_images = from_engine(["pred", "image"])(test_data)
    
    # Post process lung segmentations
    if post_process == True:
        if class_name == "lungs":
            segmentations = sorted(glob.glob(patient.dirs['orig_seg_image'] + '/*lungs.nii.gz'))
            airway_segmentations = sorted(glob.glob(patient.dirs['orig_seg_image'] + '/*airways.nii.gz'))
            for k, seg in enumerate(segmentations):
                seg_nii = nib.load(seg)
                seg_im = seg_nii.get_fdata()
                affine = seg_nii.affine

                airways_nii = nib.load(airway_segmentations[k])
                scan = nib.load(images[k])

                mask = crop_and_mask.simple_bodymask(scan, airways_nii)
                seg_im[mask == 0] = 0
                processed = crop_and_mask.keep_two_largest_components(seg_im)
                processed_nii = nib.Nifti1Image(processed, affine=affine)
                nib.save(processed_nii, seg)
            

def net_inference(image, model):
    image[image > 1] = 1
    image[image < 0] = 0
    mean = 0.5
    std = 0.35

    image = (image - mean) / std
    images = torch.from_numpy(image).float()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
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
    model_dir = patient.dirs['models_path']
    model_one = parameters.models['format.segment']['lungs_1']
    model_two = parameters.models['format.segment']['lungs_2']

    model_one_path = os.path.join(model_dir, model_one)
    model_two_path = os.path.join(model_dir, model_two)

    timepoints = patient.ct_timepoints

    #############RUNNING MODEL#################################################################################################

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #load models

    model1 = torch.load(model_one_path, map_location='cpu')
    model1.to(device)
    model1.eval()
    model1_new = torch.save(model1, os.path.join(model_dir, 'resaved_model_1'))
    model1 = torch.load(os.path.join(model_dir, 'resaved_model_1'), map_location='cpu')
    model1.to(device)
    model1.eval()
    
    
    model2 = torch.load(model_two_path, map_location='cpu')
    model2.to(device)
    model2.eval()
    model2_new = torch.save(model2, os.path.join(model_dir, 'resaved_model_2'))
    model2 = torch.load(os.path.join(model_dir, 'resaved_model_2'), map_location='cpu')
    model2.to(device)
    model2.eval()
    


    for followup in patient.ct_timepoints[0:]:  

        ct_im_path = glob.glob(os.path.join(patient.dirs['orig_base_image'], followup + ".nii.gz"))[0]
        ct_im_nii = nib.load(ct_im_path)
        ct_im = np.asarray(ct_im_nii.get_data())

        mask_im = crop_and_mask.simple_bodymask(ct_im)
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
        patient.log('Final stages for ' + followup) 
        
        for slice_nr in range(0, ct_4_net.shape[2]):

            temp_mask = np.squeeze(mask_im[:, :, slice_nr])

            if (out_size[0] < temp_mask.shape[1]) or (out_size[1] < temp_mask.shape[0]): 

                image = np.rot90(np.array(Image.fromarray(np.squeeze(ct_4_net[:, :, slice_nr])).resize(
                                    out_size, resample=Image.BICUBIC)), 3).astype('float32')

                output_1 = net_inference(image, model1)
                output_2 = net_inference(image, model2)

                output = output_1 + output_2

                output = torch.argmax(output, dim=1)

                output = output.squeeze(dim=0).cpu().data.numpy()

                label_temp_out[:, :, slice_nr] = np.rot90(np.array(Image.fromarray(output.astype('float')).resize((ct_4_net.shape[0], ct_4_net.shape[1]),
                                                                                   resample=Image.NEAREST)).astype(
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

        patient.log('Saving the image now for ' + followup)
        output_path = os.path.join(patient.dirs['orig_seg_image'], 'lungs_' + followup)

        label_out[smallest_x:largest_x, smallest_y:largest_y, smallest_z:largest_z] = label_temp_out

        label_out_path = output_path
        label_out_nii = nib.Nifti1Image(label_out, ct_im_nii.affine, ct_im_nii.header)
        nib.save(label_out_nii, label_out_path)

        print(followup + ' done')

    print('Finito')

#
# def rescale_all_dose(patient):
#
#     timepoints = patient.ct_timepoints
#     aligned_radio_path = patient.dirs['aligned_radio_image']
#     orig_radio_path = patient.dirs['orig_radio_image']
#     planning_data_path = os.path.join(patient.dirs['orig_image'], 'planning')
#     dicom_file_loc = os.path.join(planning_data_path, 'dose')
#     dicom_dose = glob.glob(dicom_file_loc + '/dose.dcm')[0]
#     aligned_dose = glob.glob(aligned_radio_path + '/dose.nii.gz')[0]
#     orig_dose = glob.glob(orig_radio_path + '/dose.nii.gz')[0]
#
#
#     # rescale plan dose
#     convert.scale_dose(dicom_dose, aligned_dose)
#     convert.scale_dose(dicom_dose, orig_dose)
#
#     for tp in timepoints:
#         nii_dose = glob.glob(patient.dirs['aligned_radio_image'] + 'dose_' + tp + '.nii.gz')[0]
#         convert.scale_dose(dicom_dose, nii_dose)
#         print('Dose rescaled for ' + patient.patient_name + tp)





