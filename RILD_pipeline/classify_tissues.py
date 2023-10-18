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
import random
from torchvision import transforms, utils
import argparse
from utils import infer, nets_arch

def classify_tissues(patient, parameters):

    ############ REFACTOR
    # out_size = (224, 224)
    out_size = (288, 384) # Assume this is hardcoded

    nets_dir = patient.dirs['models_path']
    model_path = parameters.models['classify_tissues']
    
    
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    #grab all the models from list format 
    
#     for i in range(len(model_path)):
#         exec(f'saved_model_{i}_path = os.path.join(nets_dir, model_path[{i}])')
#         exec(f'print(saved_model_{i}_path)')
    
    patient.log(patient.patient_name + ' UPDATE: ' +' Classification model is initialised.')

    data_inference =infer.net_inference(patient,
                                os.path.join(nets_dir, model_path[0]),os.path.join(nets_dir, model_path[1]), os.path.join(nets_dir, model_path[2]),
                                os.path.join(nets_dir, model_path[3]), os.path.join(nets_dir, model_path[4]), os.path.join(nets_dir, model_path[5]),
                                out_size)
