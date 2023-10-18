
import os
import logging 
import logging.config
import glob
import sys
from . import create_scan_dict, check_directory
from os import path

''' TODO documentation
'''

class Patient():

    def __init__(
        self,
        patient_name:str,
        orig_image_folder:str,
        aligned_image_folder:str,
        parameters:dict(),
        create_log_file:True,
        patients_data_dir:str,
    ):
        super().__init__()

        self.patient_name = patient_name 
        
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(message)s",
            handlers=[
                logging.StreamHandler()
            ]
        )
        
        if create_log_file:
            log_file = os.path.join(patients_data_dir, patient_name, patient_name + '_file.log')
            file_handler = logging.FileHandler(log_file, mode='a')
            
        self.logger_object = logging.getLogger(patient_name)
        logger = self.logger_object 
        logger.addHandler(file_handler)
        
        dirs = dict()
        
        #Main paths for original data, model, misc
        
        dirs['main_dir'] = os.path.join(patients_data_dir, patient_name)
        dirs['orig_image'] = os.path.join(dirs['main_dir'], 'DICOM')
        dirs['models_path'] = os.path.join(patients_data_dir, "models") #this is only referring to the main model directory 
        dirs['radio_image'] = os.path.join(dirs['main_dir'], "radiotherapy/")
        dirs['reg_path'] = os.path.join(dirs['main_dir'], "registrations/results/")
        dirs['aff_path'] = os.path.join(dirs['main_dir'], "affine")
        dirs['tissue_classes'] = os.path.join(dirs['main_dir'], "tissue_classes")
        # Original orientation and aligned image paths
        
        dirs['orig_base_image'] = os.path.join(dirs['main_dir'], 'CT', orig_image_folder)
        dirs['orig_seg_image'] = os.path.join(dirs['main_dir'], 'segmentations', orig_image_folder)
        dirs['orig_stack_path'] = os.path.join(dirs['main_dir'], "stacks", orig_image_folder)
        dirs['orig_radio_image'] = os.path.join(dirs['main_dir'], "radiotherapy", orig_image_folder)
        
        dirs['aligned_base_image'] = os.path.join(dirs['main_dir'], 'CT', aligned_image_folder)
        dirs['aligned_seg_image'] = os.path.join(dirs['main_dir'], 'segmentations', aligned_image_folder)
        dirs['aligned_stack_path'] = os.path.join(dirs['main_dir'], "stacks", aligned_image_folder)
        dirs['aligned_radio_image'] = os.path.join(dirs['main_dir'], "radiotherapy", aligned_image_folder)
        

        self.dirs = dirs
        
        
        self.log_save = True
        self._stages = 0
        self.nifti_formatting = True
        #create a logger that writes both to log file and terminal: this always keeps appending
        
        
        #decide if it needs dicom conversion or not
        if len(os.listdir(dirs['orig_base_image'])) > 0:
            self.nifti_formatting = True
            
        #set ct_timepoints
#         print(os.listdir(dirs['orig_image']))
        
        ct_timepoints = [ct_name.split(os.extsep, 1)[0] for ct_name in os.listdir(dirs['orig_image']) if not ct_name.startswith('.')]
        ct_timepoints = [ct_name for ct_name in ct_timepoints if 'planning' not in ct_name]
          
        #self._ct_timepoints = ct_timepoints
        self.ct_timepoints = ct_timepoints
        #self.ct_scan_timepoints = create_scan_dict(self._ct_timepoints) #error here
        
        
        #set the tissue lists
#         self.tissue_list = parameters.tissue_types
        
        #check if models paths exist and have the models referenced in parameter file TODO
        for model_function, model in parameters.models.items():
            self.log('searching for models for ' + model_function)
            
            if isinstance(model , dict):
                for model_key in model.keys():
                    model_file = os.path.join(dirs['models_path'], model[model_key])
                    if os.path.isfile(model_file) == False:
                        self.log('WARNING: for function ' + model_function + ' model in path ' + model_file + ' does not exist. Upload to fix')
                    else:
                        self.log('For function ' + model_function + ' model in path ' + model_file + ' exists.')
            if isinstance(model , list):
                for i in range(len(model)):
                    model_file = os.path.join(dirs['models_path'], model[i])
                    if os.path.isfile(model_file) == False:
                        self.log('WARNING: for function ' + model_function + ' model in path ' + model_file + ' does not exist. Upload to fix')
                    else:
                        self.log('For function ' + model_function + ' model in path ' + model_file+ ' exists.')

    def log(self, message, error = None):
        
        if error != None:
            self.logger_object.setLevel(logging.ERROR)
            self.logger_object.error(message)
            self.logger_object.setLevel(logging.INFO)
            
        self.logger_object.info(message)


    @property
    def stages(self):
        return self._stages

    @stages.setter
    def stages(self, value):
        self.stages = value
        self.logger.log('Finished processing stage ' + processing_stages[value])
        getattr(RILD_DB, self.patient_name).add(self.patient_name, self.stages)   ##ask Haroon for dynamic link?
        
        if value == 5:
            self.finish_patient_process()
   
    def finish_patient_process(self):
        self.log('Finished processing for patient ' + self.patient_name)
        #flag if completely finished or not, maybe a dictionary that can be updated or 
        ##FIX
        self.logger.shutdown()#
        #queue 

    @property
    def dirs(self):
        return self._dirs
    
    @dirs.setter
    def dirs(self, dirs):
        print(dirs)
        check_directory(dirs, self)
        self._dirs = dirs
    
    @property
    def tissue_list(self):
        return self._tissue_list
    
    @tissue_list.setter
    def tissue_list(self, tissue_list):
        #make sure they are ordered ascending, from baseline to longest follow up date
        self._tissue_list = tissue_list

    @property
    def ct_timepoints(self):
        return self._ct_timepoints
    
    @ct_timepoints.setter
    def ct_timepoints(self, timepoint_list):
        #make sure they are ordered ascending, from baseline to longest follow up date
        timepoint_list.sort()
        self._ct_timepoints = timepoint_list

        