import nibabel as nib
import numpy as np
import pandas as pd
import glob
import os


def dose_per_class(patient, include_ptv=True): 
    
    class_path = patient.dirs['tissue_classes']
    radio_path = patient.dirs['aligned_radio_image']
    timepoints = patient.ct_timepoints
    print(timepoints)
#     names = [patient.patient_name, '']
    classes_names = ['Default', 'Ground Glass', 'GG/nodular', 'Mostly solid', 'Opacity', 'Total Volume']
    
    mindose = 5
    maxdose = 85
    bands = []

    for step in range(mindose, maxdose, 5):
        if step == 5:
            bands.append('dose <=' + str(step))
        else:
            bands.append(str(step - 5) + '< dose <=' + str(step))
    bands.append('overall')
            
    for tp in timepoints:  
        # set up dataframe
        df = pd.DataFrame(columns = classes_names, index = bands)
        
        # load
        tissue_path = glob.glob(os.path.join(class_path , tp + ".nii*"))[0]
        dose_path = glob.glob(os.path.join(radio_path , 'dose_' + tp + ".nii*"))[0]
        label_nib = nib.load(tissue_path)
        dose_im = nib.load(dose_path).get_fdata()
        
        # extract pixdim
        label = label_nib.get_fdata()
        label = np.rint(label)
        label = np.int_(label)
        total_lungs = len(np.argwhere(label > 0))
        label_header = label_nib.header
        pixdim = label_header['pixdim']
        dx = pixdim[1]
        dy = pixdim[2]
        dz = pixdim[3]
        
        df.at['overall', 'Total Volume'] = round(dx * dy * dz * np.count_nonzero(label), 4)

        ############################
        # ORIGINAL LABELS
        #     1    "Default"
        #     2    "Default"
        #     3    "GG with nodular component"
        #     4    "Opacity"
        #     5    "Reticulation/HC"
        #     6    "Mostly solid "
        #     7    "Ground Glass"
        #     8    "Linear opacity"
        #     9    "Other"
        #    10    "Pleural Effusion"
        ############################
        # label = np.around(label)
        # label[label == 2] = 1  # left lung and right have the same label
        # label[label == 5] = 0
        # label[label == 7] = 2
        # label[label == 8] = 5  # Linear opacity and opacity have the same label
        # label[label == 10] = 5  # Pleural effusion and opacity have the same label
        # label[label == 4] = 5
        # label[label == 6] = 4
        # label[label > 5] = 0
        # label[label < 0] = 0

        ############################
        # FINAL LABELS
        #     1    "Default"
        #     2    "Ground Glass"
        #     3    "GG with nodular component"
        #     4    "Mostly solid "
        #     5    "Opacity"
        ############################
        
        
        for tissue_class in range(1, 6):
            class_mask = np.where(label == tissue_class, 1, 0)
            df.at['overall', classes_names[tissue_class-1]] = round(dx * dy * dz * np.count_nonzero(class_mask), 4)
            for dose_count, step in enumerate(range(mindose, maxdose, 5)):
                if step == 5:
                    dose_mask = np.where((dose_im <= step), 1, 0)
                    matches = np.argwhere((dose_mask == 1) & (class_mask == 1))
                    df.at[bands[dose_count], classes_names[tissue_class-1]] = round(dx * dy * dz * len(matches), 4)
                    df.at[bands[dose_count], 'Total Volume'] = round(dx * dy * dz * np.count_nonzero(dose_mask), 4)
                else:
                    dose_mask = np.where(((step - 5) < dose_im) & (dose_im <= step), 1, 0)
                    matches = np.argwhere((dose_mask == 1) & (class_mask == 1))
                    df.at[bands[dose_count], classes_names[tissue_class-1]] = round(dx * dy * dz * len(matches), 4)
                    df.at[bands[dose_count], 'Total Volume'] = round(dx * dy * dz * np.count_nonzero(dose_mask), 4)
        print(df)
        df.to_csv(os.path.join(class_path, tp + '.csv'),sep=',')

