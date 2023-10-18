###READ ME conda install --channel conda-forge nipype
from nipype.interfaces import niftyreg
import numpy as np
import glob
import os
import subprocess


def register(patient):

    base_path = patient.dirs['aligned_base_image']
    radio_path = patient.dirs['aligned_radio_image']

    reg_path = patient.dirs['reg_path']
    stack_path = patient.dirs['aligned_stack_path']
    timepoints=patient.ct_timepoints

    
    #common parameters
    common_params=" -omp 5 -ln 5 -maxit 600 -vel -sx 5 -be 0.001 -le 0 -ssd 0 -ssd 1 --nmi -ssdw 0 1 -ssdw 1 1 -nmiw 2 3.5"
    # echo$(common_params)

    #planning
    # register baseline image to planning image using RegF3D
    # transform dose with baseline.nii and dose plan using RegResample
    # transfrom ptv with baseline and ptv plan using RegResample
    plan_params=" -omp 5 -ln 5 -sx 5 -maxit 500 -vel -be 0.005 -le 0 --nmi" #add to parameter object
    
    patient.log(patient.patient_name + 'UPDATE: ' +' Registration initialising is done. Starting the process.')
    
            
    if not os.path.exists(os.path.join(reg_path, 'planning')):
        os.mkdir(os.path.join(reg_path, 'planning'))
        
    #registration
    # ${F3D_CMD} ${plan_params} -flo ${plan_img} -ref ${base_img} -res ${res_im_name} -cpp ${plan_cpp_name}
    node_plan = niftyreg.RegF3D() # ${F3D_CMD}
    node_plan.inputs.omp_core_val = 5
    node_plan.inputs.ln_val = 5
    node_plan.inputs.sx_val = 5
    node_plan.inputs.maxit_val = 500
    node_plan.inputs.vel_flag = True
    node_plan.inputs.be_val = 0.005
    node_plan.inputs.lp_val = 0
    node_plan.inputs.nmi_flag = True
    #node_plan.inputs.args=plan_params
    node_plan.inputs.ref_file=glob.glob(os.path.join(base_path , timepoints[0] +  ".nii*"))[0]
    node_plan.inputs.flo_file=glob.glob(os.path.join(base_path , "planning" + ".nii*"))[0]
    #The output resampled image
    node_plan.inputs.res_file=os.path.join(reg_path,  'planning', 'reg.nii.gz') #path_to_out=${path_to_data}/lung${pat}/registrations/planning

    #The output CPP file
    node_plan.inputs.cpp_file=os.path.join(reg_path,  'planning', 'cpp' + ".nii.gz") # plan_cpp_name=${path_to_data}/lung${pat}/registrations/planning/cpp.nii.gz
    patient.log(patient.patient_name + 'UPDATE: ' + 'Niftyreg command run: ' + node_plan.cmdline)
    node_plan.run() # -- WORKS
    
    radio_str = ['dose']
    #### for loop and string strip
    for rad in radio_str:
        try:
            node_plan = niftyreg.RegResample() #TRANS_CMD
            node_plan.inputs.ref_file=glob.glob(os.path.join(base_path , timepoints[0] + ".nii*"))[0]
            node_plan.inputs.flo_file=glob.glob(os.path.join(radio_path , rad + ".nii*"))[0]
            node_plan.inputs.out_file=os.path.join(radio_path , rad + '_baseline' + ".nii.gz") #res_file -- output file so set as nii.gz
            node_plan.inputs.trans_file=glob.glob(os.path.join(reg_path , 'planning', 'cpp' + ".nii*"))[0] #cpp_file
            patient.log(patient.patient_name + 'UPDATE: ' + 'Niftyreg command run: ' + node_plan.cmdline)
            node_plan.run() #WORKS
        except Exception:
            patient.log(patient.patient_name + 'UPDATE: ' + 'Could not transform the ' + rad + ' from planning to baseline.' )
            continue
            



#     #### follow ups 3/6/12/24 months for loop and string strip
#     compose dvf  image to  image using RegTransform
#     transform dose with followup3.nii and dose plan using RegResample
#     transfrom ptv with followup3 and ptv plan using RegResample

    for tp in timepoints[1:]: #ignoring baseline in for loop
        patient.log(patient.patient_name + 'UPDATE: ' + 'Initialising registration for timepoint : ' + tp)
        
        """ We can delete this later, atm good to keep for reference
        
        #submit registration
        #${F3D_CMD} ${common_params} -flo ${followup3} -ref ${baseline} -res ${res_im_name} -cpp ${for_cpp_name}
        node_follow_up = niftyreg.RegF3D() # ${F3D_CMD}
        
        #-omp 5 -ln 5 -maxit 600 -vel -sx 5 -be 0.001 -le 0 -ssd 0 -ssd 1 --nmi -ssdw 0 1 -ssdw 1 1 -nmiw 2 3.5
        node_follow_up.inputs.omp_core_val = 5
        node_follow_up.inputs.ln_val = 5
        node_follow_up.inputs.le_val = 0
        node_follow_up.inputs.sx_val = 5
        node_follow_up.inputs.maxit_val = 600
        node_follow_up.inputs.vel_flag = True
        node_follow_up.inputs.ssd_flag = True
        node_follow_up.inputs.be_val = 0.001
        node_follow_up.inputs.lp_val = 0
        node_plan.inputs.args=common_params
    
        node_follow_up.inputs.ref_file=glob.glob(os.path.join(stack_path , "baseline"+ ".nii*"))[0]
        node_follow_up.inputs.flo_file=glob.glob(os.path.join(stack_path , tp + ".nii*"))[0] #make sure this is the correct stacks folder location -- potentially inside segmentations/stacks
        
        #The output resampled image
        node_follow_up.inputs.res_file=os.path.join(reg_path , tp)

        #The output CPP file
        node_follow_up.inputs.cpp_file=os.path.join(reg_path , tp, 'cpp' + ".nii.gz") 
        patient.log(patient.patient_name + 'UPDATE: ' + 'Niftyreg command run: ' + node_follow_up.cmdline)
        """
        
        if not os.path.exists(os.path.join(reg_path, tp)):
            os.mkdir(os.path.join(reg_path, tp))

        cpp_file = os.path.join(reg_path , tp, 'cpp.nii.gz')
        flo_file = glob.glob(os.path.join(stack_path , tp + ".nii*"))[0]
        ref_file = glob.glob(os.path.join(stack_path , "baseline"+ ".nii*"))[0]
        res_file = os.path.join(reg_path , tp, 'reg.nii.gz')


        niftyreg_command_line = "reg_f3d -be 0.001000 -cpp {} -flo {} -le 0.000000 -ln 5 -lp 0 -maxit 600 -omp 5 -ref {} -res {} --ssd -sx 5.000000 -vel".format(cpp_file, flo_file, ref_file, res_file)

        patient.log(patient.patient_name + 'UPDATE: ' + ' Niftyreg command run from terminal: ' + niftyreg_command_line)

        niftyreg_output = subprocess.call(niftyreg_command_line, shell=True)

        patient.log(patient.patient_name + 'UPDATE: ' + str(niftyreg_output))

        #in the cmd line: reg_f3d -be 0.001000 -cpp /mnt/RILD/lung052_drafting/registrations/results/follow_up_03/cpp.nii.gz -flo /mnt/RILD/lung052_drafting/stacks/aligned2baseline/follow_up_03.nii.gz -le 0.000000 -ln 5 -lp 0 -maxit 600 -omp 5 -ref /mnt/RILD/lung052_drafting/stacks/aligned2baseline/baseline.nii.gz -res /mnt/RILD/lung052_drafting/registrations/results/follow_up_03 --ssd -sx 5.000000 -vel

        #node_follow_up.run()

        #TRANSFORM WITH CPP
        #forward
        #${TRANS_CMD} -ref ${base_img} -flo ${fu3_img} -trans ${for_cpp_name} -res ${for_trans}
        patient.log(patient.patient_name + 'UPDATE: ' + 'Tranformation with CPP (forward) initialising for: ' + str(tp))
        node_follow_up = niftyreg.RegResample() #TRANS_CMD
        node_follow_up.inputs.ref_file=glob.glob(os.path.join(base_path , timepoints[0] + ".nii*"))[0]
        node_follow_up.inputs.flo_file=glob.glob(os.path.join(base_path , tp + ".nii*"))[0]
        node_follow_up.inputs.out_file=os.path.join(reg_path , tp , 'for_trans' + ".nii.gz") #this is output file so nii.gz is specified
        node_follow_up.inputs.trans_file=glob.glob(os.path.join(reg_path , tp , 'cpp' + ".nii*"))[0]
        patient.log(patient.patient_name + 'UPDATE: ' + 'Niftyreg command run: ' + node_follow_up.cmdline)
        node_follow_up.run()

        #backward
        #${TRANS_CMD} -ref ${fu3_img} -flo ${base_img} -trans ${back_cpp_name} -res ${back_trans}
        patient.log(patient.patient_name + 'UPDATE: ' + 'Tranformation with CPP (backwards) initialising for: ' + str(tp))
        node_follow_up = niftyreg.RegResample() #TRANS_CMD
        node_follow_up.inputs.ref_file=glob.glob(os.path.join(base_path , tp + ".nii*"))[0]
        node_follow_up.inputs.flo_file=glob.glob(os.path.join(base_path ,timepoints[0]+ ".nii*"))[0]
        node_follow_up.inputs.out_file=os.path.join(reg_path, tp , 'back_trans' + ".nii.gz")
        node_follow_up.inputs.trans_file=glob.glob(os.path.join(reg_path , tp , 'cpp_backward' + ".nii*"))[0]
        patient.log(patient.patient_name + 'UPDATE: ' + 'Niftyreg command run: ' + node_follow_up.cmdline)
        node_follow_up.run()



        #compose dvf
        #${COMP_CMD} -ref ${base_img} -ref2 ${fu3_img} -comp ${plan_cpp_name} ${back_cpp_name} ${plan23months}
        try:
            patient.log(patient.patient_name + 'UPDATE: ' + 'Tranformation with CPP (compose dvf) initialising for: ' + str(tp))
            node_follow_up= niftyreg.RegTransform() #COMP_CMD
            node_follow_up.inputs.ref1_file=glob.glob(os.path.join(base_path ,timepoints[0] + ".nii*"))[0]
            node_follow_up.inputs.ref2_file=glob.glob(os.path.join(base_path , tp + ".nii*"))[0]
            node_follow_up.inputs.comp_input=glob.glob(os.path.join(reg_path , 'planning', 'cpp' + ".nii*"))[0]
            node_follow_up.inputs.comp_input2=glob.glob(os.path.join(reg_path , tp , 'cpp_backward' + ".nii*"))[0]

            node_follow_up.inputs.out_file=os.path.join(reg_path , tp, 'plan2_' + tp + ".nii.gz") #could alternatively be the output
            patient.log(patient.patient_name + 'UPDATE: ' + 'Niftyreg command run: ' + node_follow_up.cmdline)
            node_follow_up.run()
        except Exception:
            patient.log(patient.patient_name + 'UPDATE: ' + 'Could not compose dvf. Continuing to next timepoint.')
            continue

        #transform dose and ptv
        try:
            patient.log(patient.patient_name + 'UPDATE: ' + 'Tranformation with CPP (for dose and ptv) initialising for: ' + str(tp))
            for rad in radio_str:
                #${TRANS_CMD}  -ref ${fu3_img} -flo ${dose_plan} -cpp ${plan23months} -res ${dose_fu3}
                node_follow_up = niftyreg.RegResample() #TRANS_CMD
                node_follow_up.inputs.ref_file=glob.glob(os.path.join(base_path, tp + ".nii*"))[0]
                node_follow_up.inputs.flo_file=glob.glob(os.path.join(radio_path , rad + ".nii*"))[0]
                node_follow_up.inputs.out_file=os.path.join(radio_path , rad + '_' + tp + ".nii.gz")
                node_follow_up.inputs.trans_file=glob.glob(os.path.join(reg_path , tp, 'plan2_' + tp + ".nii*"))[0]
                patient.log(patient.patient_name + 'UPDATE: ' + 'Niftyreg command run: ' + node_follow_up.cmdline)
                node_follow_up.run()
        except Exception:
            patient.log(patient.patient_name + 'UPDATE: ' + 'Could not transform structures for timepoint. Continuing to next timepoint.')
            continue