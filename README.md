# Pipeline for Analysis of Radiation Induced Lung Damage


## USE: set up 

1. Clone the RILD repository and checkout the branch you want to work on, suggested: `main`


2. RILD Docker

Docker contains the command line interfaces and the python environment that is required to run the RILD pipeline. Advantages of containarising the environment
    - same dependencies across researchers/users of RILD package
    - niftyreg and dcm2niix are algorithms that require a backend, and building these packages in Linux/Mac systems are time-demanding
    - easy to use in the same command line that the RILD pipeline is run
    
NOTE: the files you create/change inside the Docker container are seen outside-- because Docker does not create a new folder environment, it wraps around the already existing system. This is the same logic for copy/pasting your test files inside the folder that can be seen from 
<your local path for RILD code data etc>:/mnt/RILD
    
### Docker user:


1. get docker in your environment, depending on your operating system, following the instructions:
https://docs.docker.com/get-docker/

2. Get the folder that contains associated Docker files
```rild_docker/env.yaml
   rild_docker/Dockerfile
```

3. Go inside the folder, and run the docker build command:
a. on an Intel chip Mac/Linux:
``` docker build . -t dev/rild_docker:v2 ```
b. on an M1 chip Mac:
```docker build . --platform linux/amd64 -t dev/rild_docker:v2 ```
Possible reasons for failure:
        1. not enough space: this can make your build quit without any clear error message besides something like "pip failed". Make sure you have enough memory allocated to your Docker, if you have Docker desktop you can check that here. Also make sure you delete Docker images you no longer use, by checking `docker image ls`
        2. Connection issues: that can show up in timeout or pipe broke error messages. Run docker build command with `--network` tag to ensure stable connection over a proxy network

4. Run the Docker image:
   !!! PAY ATTENTION !!! It is very important that the path you use before `:/mnt/RILD` in the command line contains both /RILD_pipeline_start folder, individual data folders for participants, and /models folder
   
    option a) access the command line interface 
  ``` docker run --rm -it --publish 8888:8888 -v <your local path for RILD code data etc>:/mnt/RILD  dev/rild_docker:v2 bash ```
    
    option b) access it with jupyter lab
     ``` docker run --rm -it --publish 8888:8888 -v <your local path for RILD code data etc>:/mnt/RILD  dev/rild_docker:v2 jupyter notebook --port=8888 --ip=0.0.0.0```
    

5. Once you are inside the Docker (after running docker run), there are some things you need to run. (unfortunately, currently in fix)
    a. installing RILD_pipeline module and saving this as a new Docker image (you can follow the same steps for any module/library you realise you needed.)
    1. check your running container ID and container name
    `docker PS` 
    2. in a new terminal, enter this docker as root
    `docker exec -u root <image id> /bin/bash`
    or
    `docker exec -u root  -it  <image name> /bin/bash`
    3. install the module
    RILD_pipeline:
  ``` cd /mnt/RILD/RILD_pipeline_run
      pip install -e .
  ```
    4. check that your system can see the installed module inside your running Docker jupyter view:
    you can test by importing, or 'pip list'
    
    5. Save this image of docker to directly use in your docker run next time
    ```docker commit <container id>  <new name>```
    
    new name: i.e. dev/rild_docker:v3
    
  !!! PAY ATTENTION !!! Monai package highly will not work if you are running this Docker on a M1 chip, beware when using format.segment()

### Docker dev 

How the image was created?

```pip install neurodocker
   
   docker run --rm kaczmarj/neurodocker:niftyreg generate docker --base-image debian:buster-slim --pkg-manager apt --dcm2niix version=master  method=source --niftyreg version=master --miniconda version=latest conda_install="python=3.9 jupyterlab nipype dcm2niix niftyreg numpy scipy nibabel itk SimpleITK pandas monai pytorch-ignite" --user rild_user > Dockerfile_conda
   
   docker build - < Dockerfile_conda
```

### Singularity dev 

If the analysis needs to run in a cluster for batch processing, Singularity can work better as a container in these environments mainly because
"Docker containers need root privileges for full functionality which is not suitable for a shared HPC environment, Singularity allows working with containers as a regular user." (https://docs.rc.fas.harvard.edu/kb/singularity-on-the-cluster/#:~:text=Singularity%20is%20available%20only%20on,in%20an%20interactive%20bash%20shell.) [source]

#### To create the Singularity container 

    1. `pip install spython`  #command line interface to work with Singularity
    2. ` spython recipe Dockerfile &> Singularity.def` create the Singularity definition file from Dockerfile, save in the *.def file

#### To use the Singularity container including custom build of RILD_pipeline module 

(for users, you have Singularity.def file in rild_docker/Singularity.def)
    
    1. `singularity build Singularity.sif Singularity.def` #convert .def file into the .sif file Singularity application can run. THIS IS IMPORTANT. Make sure that all text comments (#) are removed from the Singularity.def file or the .sif file may not build.
    2. `singularity shell Singularity.sif --bind /path/ source .bashrc` #launch the Singularity application with a bind that kick starts the bash scripting in command line 
    3. `conda env create --name RILD_env -f env.yaml` #create the RILD_env to build the packages in env.yaml
    4. `source .bashrc `  #source the bash again to recognise changes, make sure you're in your home directory when running this and this allows you to run conda activate RILD_env
    5. `conda activate RILD_env` #activate the RILD_env
    6. `cd <go to RILD_pipeline>`
    7. `pip install -e . ` #build the RILD_module 



## Data set-up

In order for the pipeline to run successfully, strict naming conventions and folder structure must be followed. This is the expected input structure to the pipeline:

Patient Folder/
├── DICOM/
    ├── planning_data/
    │   ├── planning_scan
    │   └── radiotherapy_data/
    │       ├── dose.dcm
    │       ├── plan.dcm
    │       └── PTV.dcm
    ├── baseline/
    │   ├── slice1.dcm
    │   ├── slice2.dcm
    │   └── ...
    └── follow_up_12/
        ├── slice1.dcm
        ├── slice2.dcm
        └── ...



Once you run the entire pipeline processing, the same patient directory tree will look something like this:

Patient Folder/
├── DICOM/
│   ├── planning_data/
│   │   ├── planning_scan
│   │   └── radiotherapy_data/
│   │       ├── dose.dcm
│   │       ├── plan.dcm
│   │       └── PTV.dcm
│   ├── baseline/
│   │   ├── slice1.dcm
│   │   ├── slice2.dcm
│   │   └── ...
│   └── follow_up_12/
│       ├── slice1.dcm
│       ├── slice2.dcm
│       └── ...
├── CT/
│   ├── orig/
│   │   ├── baseline.nii
│   │   └── follow_up_12.nii
│   └── aligned/
│       ├── baseline.nii
│       └── follow_up_12.nii
├── Radiotherapy/
│   ├── orig/
│   │   └── dose.nii
│   └── aligned/
│       ├── planning_dose.nii
│       ├── baseline_dose.nii
│       └── follow_up_12_dose.nii
├── Affine/
│   ├── planning_2_baseline.txt
│   └── follow_up_12_2_baseline.txt
├── Segmentations/
│   ├── orig/
│   │   ├── baseline_airways.nii
│   │   ├── baseline_bones.nii
│   │   ├── baseline_lungs.nii
│   │   ├── baseline_vesselness.nii
│   │   ├── follow_up_12_airways.nii
│   │   ├── follow_up_12_bones.nii
│   │   ├── follow_up_12_lungs.nii
│   │   └── follow_up_12_vesselness.nii
│   └── aligned/
│       ├── baseline_airways.nii
│       ├── baseline_bones.nii
│       ├── baseline_lungs.nii
│       ├── baseline_vesselness.nii
│       ├── follow_up_12_airways.nii
│       ├── follow_up_12_bones.nii
│       ├── follow_up_12_lungs.nii
│       └── follow_up_12_vesselness.nii
├── Stacks/
│   ├── baseline.nii
│   └── follow_up_12.nii
├── Registrations/
│   ├── planning/
│   │   ├── for_cpp.nii
│   │   ├── back_cpp.nii
│   │   ├── for_res.nii
│   │   ├── back_res.nii
│   │   ├── for_transformed.nii
│   │   └── back_transformed.nii
│   └── follow_up_12/
│       ├── for_cpp.nii
│       ├── back_cpp.nii
│       ├── for_res.nii
│       ├── back_res.nii
│       ├── for_transformed.nii
│       └── back_transformed.nii
└── Tissue_Classes/
    ├── baseline.nii
    ├── baseline.csv
    ├── follow_up_12.nii
    └── follow_up_12.csv


When you're using the docker make sure you're mounting the data path appropriately, and that the directory with the trained models is also in there.


