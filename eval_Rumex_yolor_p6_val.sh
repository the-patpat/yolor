#!/bin/sh 
### General options 
### -- specify queue -- 
#BSUB -q gpuv100 
### -- set the job Name -- 
#BSUB -J train_Rumex_obs_yolor_p6
### -- Select the resources: 1 gpu in exclusive process mode --
#BSUB -gpu "num=1:mode=exclusive_process"
### -- ask for number of cores (default: 1) -- 
#BSUB -n 1 
### -- specify that we need 2GB of memory per core/slot -- 
#BSUB -R "rusage[mem=8GB]"
### -- specify that we want the job to get killed if it exceeds 3 GB per core/slot -- 
#BSUB -M 32GB
### -- set walltime limit: hh:mm -- 
#BSUB -W 24:00 
### -- set working directory --
#BSUB -cwd /work1/s202616/yolor/
### -- set the email address -- 
# please uncomment the following line and put in your e-mail address,
# if you want to receive e-mail notifications on a non-default address
#BSUB -u s202616@dtu.dk 
### -- send notification at start -- 
#BSUB -B 
### -- send notification at completion -- 
#BSUB -N 
### -- Specify the output and error file. %J is the job-id -- 
### -- -o and -e mean append, -oo and -eo mean overwrite -- 
#BSUB -o Output_eval_rumex_yolor_p6_val.out 
#BSUB -e Error_eval_rumex_yolor_p6_val.err 

# here follow the commands you want to execute 

source activate yolor
conda env list
nvidia-smi
python detect.py --weights runs/train/yolor_p6/weights/best.pt --source ../RumexWeeds-YOLOv5/images/val/ --conf-thres 0.25 --iou-thres 0.45  --save-txt --names ../RumexWeeds-YOLOv5/dataset.names --classes 0 --img-size 1920 --output inference/Output_eval_rumex_yolor_p6_val_conf025_nms045
