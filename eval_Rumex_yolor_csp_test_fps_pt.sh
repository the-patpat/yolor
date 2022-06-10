#!/bin/sh 
### General options 
### -- specify queue -- 
#BSUB -q gpuv100 
### -- set the job Name -- 
#BSUB -J train_Rumex_yolor_csp_fps
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
#BSUB -oo Output_eval_rumex_yolor_csp_test_fps_wandb.out 
#BSUB -eo Error_eval_rumex_yolor_csp_test_fps_wandb.err 

# here follow the commands you want to execute 

source activate yolor
conda env list
nvidia-smi
python test.py --plot_imgs --weights runs/train/yolor_csp_pretrained/weights/best.pt --batch-size 1 --device 0 --task test --cfg cfg/yolor_csp_rumex.cfg --data /zhome/d4/b/153599/scratch/RumexWeeds-YOLOv5/dataset.yaml --conf-thres 0.01 --names ../RumexWeeds-YOLOv5/dataset.names --iou-thres 0.65 --img-size 640 --project inference/csp/test_fps_withwandb
