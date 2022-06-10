#!/bin/sh 
### General options 
### -- specify queue -- 
#BSUB -q gpuv100 
### -- set the job Name -- 
#BSUB -J train_Rumex_single_csp
### -- Select the resources: 1 gpu in exclusive process mode --
#BSUB -gpu "num=1:mode=exclusive_process"
### -- ask for number of cores (default: 1) -- 
#BSUB -n 2 
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
#BSUB -oo Output_train_rumex_yolor_csp_single_pretrained.out 
#BSUB -eo Error_train_rumex_yolor_csp_single_pretrained.err 

# here follow the commands you want to execute 

source activate yolor
conda env list
nvidia-smi
# python -m torch.distributed.launch --nproc_per_node 2 --master_port 9527 train.py --batch-size 16 --img 640 640 --data ../RumexWeeds-YOLOv5/dataset.yaml --cfg cfg/yolor_p6.cfg --weights yolor_p6.pt --device 0,1 --sync-bn --name yolor_p6 --hyp hyp.scratch.640.yaml --epochs 100
python train.py --batch-size 16 --img 640 --data ../RumexWeeds-YOLOv5/dataset.yaml --cfg cfg/yolor_csp_rumex.cfg --weights yolor_csp.pt --device 0 --name yolor_csp_pretrained --hyp hyp.scratch.640.yaml --epochs 100