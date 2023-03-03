# AHBF
This repository provides code of our paper "Adaptive Hierarchy-Branch Fusion for Online Knowledge Distillation". We provide training code on Cifar10\CIFAR100\Imagenet.
The dataset will be downloaded automatically to the configured data root.

At first, 
pip install wandb

If you do not use wandb for visualization,after installation use oder:
wandb offline 
to deactivate wandb.

## train baseline

python train.py --model resnet32 

## train AHBF

python train_ahbf.py --model resnet32 --num_branches 4 --aux 4 --lambda1 2 lambda2 4

## test AHBF

After training,the model will be saved in './dataset/num_epochs/model/'

python eval.py --model resnet32 --num_branches 4 --aux 4 --root model_root


