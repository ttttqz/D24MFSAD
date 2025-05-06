#!/bin/bash
#JSUB -q gpu
#JSUB -gpgpu 1
#JSUB -m gpu03
#JSUB -e error.%J
#JSUB -o output.%J
#JSUB -n 16

cd /home/FSRD4AD/
source /home/anaconda3/etc/profile.d/conda.sh
conda activate pytorch
python main.py
