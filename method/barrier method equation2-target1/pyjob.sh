#!/bin/bash
source activate
conda deactivate
conda activate py39tf2tfp-gpu
python -u main.py > result.log 2>&1 & 
echo $!>pidfile.txt