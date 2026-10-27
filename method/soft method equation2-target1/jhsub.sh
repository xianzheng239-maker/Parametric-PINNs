#!/bin/bash
#JSUB -q fat
#JSUB -n 32
#JSUB -e error.%J
#JSUB -o ouput.%J
#JSUB -J my_job
./pyjob.sh
