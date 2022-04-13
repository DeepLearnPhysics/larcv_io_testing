import sys, os

# This file contains the bare startup scripts for systems

ThetaGPU = '''
#!/bin/bash -l

module load conda/2021-11-30
conda activate
module load hdf5

WORKDIR=/home/cadams/ThetaGPU/larcv_iotest/
cd $WORKDIR

'''
