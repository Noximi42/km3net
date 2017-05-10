#!/bin/bash

echo "Load init"

export LD_LIBRARY_PATH=~/cuDNN/cuda/lib64/
LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/cntk/cntk/dependencies/lib/

module load cuda
module load openmpi/1.10.5-gcc

echo "loaded init"
