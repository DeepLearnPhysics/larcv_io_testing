#!/bin/sh
#COBALT -t 60
#COBALT -n 1
#COBALT -q debug-cache-quad
#COBALT -A datascience

WORKDIR=/home/cadams/Theta/larcv3_scaling/larcv_io_testing/
cd $WORKDIR

# Set up software deps:
source /home/cadams/Theta/larcv3_scaling/setup.sh

# Loop over minibatch sizes in powers of two:
for power in {0..8}
do
	let minibatch=2**${power}
    aprun -n 1 -N 1 -cc depth -j 1 \
    python exec.py distributed=false id=single_proc \
    dataset.output_shape=sparse \
    minibatch_size=${minibatch}
done

# Loop over minibatch sizes in powers of two:
for power in {0..8}
do
	let minibatch=2**${power}
    aprun -n 1 -N 1 -cc depth -j 1 \
    python exec.py distributed=false id=single_proc \
    dataset=dune3d \
    dataset.output_shape=sparse \
    minibatch_size=${minibatch}
done

# Loop over minibatch sizes in powers of two:
for power in {0..8}
do
	let minibatch=2**${power}
    aprun -n 1 -N 1 -cc depth -j 1 \
    python exec.py distributed=false id=single_proc \
    dataset.output_shape=dense \
    minibatch_size=${minibatch}
done

# Loop over minibatch sizes in powers of two:
# for power in {0..8}
# do
# 	let minibatch=2**${power}
#     aprun -n 1 -N 1 -cc depth -j 1 \
#     python exec.py distributed=false id=single_proc \
#     dataset=dune3d \
#     dataset.output_shape=dense \
#     minibatch_size=${minibatch}
# done
