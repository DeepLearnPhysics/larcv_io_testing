#!/bin/sh
#COBALT -t 60
#COBALT -n 1
#COBALT -q full-node
#COBALT -A datascience

module load conda/2021-11-30
conda activate
module load hdf5

WORKDIR=/home/cadams/Theta/larcv3_scaling/larcv_io_testing/
cd $WORKDIR

# Set up software deps:
source /home/cadams/Theta/larcv3_scaling/setup.sh


# Loop over minibatch sizes in powers of two
# This isn't real scaling just increasing work size

for dataset in "dune2d" "dune3d" "cosmic_tagger"
do
    echo ${dataset}
done


for power in {0..8}
do
    for run in {0..10}
    do
        let minibatch=2**${power}
        echo ${minibatch}
        python exec.py distributed=false \
        id=single_proc_${run} \
        dataset.output_shape=sparse \
        minibatch_size=${minibatch}
    done
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
