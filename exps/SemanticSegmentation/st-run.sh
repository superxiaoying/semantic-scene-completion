#!/usr/bin/env bash
#export OMP_NUM_THREADS=4

GPUNUM=8
JOBNAME=seg

PART=HA_senseAR

export TMPDIR=/mnt/lustre/chenxiaokang/
TOOLS="srun --partition=$PART --mpi=pmi2 --gres=gpu:$GPUNUM -n1 --ntasks-per-node=$GPUNUM"
$TOOLS --job-name=$JOBNAME python -m torch.distributed.launch --nproc_per_node=$GPUNUM train.py

TOOLS="srun --partition=$PART --mpi=pmi2 --gres=gpu:4 -n1 --ntasks-per-node=$GPUNUM"
$TOOLS --job-name=$JOBNAME python eval.py -e 360-400 -d 0-3
