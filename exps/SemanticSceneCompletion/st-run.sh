#!/usr/bin/env bash
#export OMP_NUM_THREADS=4

GPUNUM=2
NODE1=SH-IDC1-10-5-36-176
NODE2=SH-IDC1-10-5-30-218
NODE3=SH-IDC1-10-5-34-45
JOBNAME=nd.fu.clean

PART1=Pose
PART2=HA_senseAR
PART3=ha_vug

NODE=$NODE1
PART=$PART2

#export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export TMPDIR=/mnt/lustre/chenxiaokang/
TOOLS="srun --partition=$PART --mpi=pmi2 -x SH-IDC1-10-5-36-175 --gres=gpu:$GPUNUM -n1 --ntasks-per-node=$GPUNUM"


$TOOLS --job-name=$JOBNAME python -m torch.distributed.launch --nproc_per_node=$GPUNUM train.py

TOOLS="srun --partition=$PART --mpi=pmi2 --gres=gpu:$GPUNUM -n1 --ntasks-per-node=$GPUNUM"
$TOOLS --job-name=$JOBNAME python eval.py -e 200-250 -d 0-1