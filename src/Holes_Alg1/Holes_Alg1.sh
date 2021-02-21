#!/bin/bash
#PBS -M Som.Dhulipala@inl.gov
#PBS -m abe
#PBS -N Holes_model_GP1
#PBS -P moose
#PBS -l select=1:ncpus=1:mpiprocs=4:ngpus=1
#PBS -q gpu
#PBS -l walltime=48:00:00

JOB_NUM=${PBS_JOBID%%\.*}

cd $PBS_O_WORKDIR

\rm -f out
date > out
module load tensorflow/2.4_gpu
MV2_ENABLE_AFFINITY=0 python Alg_nD_Holes_LF1.py >> out
date >> out
