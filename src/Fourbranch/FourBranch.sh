#!/bin/bash
#PBS -M Som.Dhulipala@inl.gov
#PBS -m abe
#PBS -N Borehole_GP_recommender_GP1
#PBS -P moose
#PBS -l select=1:ncpus=1:ngpus=3
#PBS -q gpu
#PBS -l walltime=15:00:00

JOB_NUM=${PBS_JOBID%%\.*}

cd $PBS_O_WORKDIR

\rm -f out
date > out
module load tensorflow/2.4_gpu
MV2_ENABLE_AFFINITY=0 python Alng_new.py >> out
date >> out
