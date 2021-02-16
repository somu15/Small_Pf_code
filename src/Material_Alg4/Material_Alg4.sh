#!/bin/bash
#PBS -M Som.Dhulipala@inl.gov
#PBS -m abe
#PBS -N Material_GP_recommender4
#PBS -P moose
#PBS -l select=1:ncpus=1:ngpus=1
#PBS -q gpu
#PBS -l walltime=144:00:00

JOB_NUM=${PBS_JOBID%%\.*}

cd $PBS_O_WORKDIR

\rm -f out
date > out
module load tensorflow/2.4_gpu
MV2_ENABLE_AFFINITY=0 python Alg_nD_Material_LF1.py >> out
date >> out
