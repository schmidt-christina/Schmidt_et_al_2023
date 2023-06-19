#!/bin/bash
#PBS -P e14
#PBS -l ncpus=48
#PBS -l mem=188GB
#PBS -q normal
#PBS -l walltime=1:00:00
#PBS -l storage="gdata/hh5+gdata/ik11+gdata/v45+gdata/e14+gdata/cj50+scratch/v45+scratch/x77"
#PBS -l wd
#PBS -o transport_postprocessing.out
#PBS -j oe

module use /g/data/hh5/public/modules

python3 transport_across_1000m_isobath_postprocessing.py ${year} &>> transport_across_1000m_isobath_postprocessing_${year}.txt