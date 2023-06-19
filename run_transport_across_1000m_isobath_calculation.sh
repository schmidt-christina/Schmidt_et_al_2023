#!/bin/bash

for y in {1958..2018}; do
    if (( $y % 4 == 0 ))
    then
        for t in {0..365}; do
            qsub -v year=$y,timestep=$t transport_across_1000m_isobath_calculation.sh
        done
    else
        for t in {0..364}; do
            qsub -v year=$y,timestep=$t transport_across_1000m_isobath_calculation.sh
        done
    fi
done