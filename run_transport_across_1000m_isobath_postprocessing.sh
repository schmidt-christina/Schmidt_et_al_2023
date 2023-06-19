#!/bin/bash

for y in {1958..2018}; do
    qsub -v year=$y transport_across_1000m_isobath_postprocessing.sh
done