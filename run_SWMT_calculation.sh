#!/bin/bash

for y in {1958..2018}; do
   qsub -v year=$y SWMT_calculation.sh
done