# Schmidt_et_al_2023

This repository contains all scripts used to analysis the model output and produce the figures in

Schmidt, C., Morrison, A. K., & England, M. H. (2023). Wind– and sea-ice–driven interannual variability of Antarctic Bottom Water formation. Journal of Geophysical Research: Oceans, 128, e2023JC019774. https://doi.org/10.1029/2023JC019774

## Content
- Python script to create all figures and calculate all values refered to in the manuscript:
  - Plots.ipynb
- calculation of all time series analysed:
  - timeseries.ipynb
- calculation of the surface water mass transformation:
  -  run_SWMT_calculation.sh which calls SWMT_calculation.sh and SWMT_calculation.py
- calculation and postprocessing of the transport across the 1000 m isobath:
  -  run_transport_across_1000m_isobath_calculation.sh which calls transport_across_1000m_isobath_calculation.sh and transport_across_1000m_isobath_calculation.py
  - transport_across_1000m_isobath_postprocessing.sh which calls transport_across_1000m_isobath_postprocessing.sh and transport_across_1000m_isobath_postprocessing.py
- functions used in all scripts:
  - iav_AABW_finctions.py


