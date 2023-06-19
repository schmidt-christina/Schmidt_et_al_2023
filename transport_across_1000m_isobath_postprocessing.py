#!/usr/bin/env python
# coding: utf-8

'''Compute cross contour transport from daily data'''

import cosima_cookbook as cc
import numpy as np
import xarray as xr
from glob import glob
import os
import sys
from dask.distributed import Client
import logging
logging.captureWarnings(True)
logging.getLogger('py.warnings').setLevel(logging.ERROR)

if __name__ == '__main__':
    expt = '01deg_jra55v140_iaf_cycle3'
    path_output = '/g/data/e14/cs6673/iav_AABW/data_iav_AABW_final/'
    
    year = int(sys.argv[1])

    ds = xr.open_mfdataset(
        path_output + 'vol_trans_across_contour_' + expt + '_1d_' +
        str(year) + '-*.nc')
    ds.attrs = {
        'units': 'm^3/s',
        'long_name': 'Volume transport across 1000-m isobath'}
    ds.to_netcdf(
        path_output + 'vol_trans_across_contour_' + expt + '_1d_' +
        str(year) + '.nc')
    
    for f in glob(
        path_output + 'vol_trans_across_contour_' + expt + '_1d_' +
        str(year) + '-*.nc'):
        os.remove(f)
