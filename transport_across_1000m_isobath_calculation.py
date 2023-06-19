#!/usr/bin/env python
# coding: utf-8

'''Compute cross contour transport from daily data'''

import cosima_cookbook as cc
import numpy as np
import xarray as xr
from gsw import SA_from_SP, p_from_z, sigma0
import sys
from dask.distributed import Client
import logging
logging.captureWarnings(True)
logging.getLogger('py.warnings').setLevel(logging.ERROR)

if __name__ == '__main__':
    #client = Client()
    #client

    session = cc.database.create_session()

    year = int(sys.argv[1])
    year = str(year)
    time_step = int(sys.argv[2])

    expt = '01deg_jra55v140_iaf_cycle3'

    start_time = year + '-01-01'
    end_time = year + '-12-31'

    # reference density value:
    rho_0 = 1025.0
    # Note: change this range, so it matches the size of your contour arrays:
    lat_range = slice(-90, -59)

    '''Open grid cell width data for domain'''

    # some grid data is required, a little complicated because these variables
    # don't behave well with some
    dyt = cc.querying.getvar(expt, 'dyt', session, n=1)
    dxu = cc.querying.getvar(expt, 'dxu', session, n=1)

    # select latitude range:
    dxu = dxu.sel(yu_ocean=lat_range)
    dyt = dyt.sel(yt_ocean=lat_range)

    '''Open contour data'''
    isobath_depth = 1000
    outfile = (
        '/g/data/v45/akm157/model_data/access-om2/Antarctic_slope_contour_' +
        str(isobath_depth) + 'm.npz')
    data = np.load(outfile)
    mask_y_transport = data['mask_y_transport']
    mask_x_transport = data['mask_x_transport']
    mask_y_transport_numbered = data['mask_y_transport_numbered']
    mask_x_transport_numbered = data['mask_x_transport_numbered']

    yt_ocean = cc.querying.getvar(expt, 'yt_ocean', session, n=1)
    yt_ocean = yt_ocean.sel(yt_ocean=lat_range)
    yu_ocean = cc.querying.getvar(expt, 'yu_ocean', session, n=1)
    yu_ocean = yu_ocean.sel(yu_ocean=lat_range)
    xt_ocean = cc.querying.getvar(expt, 'xt_ocean', session, n=1)
    xu_ocean = cc.querying.getvar(expt, 'xu_ocean', session, n=1)

    # Convert contour masks to data arrays, so we can multiply them later.
    # We need to ensure the lat lon coordinates correspond to the actual data
    # location:
    #       The y masks are used for vhrho, so like vhrho this should have
    #       dimensions (yu_ocean, xt_ocean).
    #       The x masks are used for uhrho, so like uhrho this should have
    #       dimensions (yt_ocean, xu_ocean).
    #       However the actual name will always be simply y_ocean/x_ocean
    #       irrespective of the variable to make concatenation of transports
    #       in both direction and sorting possible.

    mask_x_transport = xr.DataArray(
        mask_x_transport,
        coords=[('y_ocean', yt_ocean.data), ('x_ocean', xu_ocean.data)])
    mask_y_transport = xr.DataArray(
        mask_y_transport,
        coords=[('y_ocean', yu_ocean.data), ('x_ocean', xt_ocean.data)])
    mask_x_transport_numbered = xr.DataArray(
        mask_x_transport_numbered,
        coords=[('y_ocean', yt_ocean.data), ('x_ocean', xu_ocean.data)])
    mask_y_transport_numbered = xr.DataArray(
        mask_y_transport_numbered,
        coords=[('y_ocean', yu_ocean.data), ('x_ocean', xt_ocean.data)])

    # number of points along contour:
    num_points = int(np.maximum(np.max(mask_y_transport_numbered),
                     np.max(mask_x_transport_numbered)))

    '''Stack contour data into 1D and extract lat/lon on contour'''
    # Create the contour order data-array. Note that in this procedure the
    # x-grid counts have x-grid dimensions and the y-grid counts have y-grid
    # dimensions, but these are implicit, the dimension *names* are kept
    # general across the counts, the generic y_ocean, x_ocean, so that
    # concatening works but we dont double up with numerous counts for one
    # lat/lon point.

    # stack contour data into 1d:
    mask_x_numbered_1d = mask_x_transport_numbered.stack(
        contour_index=['y_ocean', 'x_ocean'])
    mask_x_numbered_1d = mask_x_numbered_1d.where(
        mask_x_numbered_1d > 0, drop=True)
    mask_y_numbered_1d = mask_y_transport_numbered.stack(
        contour_index=['y_ocean', 'x_ocean'])
    mask_y_numbered_1d = mask_y_numbered_1d.where(
        mask_y_numbered_1d > 0, drop=True)
    contour_ordering = xr.concat((mask_x_numbered_1d, mask_y_numbered_1d),
                                 dim='contour_index')
    contour_ordering = contour_ordering.sortby(contour_ordering)

    # get lat and lon along contour, useful for plotting later:
    lat_along_contour = contour_ordering.y_ocean
    lon_along_contour = contour_ordering.x_ocean
    contour_index_array = np.arange(1, len(contour_ordering)+1)
    # don't need the multi-index anymore, replace with contour count and save
    lat_along_contour.coords['contour_index'] = contour_index_array
    lon_along_contour.coords['contour_index'] = contour_index_array

    '''Open uhrho, vhrho from daily data'''

    # Note vhrho_nt is v*dz*1035 and positioned on north centre edge of t-cell.
    vhrho = cc.querying.getvar(expt,  'vhrho_nt', session,
                               start_time=start_time, end_time=end_time)
    uhrho = cc.querying.getvar(expt, 'uhrho_et', session,
                               start_time=start_time, end_time=end_time)

    # select latitude range and this month:
    vhrho = vhrho.sel(yt_ocean=lat_range).sel(time=slice(start_time, end_time))
    uhrho = uhrho.sel(yt_ocean=lat_range).sel(time=slice(start_time, end_time))

    # Note that vhrho is defined as the transport across the northern edge of
    #       a tracer cell so its coordinates should be (yu_ocean, xt_ocean).
    #  uhrho is defined as the transport across the eastern edge of a tracer
    #       cell so its coordinates should be (yt_ocean, xu_ocean).
    #  However we will keep the actual name as simply y_ocean/x_ocean
    #       irrespective of the variable to make concatenation and sorting
    #       possible.
    yt_ocean = dyt.yt_ocean.values
    yu_ocean = dxu.yu_ocean.values
    xu_ocean = dxu.xu_ocean.values
    xt_ocean = dyt.xt_ocean.values
    vhrho.coords['yt_ocean'] = yu_ocean
    uhrho.coords['xt_ocean'] = xu_ocean
    vhrho = vhrho.rename({'yt_ocean': 'y_ocean', 'xt_ocean': 'x_ocean'})
    uhrho = uhrho.rename({'yt_ocean': 'y_ocean', 'xt_ocean': 'x_ocean'})

    '''Convert to transports'''
    # First we also need to change coords on dxu, dyt, so we can multiply the
    # transports:
    dyt = dyt.reset_coords().dyt  # remove geolon_t/geolat_t coordinates
    dxu = dxu.reset_coords().dxu  # remove geolon_t/geolat_t coordinates
    dxu.coords['xu_ocean'] = xt_ocean
    dxu = dxu.rename({'yu_ocean': 'y_ocean', 'xu_ocean': 'x_ocean'})
    dyt.coords['xt_ocean'] = xu_ocean
    dyt = dyt.rename({'yt_ocean': 'y_ocean', 'xt_ocean': 'x_ocean'})

    # convert to transports and multiply by contour masks:
    vhrho = vhrho*dxu*mask_y_transport/rho_0
    uhrho = uhrho*dyt*mask_x_transport/rho_0

    '''Extract transport values along contour:'''
    # load one timestep of transport data:
    # loading here speeds it up by about 5x...
    uhrho_i = uhrho[time_step, ...]
    uhrho_i = uhrho_i.load()
    vhrho_i = vhrho[time_step, ...]
    vhrho_i = vhrho_i.load()

    # stack transports into 1d and drop any points not on contour:
    x_transport_1d_i = uhrho_i.stack(contour_index=['y_ocean', 'x_ocean'])
    x_transport_1d_i = x_transport_1d_i.where(
        mask_x_numbered_1d > 0, drop=True)
    y_transport_1d_i = vhrho_i.stack(contour_index=['y_ocean', 'x_ocean'])
    y_transport_1d_i = y_transport_1d_i.where(
        mask_y_numbered_1d > 0, drop=True)

    # combine all points on contour:
    vol_trans_across_contour = xr.concat((x_transport_1d_i, y_transport_1d_i),
                                         dim='contour_index')
    vol_trans_across_contour = vol_trans_across_contour.sortby(
        contour_ordering)
    vol_trans_across_contour.coords['contour_index'] = contour_index_array
    vol_trans_across_contour = vol_trans_across_contour.load()

    '''Open salt data, interpolate onto transport grids, and extract along
       contour'''
    salt = cc.querying.getvar(expt, 'salt', session, start_time=start_time,
                              end_time=end_time, ncfile='%daily%')
    salt = salt.sel(yt_ocean=lat_range).sel(time=slice(start_time, end_time))

    # This is faster if we load first here:
    salt_i = salt[time_step, ...]
    salt_i = salt_i.load()
    salt_i = salt_i.rename({'yt_ocean': 'y_ocean', 'xt_ocean': 'x_ocean'})

    # Note that this interpolation does not work as generically as e.g.
    # salt.interp(), but it is much faster and doesn't require removing
    # chunking (which also slow things down).
    # Be careful that your latitude range extends at least one point either
    # direction beyond your contour.
    # If your domain is not the full longitude range, you will need to adapt
    # this, so you have the correct interpolation only the edges of your domain
    # (it assumes it is reentrant).
    # Need to overwrite coords, so these two variables can be added together:
    salt_w = salt_i.copy()
    salt_w.coords['x_ocean'] = xu_ocean
    salt_e = salt_i.roll(x_ocean=-1)
    salt_e.coords['x_ocean'] = xu_ocean
    # salt_xgrid will be on the uhrho grid:
    salt_xgrid = (salt_e + salt_w)/2

    salt_s = salt_i.copy()
    salt_s.coords['y_ocean'] = yu_ocean
    salt_n = salt_i.roll(y_ocean=-1)
    salt_n.coords['y_ocean'] = yu_ocean
    # salt_ygrid will be on the vhrho grid:
    salt_ygrid = (salt_s + salt_n)/2

    # stack transports into 1d and drop any points not on contour:
    salt_xgrid = salt_xgrid.where(mask_x_transport_numbered > 0)
    salt_ygrid = salt_ygrid.where(mask_y_transport_numbered > 0)
    x_salt_1d = salt_xgrid.stack(contour_index=['y_ocean', 'x_ocean'])
    y_salt_1d = salt_ygrid.stack(contour_index=['y_ocean', 'x_ocean'])
    x_salt_1d = x_salt_1d.where(mask_x_numbered_1d > 0, drop=True)
    y_salt_1d = y_salt_1d.where(mask_y_numbered_1d > 0, drop=True)

    # combine all points on contour:
    salt_along_contour = xr.concat((x_salt_1d, y_salt_1d), dim='contour_index')
    salt_along_contour = salt_along_contour.sortby(contour_ordering)
    salt_along_contour.coords['contour_index'] = contour_index_array
    salt_along_contour = salt_along_contour.load()

    del (salt_i, salt_w, salt_e, salt_s, salt_n, salt_xgrid, salt_ygrid,
         x_salt_1d, y_salt_1d)

    '''Open temp data, interpolate onto transport grids, and extract along
       contour'''
    temp = cc.querying.getvar(expt, 'temp', session, start_time=start_time,
                              end_time=end_time, ncfile='%daily%') - 273.15
    temp = temp.sel(yt_ocean=lat_range).sel(time=slice(start_time, end_time))

    # This is faster if we load first here:
    temp_i = temp[time_step, ...]
    temp_i = temp_i.load()
    temp_i = temp_i.rename({'yt_ocean': 'y_ocean', 'xt_ocean': 'x_ocean'})

    # Note that this interpolation does not work as generically as e.g.
    # temp.interp(), but it is much faster and doesn't require removing
    # chunking (which also slow things down).
    # Be careful that your latitude range extends at least one point either
    # direction beyond your contour.
    # If your domain is not the full longitude range, you will need to adapt
    # this, so you have the correct interpolation only the edges of your domain
    # (it assumes it is reentrant).
    # Need to overwrite coords, so these two variables can be added together:
    temp_w = temp_i.copy()
    temp_w.coords['x_ocean'] = xu_ocean
    temp_e = temp_i.roll(x_ocean=-1)
    temp_e.coords['x_ocean'] = xu_ocean
    # temp_xgrid will be on the uhrho grid:
    temp_xgrid = (temp_e + temp_w)/2

    temp_s = temp_i.copy()
    temp_s.coords['y_ocean'] = yu_ocean
    temp_n = temp_i.roll(y_ocean=-1)
    temp_n.coords['y_ocean'] = yu_ocean
    # temp_ygrid will be on the vhrho grid:
    temp_ygrid = (temp_s + temp_n)/2

    # stack transports into 1d and drop any points not on contour:
    temp_xgrid = temp_xgrid.where(mask_x_transport_numbered > 0)
    temp_ygrid = temp_ygrid.where(mask_y_transport_numbered > 0)
    x_temp_1d = temp_xgrid.stack(contour_index=['y_ocean', 'x_ocean'])
    y_temp_1d = temp_ygrid.stack(contour_index=['y_ocean', 'x_ocean'])
    x_temp_1d = x_temp_1d.where(mask_x_numbered_1d > 0, drop=True)
    y_temp_1d = y_temp_1d.where(mask_y_numbered_1d > 0, drop=True)

    # combine all points on contour:
    temp_along_contour = xr.concat((x_temp_1d, y_temp_1d), dim='contour_index')
    temp_along_contour = temp_along_contour.sortby(contour_ordering)
    temp_along_contour.coords['contour_index'] = contour_index_array
    temp_along_contour = temp_along_contour.load()

    del (temp_i, temp_w, temp_e, temp_s, temp_n, temp_xgrid, temp_ygrid,
         x_temp_1d, y_temp_1d)

    '''Calculate density on contour'''
    st_ocean = vhrho.st_ocean.values
    depth = -st_ocean
    depth = xr.DataArray(depth, coords=[st_ocean], dims=['st_ocean'])
    depth_along_contour = (salt_along_contour*0+1)*depth

    pressure_along_contour = xr.DataArray(
        p_from_z(depth_along_contour, lat_along_contour),
        coords=[st_ocean, contour_index_array],
        dims=['st_ocean', 'contour_index'],
        name='pressure', attrs={'units': 'dbar'})

    # absolute salinity:
    abs_salt_along_contour = xr.DataArray(
        SA_from_SP(salt_along_contour, pressure_along_contour,
                   lon_along_contour, lat_along_contour),
        coords=[st_ocean, contour_index_array],
        dims=['st_ocean', 'contour_index'],
        name='Absolute salinity',
        attrs={'units': 'Absolute Salinity (g/kg)'})

    sigma0_along_contour = xr.DataArray(
        sigma0(abs_salt_along_contour, temp_along_contour),
        coords=[st_ocean, contour_index_array],
        dims=['st_ocean', 'contour_index'],
        name='potential density ref 1000dbar',
        attrs={'units': 'kg/m^3 (-1000 kg/m^3)'})

    # """Save salt flux"""
    # salt_trans_across_contour = abs_salt_along_contour * vol_trans_across_contour
    # path_output = '/g/data/e14/cs6673/iav_AABW/data_SWMT/'
    # salt_trans_across_contour = salt_trans_across_contour.expand_dims(
    #     time=[salt.time[time_step].values])
    # ds_salt_trans_across_contour = xr.Dataset(
    #     {'salt_trans_across_contour': salt_trans_across_contour})
    # ds_salt_trans_across_contour.to_netcdf(
    #     path_output + 'salt_trans_across_contour_' + expt + '_1d_' +
    #     np.datetime_as_string(salt.time[time_step].values, unit='D') + '.nc')

    '''Bin into isopycnals'''
    # define isopycnal bins
    isopycnal_bins_sigma0 = np.arange(27.6, 28.11, 0.01)

    # intialise empty transport along contour in density bins array
    vol_trans_across_contour_binned = xr.DataArray(
        np.zeros((len(isopycnal_bins_sigma0), len(contour_ordering))),
        coords=[isopycnal_bins_sigma0, contour_index_array],
        dims=['isopycnal_bins', 'contour_index'],
        name='vol_trans_across_contour_binned')

    # loop through density bins:
    for i in range(len(isopycnal_bins_sigma0)-1):
        bin_mask = sigma0_along_contour.where(
            sigma0_along_contour <= isopycnal_bins_sigma0[i+1]).where(
                sigma0_along_contour > isopycnal_bins_sigma0[i])*0+1
        bin_fractions = (isopycnal_bins_sigma0[i+1]-sigma0_along_contour *
                         bin_mask)/(isopycnal_bins_sigma0[i+1] -
                                    isopycnal_bins_sigma0[i])
        # transport
        transport_across_contour_in_sigmalower_bin = (
            vol_trans_across_contour * bin_mask * bin_fractions).sum(
            dim='st_ocean')
        vol_trans_across_contour_binned[i, :] += (
            transport_across_contour_in_sigmalower_bin.fillna(0))
        del transport_across_contour_in_sigmalower_bin
        transport_across_contour_in_sigmaupper_bin = (
            vol_trans_across_contour * bin_mask * (1-bin_fractions)).sum(
            dim='st_ocean')
        vol_trans_across_contour_binned[i+1, :] += (
            transport_across_contour_in_sigmaupper_bin.fillna(0))
        del bin_mask, bin_fractions, transport_across_contour_in_sigmaupper_bin

    '''Save'''
    path_output = '/g/data/e14/cs6673/iav_AABW/data_iav_AABW_final/'
    vol_trans_across_contour_binned = vol_trans_across_contour_binned.expand_dims(
        time=[salt.time[time_step].values])
    ds_vol_trans_across_contour = xr.Dataset(
        {'vol_trans_across_contour': vol_trans_across_contour_binned})
    ds_vol_trans_across_contour.attrs = {
        'units': 'm^3/s', 'long_name': 'Volume transport across 1000-m isobath'}
    enc = {'vol_trans_across_contour':
           {'zlib': True, 'complevel': 5, 'shuffle': True}}
    ds_vol_trans_across_contour.to_netcdf(
        path_output + 'vol_trans_across_contour_' + expt + '_1d_' +
        np.datetime_as_string(salt.time[time_step].values, unit='D') +
        '.nc', encoding=enc)
