#!/usr/bin/env python
# coding: utf-8

"""Calculation of Surface Water Mass Transformation"""
# In this notebook, we compute surface water mass transformation rates (both
# in the net and partitioned into contributions from heat and salt fluxes) for
# the Southern Ocean, south of 60°S.
#
# Here we are looking specifically at ACCESS-OM2-01 output (0.1° resolution,
# 3rd cycle), as this configuration has demonstrated considerable success
# when simulating high latitude dense water formation processes.

import xarray as xr
import numpy as np
import sys
import cosima_cookbook as cc
from gsw import SA_from_SP, p_from_z, alpha, beta
from iav_AABW_functions import shelf_mask_isobath, mask_from_polygon

if __name__ == '__main__':
    """Defining surface watermass transformation"""
    # The surface water-mass transformation framework described here follows
    # Newsom et al. (2016) and Abernathey et al. (2016). Surface water-mass
    # transformation may be defined as the volume flux into a given density
    # class sigma from lighter density classes due to surface buoyancy forcing.

    """Parameters"""
    expt = '01deg_jra55v140_iaf_cycle3'
    frequency = '1 monthly'
    year_all = range(1958, 2019)
    path_output = '/g/data/e14/cs6673/iav_AABW/data_iav_AABW_final/'
    
    year = int(sys.argv[1])

    DSW_region = {
        'name': ['Weddell', 'Prydz', 'Adelie', 'Ross'],
        'name_long': ['Weddell Sea', 'Prydz Bay', 'Adélie Coast', 'Ross Sea'],
        'lon': [[-60, -35, -48, -62, -60],
                [48, 73, 74, 48, 48],
                [128-360, 152-360, 152-360, 128-360, 128-360],
                [185-360, 160-360, 164-360, 172-360, 185-360]],
        'lat': [[-71, -75, -78, -75, -71],
                [-65, -66.5, -69, -68, -65],
                [-64.5, -66, -69, -67.5, -64.5],
                [-78, -78, -73, -71.5, -78]],
        'sig0_bins': [27.92, 27.81, 27.81, 28]}

    """Load data"""
    session = cc.database.create_session()
    ht = cc.querying.getvar(expt, 'ht', session, n=1)
    ht = ht.sel(yt_ocean=slice(-90, -59))
    land_mask = (ht*0).fillna(1)
    yt_ocean = ht.yt_ocean
    xt_ocean = ht.xt_ocean
    area_t = cc.querying.getvar(expt, 'area_t', session, n=1)

    """Surface water mass transformation (monthly mean) on Antarctic shelf"""
    def calculate_SWMT_monthly_mean(expt, session, start_time, end_time,
                                    frequency, path_output, filename,
                                    lat_north=-59):

        '''
        original function from Surface_Water_Mass_Transformation.ipynb,
        modified by Christina Schmidt

        Computes southern ocean surface water mass transformation rates
        (partitioned into transformation from heat and freshwater) referenced
        to 1000 db from monthly ACCESS-OM2 output.
        Suitable for analysis of high-resolution (0.1 degree) output
        (the scattered .load()'s allowed this)

        expt - text string indicating the name of the experiment
        session - a database session created by cc.database.create_session()
        start_time - text string designating the start date ('YYYY-MM-DD')
        end_time - text string indicating the end date ('YYYY-MM-DD')
        path_output - text string indicating directory where output databases
            are to be saved
        filename - text string of the name of the saved file
        lat_north - function computes processes between lat = -90 and
            lat = lat_north

        NOTE: assumes surface_temp and surface_salt variables are in
        conservative temperature (K) and practical salinity (PSU)

        required modules:
        xarray as xr
        numpy as np
        cosima_cookbook as cc
        from gsw import alpha, SA_from_SP, p_from_z, CT_from_pt, beta
        '''

        # load variables
        SST = cc.querying.getvar(
            expt, 'temp', session, frequency=frequency,
            chunks={'time': '200MB'}) - 273.15
        SST = SST.isel(st_ocean=0)
        # conservative temperature in K
        SSS_PSU = cc.querying.getvar(
            expt, 'surface_salt', session, frequency=frequency,
            chunks={'time': '200MB'})
        # practical salinity (not absolute)
        pot_rho_0 = cc.querying.getvar(
            expt, 'pot_rho_0', session, frequency=frequency,
            chunks={'time': '200MB'})
        # potential density referenced to the surface in kg/m^3
        pme_river = cc.querying.getvar(
            expt, 'pme_river', session, frequency=frequency,
            chunks={'time': '200MB'})
        # mass flux of precip - evap + river (water flux)
        sfc_salt_flux_ice = cc.querying.getvar(
            expt, 'sfc_salt_flux_ice', session, frequency=frequency,
            chunks={'time': '200MB'})
        # Salt flux from ice in kg/m^2/s (salt flux)
        sfc_salt_flux_restore = cc.querying.getvar(
            expt, 'sfc_salt_flux_restore',  session,
            frequency=frequency, chunks={'time': '200MB'})
        # Salt flux and restoring in kg/m^2/s (salt flux)

        net_sfc_heating = cc.querying.getvar(
            expt, 'net_sfc_heating',  session, frequency=frequency,
            chunks={'time': '200MB'})
        # net_surface_heating  in W/m2
        frazil_3d_int_z = cc.querying.getvar(
            expt, 'frazil_3d_int_z', session, frequency=frequency,
            chunks={'time': '200MB'})
        # W/m2

        geolon_t = cc.querying.getvar(expt, 'geolon_t', session, n=1,
                                      chunks={'yt_ocean': '200MB'})
        geolat_t = cc.querying.getvar(expt, 'geolat_t', session, n=1,
                                      chunks={'yt_ocean': '200MB'})

        # slice for time and latitudinal constraints
        time_slice = slice(start_time, end_time)
        lat_slice = slice(-90, lat_north)
        SST = SST.sel(time=time_slice, yt_ocean=lat_slice)
        SSS_PSU = SSS_PSU.sel(time=time_slice, yt_ocean=lat_slice)
        pot_rho_0 = pot_rho_0.sel(time=time_slice, yt_ocean=lat_slice)
        pot_rho_0 = pot_rho_0.isel(st_ocean=0).load() - 1000
        pme_river = pme_river.sel(time=time_slice, yt_ocean=lat_slice)
        net_sfc_heating = net_sfc_heating.sel(
            time=time_slice, yt_ocean=lat_slice)
        frazil_3d_int_z = frazil_3d_int_z.sel(
            time=time_slice, yt_ocean=lat_slice)
        lon_t = geolon_t.sel(yt_ocean=lat_slice)
        lat_t = geolat_t.sel(yt_ocean=lat_slice)

        # extract coordinate arrays
        yt_ocean = SST.yt_ocean.values
        xt_ocean = SST.xt_ocean.values
        st_ocean = cc.querying.getvar(expt, 'st_ocean', session, n=1).load()
        time = SST.time.values

        # construct an xarray of days per month
        months_standard_noleap = np.array(
            [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])
        days_per_month = xr.DataArray(
            months_standard_noleap,
            coords=[time], dims=['time'], name='days per month')

        # compute net surface heat flux from its component terms
        net_surface_heating = net_sfc_heating + frazil_3d_int_z
        # W/m2

        depth = -st_ocean[0].values # st_ocean value of the uppermost cell
        depth_tile = (lat_t*0+1)*depth
        pressure = xr.DataArray(
            p_from_z(depth_tile, lat_t), coords=[yt_ocean, xt_ocean],
            dims=['yt_ocean', 'xt_ocean'], name='pressure',
            attrs={'units': 'dbar'})

        # convert units to absolute salinity
        SSS = xr.DataArray(
            SA_from_SP(SSS_PSU, pressure, lon_t, lat_t),
            coords=[time, yt_ocean, xt_ocean],
            dims=['time', 'yt_ocean', 'xt_ocean'],
            name='sea surface salinity',
            attrs={'units': 'Absolute Salinity (g/kg)'})

        # compute salt transformation (no density binning)
        haline_contraction = xr.DataArray(
            beta(SSS, SST, pressure), coords=[time, yt_ocean, xt_ocean],
            dims=['time', 'yt_ocean', 'xt_ocean'],
            name='saline contraction coefficient (constant conservative temp)',
            attrs={'units': 'kg/g'})
        # Note that the salt fluxes have units of (kg of salt)/m^2/s, while beta has
        # units of kg / (g of salt), so we need to multiply the salt fluxes by 1000,
        # the fresh water flux pme_river has units of (kg of water)/m^2/s and needs
        # to be multiplied by SSS to convert to (g of salt)/m^2/s
        # units of salt_transformation is (kg of water)/m^2 but it will later be
        # divided by time and density and be in m/s:
        salt_transformation = haline_contraction * (
            SSS * pme_river - 1000 * (sfc_salt_flux_ice + sfc_salt_flux_restore)
            ) * days_per_month
        salt_transformation = salt_transformation.load()

        # compute heat transformation (no density binning)
        thermal_expansion = xr.DataArray(
            alpha(SSS, SST, pressure), coords=[time, yt_ocean, xt_ocean],
            dims=['time', 'yt_ocean', 'xt_ocean'],
            name='thermal expansion coefficient (constant conservative temp)',
            attrs={'units': '1/K'})
        heat_transformation = (
            thermal_expansion*net_surface_heating*days_per_month)
        heat_transformation = heat_transformation.load()

        # Isopycnal binning in several steps:
        # cycle through isopycnal bins, determine which cells are within the
        # given bin for each time step, find the transformation values for
        # those cells for each time step, sum these through time.
        # -> array of shape (isopyncal bins * lats * lons) where the array
        # associated with a given isopycnal bin is NaN everywhere except where
        # pot_rho_0 was within the bin, there it has a time summed
        # transformation value

        isopycnal_bins = np.arange(27.2, 28.11, 0.01)
        isopycnal_bin_mid = (isopycnal_bins[1:] + isopycnal_bins[:-1])/2

        binned_salt_transformation = xr.DataArray(
            np.zeros((len(time), len(isopycnal_bin_mid), len(yt_ocean),
                      len(xt_ocean))),
            coords=[time, isopycnal_bin_mid, yt_ocean, xt_ocean],
            dims=['time', 'isopycnal_bins', 'yt_ocean', 'xt_ocean'],
            name='salt transformation in isopycnal bins')
        binned_salt_transformation.chunk({'isopycnal_bins': 1})
        for i in range(len(isopycnal_bin_mid)):
            bin_mask = pot_rho_0.where(
                pot_rho_0 <= isopycnal_bin_mid[i] + 0.005).where(
                pot_rho_0 > isopycnal_bin_mid[i] - 0.005) * 0 + 1
            masked_transform = (salt_transformation * bin_mask)
            masked_transform = masked_transform.where(masked_transform != 0)
            masked_transform = masked_transform.load()
            binned_salt_transformation[:, i, :, :] = masked_transform

        binned_heat_transformation = xr.DataArray(
            np.zeros((len(time), len(isopycnal_bin_mid), len(yt_ocean),
                      len(xt_ocean))),
            coords=[time, isopycnal_bin_mid, yt_ocean, xt_ocean],
            dims=['time', 'isopycnal_bins', 'yt_ocean', 'xt_ocean'],
            name='heat transformation in isopycnal bins')
        binned_heat_transformation.chunk({'isopycnal_bins': 1})
        for i in range(len(isopycnal_bin_mid)-1):
            bin_mask = pot_rho_0.where(
                pot_rho_0 <= isopycnal_bin_mid[i] + 0.005).where(
                pot_rho_0 > isopycnal_bin_mid[i] - 0.005) * 0 + 1
            masked_transform = (heat_transformation * bin_mask)
            masked_transform = masked_transform.where(masked_transform != 0)
            masked_transform = masked_transform.load()
            binned_heat_transformation[:, i, :, :] = masked_transform

        salt_transformation = binned_salt_transformation/days_per_month
        c_p = 3992.1
        heat_transformation = binned_heat_transformation/c_p/days_per_month

        isopycnal_bin_diff = np.diff(isopycnal_bins)
        salt_transformation = salt_transformation/isopycnal_bin_diff[
            np.newaxis, :, np.newaxis, np.newaxis]
        heat_transformation = heat_transformation/isopycnal_bin_diff[
            np.newaxis, :, np.newaxis, np.newaxis]

        # this procedure defined fluxes from lighter to denser classes as
        # negative, now they are defined possitive
        salt_transformation = salt_transformation * -1
        heat_transformation = heat_transformation * -1

        # save as nc file
        ds = xr.Dataset({'binned_salt_transformation': salt_transformation})
        ds['binned_heat_transformation'] = heat_transformation
        ds.attrs = {'units': 'm/s'}
        enc = {'binned_salt_transformation':
           {'chunksizes': (12, 16, 255, 360),
            'zlib': True, 'complevel': 5, 'shuffle': True},
              'binned_heat_transformation':
           {'chunksizes': (12, 16, 255, 360),
            'zlib': True, 'complevel': 5, 'shuffle': True}}
        ds.to_netcdf(path_output + filename, encoding=enc)
        del (heat_transformation, salt_transformation)
        print('file', filename, 'saved at', path_output)

    start_time = str(year) + '-01-01'
    end_time = str(year) + '-12-31'
    filename_SWMT = ('SWMT_for_sig0_27-2_28-1_' + expt + '_' +
                     frequency[0:3:2] + '_' + str(year) + '.nc')
    calculate_SWMT_monthly_mean(expt, session, start_time, end_time, frequency,
                                path_output, filename_SWMT, lat_north=-59)

    """Surface water mass transformation (spatial sum) in DSW regions"""

    for a, area_text in enumerate(DSW_region['name']):
        mask = mask_from_polygon(DSW_region['lon'][a], DSW_region['lat'][a],
                                 xt_ocean, yt_ocean)
        mask = mask.where(mask == True, 0)
        if a == 0:
            mask_DSW = mask.expand_dims(area=[area_text])
        else:
            mask_DSW = xr.concat((mask_DSW, mask.expand_dims(
                area=[area_text])), dim='area')
    mask_DSW = mask_DSW.where(mask_DSW != 0)

    ds_SWMT = xr.open_dataset(path_output + filename_SWMT)
    swmt_heat = ds_SWMT.binned_heat_transformation
    swmt_salt = ds_SWMT.binned_salt_transformation

    swmt_heat_shelf = shelf_mask_isobath(swmt_heat)
    swmt_salt_shelf = shelf_mask_isobath(swmt_salt)
    area_t_shelf = shelf_mask_isobath(area_t)

    area_t_DSW = (mask_DSW * area_t.sel(yt_ocean=slice(-90, -59)))
    swmt_heat_DSW_regions = (swmt_heat_shelf * area_t_DSW/1e6).sum(
        ['xt_ocean', 'yt_ocean']).compute()
    swmt_salt_DSW_regions = (swmt_salt_shelf * area_t_DSW/1e6).sum(
        ['xt_ocean', 'yt_ocean']).compute()

    swmt_heat_DSW_regions.name = 'binned_heat_transformation_in_AABW_region'
    ds = swmt_heat_DSW_regions.to_dataset()
    ds['binned_salt_transformation_in_AABW_region'] = swmt_salt_DSW_regions
    ds.attrs = {'units': 'Sv'}
    comp = dict(zlib=True, complevel=5, shuffle=True)
    enc = {var: comp for var in ds.data_vars}
    ds.to_netcdf(
        path_output + 'SWMT_in_AABW_formation_region_for_sig0_27-2_28-1_' +
        expt + '_' + frequency[0:3:2] + '_' + str(year) + '.nc',
        encoding=enc)
