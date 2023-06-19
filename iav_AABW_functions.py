import numpy as np
import xarray as xr
import matplotlib.path as mpath
import cosima_cookbook as cc
import scipy.signal as scipy_sig
import scipy.stats as scipy_stats
import warnings
warnings.simplefilter('ignore')

def shelf_mask_isobath(var, output_mask=False):
    '''
    Masks ACCESS-OM2-01 variables by the region polewards of the 1000m isobath as computed using 
    a script contributed by Adele Morrison.
    Only to be used with ACCESS-OM2-0.1 output!
    '''
    contour_file = np.load('/g/data/ik11/grids/Antarctic_slope_contour_1000m.npz')
    
    shelf_mask = contour_file['contour_masked_above']
    yt_ocean = contour_file['yt_ocean']
    xt_ocean = contour_file['xt_ocean']
    
    # in this file the points along the isobath are given a positive value, the points outside (northwards) 
    # of the isobath are given a value of -100 and all the points on the continental shelf have a value of 0 
    # so we mask for the 0 values 
    shelf_mask[np.where(shelf_mask!=0)] = np.nan
    shelf_mask = shelf_mask+1
    shelf_map = np.nan_to_num(shelf_mask)
    shelf_mask = xr.DataArray(shelf_mask, coords = [('yt_ocean', yt_ocean), ('xt_ocean', xt_ocean)])
    shelf_map = xr.DataArray(shelf_map, coords = [('yt_ocean', yt_ocean), ('xt_ocean', xt_ocean)])
    
    # then we want to multiply the variable with the mask so we need to account for the shape of the mask. 
    # The mask uses a northern cutoff of 59S.
    masked_var = var.sel(yt_ocean = slice(-90, -59.03)) * shelf_mask
    
    if output_mask == True:
        return masked_var, shelf_map
    else:
        return masked_var
    
    
def mask_from_polygon(lon, lat, xt_ocean, yt_ocean):
    polygon = [(lon[0], lat[0])]
    for l in range(1, len(lon)):
        polygon += [(lon[l], lat[l])]
    poly_path = mpath.Path(polygon)

    x, y = xr.broadcast(xt_ocean, yt_ocean)
    coors = np.hstack((x.values.reshape(-1, 1), y.values.reshape(-1, 1)))

    mask = poly_path.contains_points(coors)
    mask = mask.reshape(xt_ocean.size, yt_ocean.size).transpose()
    mask = xr.DataArray(
    mask, dims=['yt_ocean', 'xt_ocean'],
    coords={'xt_ocean': xt_ocean, 'yt_ocean': yt_ocean})
    return mask


def fix_coordinates_of_ice_model(var, area):
    from datetime import timedelta
    var = var.where(var != 0)
    if var.attrs['coordinates'][:9] == 'TLON TLAT':
        var.coords['ni'] = area['xt_ocean'].values
        var.coords['nj'] = area['yt_ocean'].values
        var = var.rename({'ni':'xt_ocean', 'nj':'yt_ocean'})
    elif var.attrs['coordinates'][:9] == 'ULON ULAT':
        var.coords['ni'] = area['xu_ocean'].values
        var.coords['nj'] = area['yu_ocean'].values
        var = var.rename({'ni':'xu_ocean', 'nj':'yu_ocean'})
    if (var.time[1] - var.time[0])/(1e9*60*60) == 24:
        print('Fixed time assuming daily resolution.')
        var['time'] = var.time - np.timedelta64(12, 'h')
    else:
        print('Fixed time assuming monthly resolution')
        var['time'] = (var.time - np.timedelta64(12, 'h') -
                       np.timedelta64(15, 'D'))
    return var

def yearly_mean(var):
    month_length = var.time.dt.days_in_month
    weights_month = (month_length.groupby('time.year') /
                     month_length.groupby('time.year').sum())
    var = (var * weights_month).groupby('time.year').sum()
    var = var.rename({'year': 'time'})
    return var


def yearly_sum(var):
    month_length = var.time.dt.days_in_month
    var = (var * month_length).groupby('time.year').sum()
    var = var.rename({'year': 'time'})
    return var


def yearly_mean_different_start_1D(var, start_month=1, num_months=12,
                                   year_all=range(1958, 2019)):
    """ calculate a yearly time series using a specific start month and averaging over 
    a certain number of months e.g., for start_month = 3 and num_months = 6 an annual
    time series is calculated as a sum of the months March to August
    
    var - needs to be a DataArray with only the dimension "time"
    start_month - integer between 1 and 12
    num_months - integer between 1 and 12"""

    month_all = np.tile(np.arange(1, 13), 2)
    
    assert len(var.shape) == 1, "variable has more than 1 dimension"

    # select specified month
    var = var.sel(time=var.time.dt.month.isin(
        month_all[start_month-1:start_month-1+num_months]))

    # remove time steps that are before the start month and the last ones so
    # that the length of the time series can be divided by the number of months
    ind_start_month = var.indexes["time"].get_loc(
        str(year_all[0]) + '-' + str(start_month) + '-16T12', method='nearest')
    if ind_start_month != 0:
        var = var[ind_start_month:-(num_months-ind_start_month)]
    assert len(var.time) % num_months == 0,\
        "length of time axis can not be divided by number of months"
    start_date = var.time[0].values
    
    # weights determined by the number of days of each month
    month_length = var.time.dt.days_in_month
    month_length = xr.apply_ufunc(np.reshape, month_length, (-1, num_months),
                   input_core_dims=[['time'], []],
                   output_core_dims=[['time'] + ['time_new']],
                   exclude_dims=set(('time',)))
    weights_month = month_length / month_length.sum('time_new')

    # reshape time axis into len(years) x len(num_months)
    var = np.reshape(var.values, (-1, num_months))

    # calculate weighted average
    var = (var*weights_month).sum(axis=1)
    var = xr.DataArray(
        var, dims='time', coords={'time': year_all[:var.size]},
        attrs = {'start_month': start_month, 'number_of_months': num_months,
                 'start_date': start_date,
                 'average': 'weighted by length of month'})
    return var


def yearly_mean_different_start(var, start_month=1, num_months=12,
                                year_all=range(1958, 2019)):
    """ calculate a yearly time series using a specific start month and averaging over 
    a certain number of months e.g., for start_month = 3 and num_months = 6 an annual
    time series is calculated as an average of the months March to August
    
    var - needs to be an DataArray with the dimension "time" as the first dimension
    start_month - integer between 1 and 12
    num_months - integer between 1 and 12"""
    
    month_all = np.tile(np.arange(1, 13), 2)

    assert var.dims[0] == 'time', "time is not the first dimension"
    
    # select specified month
    var = var.sel(time=var.time.dt.month.isin(
        month_all[start_month-1:start_month-1+num_months]))

    # remove time steps that are before the start month and the last ones so
    # that the length of the time series can be divided by the number of months
    ind_start_month = var.indexes["time"].get_loc(
        '1958-' + str(start_month) + '-16T12', method='nearest')
    if ind_start_month != 0:
        var = var[ind_start_month:-(num_months-ind_start_month)]
    assert len(var.time) % num_months == 0,\
        "length of time axis can not be divided by number of months"
    start_date = var.time[0].values

    # weights determined by the number of days of each month
    month_length = var.time.dt.days_in_month
    month_length = xr.apply_ufunc(np.reshape, month_length, (-1, num_months),
                   input_core_dims=[['time'], []],
                   output_core_dims=[['time'] + ['time_new']],
                   exclude_dims=set(('time',)))
    weights_month = month_length / month_length.sum('time_new')

    # reshape time axis into len(years) x len(num_months)
    var = xr.apply_ufunc(np.reshape, var, (-1, num_months) + var.shape[1:],
                   input_core_dims=[list(var.dims), []],
                   output_core_dims=[['time'] + ['time_new'] + list(var.dims[1:])],
                   exclude_dims=set(('time',)))

    # calculate weighted average
    var = (var*weights_month).sum('time_new')
    var = var.assign_coords({'time': year_all[:var.time.size]})
    var.attrs = {'start_month': start_month, 'number_of_months': num_months,
                 'start_date': start_date,
                 'average': 'weighted by length of month'}
    return var


def select_bottom_values(var, expt, session, lat_north=-59):
    from gsw import p_from_z
    var = var.sel(yt_ocean=slice(-90, lat_north))
    var = var.where(var != 0)
    
    ht = cc.querying.getvar(expt, 'ht', session, n=1) 
    ht = ht.sel(yt_ocean=slice(-90, lat_north))
    land_mask = (ht*0).fillna(1)

    # select bottom values
    depth_array = var*0 + var.st_ocean
    max_depth = depth_array.max(dim='st_ocean', skipna=True)

    var_bottom = var.where(depth_array.st_ocean >= max_depth)
    var_bottom = var_bottom.sum(dim='st_ocean')
    var_bottom = var_bottom.where(land_mask == 0)
    return var_bottom


def mask_1000m_isobath_bottom(var):
    '''
    Masks ACCESS-OM2-01 variables by the region polewards of the 1000m isobath as computed using 
    a script contributed by Adele Morrison.
    Only to be used with ACCESS-OM2-0.1 output!
    '''
    contour_file = np.load('/g/data/ik11/grids/Antarctic_slope_contour_1000m.npz')

    shelf_mask = contour_file['contour_masked_above']
    contour_mask_numbered = contour_file['contour_mask_numbered']
    yt_ocean = contour_file['yt_ocean']
    xt_ocean = contour_file['xt_ocean']

    shelf_mask = xr.DataArray(shelf_mask, dims=['yt_ocean', 'xt_ocean'],
                              coords=[yt_ocean, xt_ocean])
    contour_mask_numbered = xr.DataArray(contour_mask_numbered,
                                         coords = [('yt_ocean', yt_ocean),
                                                   ('xt_ocean', xt_ocean)])

    # Create the contour order data-array.

    # stack contour data into 1d:
    contour_mask_numbered_1d = contour_mask_numbered.stack(contour_index = ['yt_ocean', 'xt_ocean'])
    contour_mask_numbered_1d = contour_mask_numbered_1d.where(contour_mask_numbered_1d > 0, drop = True)

    contour_ordering = contour_mask_numbered_1d.sortby(contour_mask_numbered_1d)
    contour_index_array = np.arange(1,len(contour_ordering)+1)

    # get lat and lon along contour, useful for plotting later:
    lat_along_contour = contour_ordering.yt_ocean
    lon_along_contour = contour_ordering.xt_ocean
    # don't need the multi-index anymore, replace with contour count and save
    lat_along_contour.coords['contour_index'] = contour_index_array
    lon_along_contour.coords['contour_index'] = contour_index_array


    # inititalise empty array
    if 'time' in var.dims:
        var_along_contour = xr.DataArray(
            np.zeros((len(var.time), len(contour_index_array))),
            coords = [var.time, contour_index_array],
            dims = ['time', 'contour_index'], name = 'var_along_contour')
    else:
            var_along_contour = xr.DataArray(
            np.zeros(len(contour_index_array)),
            coords = [contour_index_array],
            dims = ['contour_index'], name = 'var_along_contour')

    # stack transports into 1d and drop any points not on contour:
    var_along_contour_1d = (var.where(contour_mask_numbered>0)).stack(contour_index = ['yt_ocean', 'xt_ocean'])
    var_along_contour_1d = var_along_contour_1d.where(contour_mask_numbered_1d>0,drop=True)
    # sort by contour index:
    var_along_contour_1d = var_along_contour_1d.sortby(contour_ordering)

    # assign contour index and load:
    var_along_contour_1d.coords['contour_index'] = contour_index_array
    var_along_contour_1d = var_along_contour_1d.assign_coords(
        {'lon': lon_along_contour, 'lat': lat_along_contour})
    return var_along_contour_1d


def correlation_3D(x, y, dims='time', return_p_value_text_Neff=False):
    def detrend(da, dims):
        """Wrapper function for applying detrending and returning as DataArray"""
        da_detrended = xr.DataArray(scipy_sig.detrend(
            da, axis=da.get_axis_num(dims)), dims=da.dims, coords=da.coords)
        return da_detrended
    
    # Detrend data
    x = detrend(x, dims)
    y = detrend(y, dims)
    
    # Compute covariance
    cov = (xr.dot(x - x.mean(dims), (y - y.mean(dims)).drop(dims), dims=dims) /
           x.count(dims))
    
    # Compute correlation along time axis
    cor = cov / (x.std(dims) * y.std(dims))
    
    # Compute effective degrees of freedom
    # calculate autocorrelation function
    xauto = xr.apply_ufunc(np.correlate, x, x, 'full',
                           input_core_dims=[[dims], [dims], []],
                           output_core_dims=[["lag"]],
                           exclude_dims=set((dims,)),
                           vectorize=True) / (x.count(dims)*x.std(dims)**2)
    xauto = xauto.assign_coords(
        lag=np.arange(-x.count(dims).mean()+1, x.count(dims).mean()))
    yauto = xr.apply_ufunc(np.correlate, y, y, 'full',
                           input_core_dims=[[dims], [dims], []],
                           output_core_dims=[["lag"]],
                           exclude_dims=set((dims,)),
                           vectorize=True) / (y.count(dims)*y.std(dims)**2)
    yauto = yauto.assign_coords(
        lag=np.arange(-y.count(dims).mean()+1, y.count(dims).mean()))

    # interpolate product of both autocorrelation functions
    # on time axis with higher resolution dt
    dt = 0.2
    new_axis = np.arange(-x.count(dims).mean()+1, x.count(dims).mean()-1+dt, dt)
    xauto_yauto_int = (xauto * yauto).interp(lag=new_axis)

    # calculate effective degrees of freedom
    ## index of lag 0
    tcen = int((len(new_axis)-1)/2)
    xauto_yauto_int_no_nan = xauto_yauto_int.isel(lag=slice(tcen, -1))
    xauto_yauto_int_no_nan = xauto_yauto_int_no_nan.where(
        np.isnan(xauto_yauto_int_no_nan) == False, 0)
    # timescale over which product of autocorrelation functions decays to 1/e
    Tx = (xauto_yauto_int_no_nan >= xauto_yauto_int_no_nan.isel(lag=0)/np.e).argmin('lag')

    N_eff = x.count(dims) / (2*dt*xauto_yauto_int.isel(lag=slice(tcen, None)).where(
        xr.DataArray(np.arange(tcen, tcen*2+1), dims='lag') <= (tcen+Tx)).sum(
        'lag', skipna=True))
    N_eff = N_eff.where(np.isinf(N_eff) == False)
    N_eff = N_eff.where((N_eff < x.count(dims).mean()) |
                        (np.isnan(N_eff) == True),
                        x.count(dims).mean())
    
    # Compute P-value using the students t test
    t_stat = cor*np.sqrt(N_eff)/(np.sqrt(1-cor*cor))
    p = (1.0 - scipy_stats.t.cdf(abs(t_stat), N_eff)) * 2.0
    p = xr.DataArray(p, dims=cor.dims, coords=cor.coords)
    
    if return_p_value_text_Neff == True:
        if p < 0.01:
            p_str = 'p<0.01'
        elif p < 0.05:
            p_str = 'p<0.05'
        else:
            p_str = 'not sign.'
        return cor, p, p_str, N_eff
    else:
        return cor, p