"""
Louis Rosas 2023, BYU Hydroinformatics lab
Based off work by Cedric H. David and Riley Hales
"""

import xarray as xr
import pandas as pd
import numpy as np
import os
import psutil

from datetime import datetime
from netCDF4 import Dataset
from pytz import utc

def inflow_fast(lsm_file_list: list, weight_table: str, inputs: str, out_nc_file: str) -> None:
    """
    Generate inflow files for RAPID. Note that the order of the streams in the csvs (besides the weight table) should all match. 
    If you get warnings from hdf5, run 'pip uninstall numpy' and then 'conda install numpy'

    Parameters
    ----------
    lsm_file_list: list
        List of netcdf files that contain a variable 'ro'. Dimension expected in the order (time, latitude, longitude) or (time, expver, latitude, longitude)
    weight_table: str
        Path and name of the weight table
    inputs: str
        Path to the directory containing 'comid_lat_lon_z.csv'
    out_nc_file: str
        Path and name of the output netcdf
    """
    # load in weight table and get some information
    weight_df = pd.read_csv(weight_table)

    # load in comid_lat_lon_csv
    comid_lat_lon_z = os.path.join(inputs, 'comid_lat_lon_z.csv')
    if not os.path.exists(comid_lat_lon_z):
        raise ValueError(f'{comid_lat_lon_z} does not exist')

    comid_df = pd.read_csv(comid_lat_lon_z)
    rivid_list = comid_df.iloc[:,0].to_numpy()

    min_lon = weight_df['lon'].min()
    max_lon = weight_df['lon'].max()
    min_lat = weight_df['lat'].min()
    max_lat = weight_df['lat'].max()

    min_lon_i = weight_df.loc[weight_df['lon'] == min_lon, 'lon_index'].values[0]
    max_lon_i = weight_df.loc[weight_df['lon'] == max_lon, 'lon_index'].values[0]
    min_lat_i = weight_df.loc[weight_df['lat'] == min_lat, 'lat_index'].values[0]
    max_lat_i = weight_df.loc[weight_df['lat'] == max_lat, 'lat_index'].values[0]

    if min_lon_i > max_lon_i:
        min_lon_i, max_lon_i = max_lon_i, min_lon_i
    if min_lat_i > max_lat_i:
        min_lat_i, max_lat_i = max_lat_i, min_lat_i

    lon_slice = slice(min_lon_i, max_lon_i+1)
    lat_slice = slice(min_lat_i, max_lat_i+1)

    # for readability, select certain cols from the weight table
    stream_ids = weight_df['streamID'].values
    lat_indices = weight_df['lat_index'].values - min_lat_i
    lon_indices = weight_df['lon_index'].values - min_lon_i
    area_sqm = weight_df['area_sqm'].values

    # open all the ncs and select only the area within the weight table 
    dataset = (
        xr.open_mfdataset(lsm_file_list)
        .isel(latitude=lat_slice, longitude=lon_slice)
        ['ro']
    )

    # Get aproximate sizes of arrays and check if we have enough memory
    out_array_size = dataset['time'].shape[0] * rivid_list.shape[0]
    in_array_size = dataset['time'].shape[0] * area_sqm.shape[0]
    if dataset.ndim == 4:
        in_array_size *= 2
        total_size = out_array_size + in_array_size
    else:
        total_size = np.maximum(in_array_size, out_array_size)
    _memory_check(total_size)
    
    # Get conversion factor
    if dataset.attrs['units'] == 'm':
        conversion_factor = 1
    elif dataset.attrs['units'] == 'mm':
        conversion_factor = .001
    else:
        print(f"WARNING: unsupported conversion factor {dataset.attrs['units']} found. Using one by default...")

    # get the time array from the dataset
    time_array = dataset['time'].to_numpy()

    # check if we need to merge a fourth dimension
    # We use fancy indexing to multiply the areas (This results in a big array, rows are each time step, columns are each reach id (which may be more than 1))
    # This is actaully faster than any numpy operations I've found. Make a dataframe, group by stream id, sum up groups, and return a numpy array
    if dataset.ndim == 3:
        output_array = (
            pd.DataFrame(
                dataset.values[:, lat_indices, lon_indices] * area_sqm * conversion_factor,
                columns=stream_ids
            )
            .groupby(by=stream_ids, axis=1)
            .sum()
            .to_numpy()
        )
    elif dataset.ndim == 4:
        ro = dataset.values[:,:,lat_indices, lon_indices] * area_sqm * conversion_factor
        output_array = (
            pd.DataFrame(
                np.where(np.isnan(ro[:,0,:]), ro[:,1,:], ro[:,0,:]),
                columns=stream_ids
            )
            .groupby(by=stream_ids, axis=1)
            .sum()
            .to_numpy()
        )
        del ro
    else:
        ndims = dataset.ndim
        dataset.close()
        raise ValueError(f"Got {ndims} dimensions; expected 3 or 4")
    
    dataset.close()

    # Negative and zero values are set to nan
    output_array = np.where(output_array <=0, np.nan, output_array)

    # Create output inflow netcdf data
    print("Generating inflow file ...")
    data_out_nc = Dataset(out_nc_file, "w", format="NETCDF3_CLASSIC")

    # create dimensions
    data_out_nc.createDimension('time', time_array.shape[0])
    data_out_nc.createDimension('rivid', len(rivid_list))
    data_out_nc.createDimension('nv', 2)

    # m3_riv
    m3_riv_var = data_out_nc.createVariable('m3_riv', 'f4', ('time', 'rivid'), fill_value=0)
    m3_riv_var.long_name = 'accumulated external water volume ' \
                            'inflow upstream of each river reach'
    m3_riv_var.units = 'm3'
    m3_riv_var.coordinates = 'lon lat'
    m3_riv_var.grid_mapping = 'crs'
    m3_riv_var.cell_methods = "time: sum"

    try:
        # rivid
        rivid_var = data_out_nc.createVariable('rivid', 'i4', ('rivid',))
        rivid_var.long_name = 'unique identifier for each river reach'
        rivid_var.units = '1'
        rivid_var.cf_role = 'timeseries_id'

        rivid_var[:] = rivid_list

        time_var = data_out_nc.createVariable('time', 'i4', ('time',))
        time_var.long_name = 'time'
        time_var.standard_name = 'time'
        time_var.units = f'seconds since {time_array[0]}' # Must be seconds
        time_var.axis = 'T'
        time_var.calendar = 'gregorian'
        time_var.bounds = 'time_bnds'

        time_var[:] = (time_array - time_array[0]).astype('timedelta64[s]')

        # time_bnds
        _ = data_out_nc.createVariable('time_bnds', 'i4', ('time', 'nv',))

        # longitude
        lon_var = data_out_nc.createVariable('lon', 'f8', ('rivid',), fill_value=-9999.0)
        lon_var.long_name = 'longitude of a point related to each river reach'
        lon_var.standard_name = 'longitude'
        lon_var.units = 'degrees_east'
        lon_var.axis = 'X'

        # latitude
        lat_var = data_out_nc.createVariable('lat', 'f8', ('rivid',), fill_value=-9999.0)
        lat_var.long_name = 'latitude of a point related to each river reach'
        lat_var.standard_name = 'latitude'
        lat_var.units = 'degrees_north'
        lat_var.axis = 'Y'

        crs_var = data_out_nc.createVariable('crs', 'i4')
        crs_var.grid_mapping_name = 'latitude_longitude'
        crs_var.epsg_code = 'EPSG:4326'  # WGS 84
        crs_var.semi_major_axis = 6378137.0
        crs_var.inverse_flattening = 298.257223563

        # add global attributes
        data_out_nc.Conventions = 'CF-1.6'
        data_out_nc.history = 'date_created: {0}'.format(datetime.utcnow())
        data_out_nc.featureType = 'timeSeries'

        # Get relevant arrays while we update them
        lats = data_out_nc.variables['lat'][:]
        lons = data_out_nc.variables['lon'][:]

        # Process each row in the comid table
        lats = comid_df['lat'].values
        lons = comid_df['lon'].values

        # Overwrite netCDF variable values
        data_out_nc.variables['lat'][:] = lats
        data_out_nc.variables['lon'][:] = lons

        # Update metadata
        data_out_nc.geospatial_lat_min = lats.min()
        data_out_nc.geospatial_lat_max = lats.max()
        data_out_nc.geospatial_lon_min = lons.min()
        data_out_nc.geospatial_lon_max = lons.max()

        # insert the data
        data_out_nc.variables['m3_riv'][:] = output_array

    except RuntimeError:
        print("File size too big to add data beforehand."
                " Performing conversion after ...")
    finally:
        data_out_nc.close()

    print('Finished inflows') 

def _memory_check(size: int, dtype: type = np.float32, ram_buffer_percentage: float = 0.8):
    """
    Internal function to check if the arrays we create will be larger than the memory available. 
    Also warns against very large arrays. Default datatype of arrays is np.float32.
    By default, the warning will be announced if memory consumption is projected to be greater than 
    80% of avaiable memory
    """
    num_bytes = np.dtype(dtype).itemsize * size
    available_mem = psutil.virtual_memory().available

    if num_bytes >= available_mem:
        raise MemoryError(f"Trying to allocate {psutil._common.bytes2human(num_bytes)} of {psutil._common.bytes2human(available_mem)} available")
    if num_bytes >= available_mem * ram_buffer_percentage:
        print(f"WARNING: arrays will use ~{round(num_bytes/available_mem, 1)}% of \
        {psutil._common.bytes2human(available_mem)} available memory...")
