"""
Louis Rosas 2023, BYU Hydroinformatics lab
Based off work by Cedric H. David and Riley Hales
"""

import xarray as xr
import pandas as pd
from datetime import datetime
import numpy as np
from netCDF4 import Dataset
from pytz import utc

def inflow_fast(lsm_file_list: list, weight_table: str, rapid_connect: str, out_nc_file: str, comid_lat_lon_z: str, 
    simulation_start_datetime: datetime, simulation_end_datetime: datetime, conversion_factor: float = 1.0) -> None:
    """
    Generate inflow files for RAPID. Note that the order of the streams in the csvs (besides the weight table) should all match. 
    If you get warnings from hdf5, run 'pip uninstall numpy' and then 'conda install numpy'

    Parameters
    ----------
    lsm_file_list: list
        List of netcdf files in the format YYYYMMDD.nc that contain a variable 'ro', or runoff
    weight_table: str
        Path to weight table
    rapid_connect: str
        Path to rapid connect file
    out_file: str
        Path and name of the output netcdf
    comid_lat_lon_z: str
        Path to the comid_lat_lon_z file
    simulation_start_datetime: datetime
        Simulation start time as a datetime object 
    simulation_end_datetime: datetime
        Simulation end time as a datetime object 
    conversion_factor: float, optional
        Number to multiply the run off by, default is 1 (no conversion)
    """
    # load in weight table and get some information
    weight_df = pd.read_csv(weight_table)

    date_range = pd.date_range(start=simulation_start_datetime, end=simulation_end_datetime)
    rivid_list = pd.read_csv(rapid_connect, dtype=int, header=None)[0].to_numpy()

    initial_time_seconds = (simulation_start_datetime.replace(tzinfo=utc) - datetime(1970, 1, 1, tzinfo=utc)).total_seconds()
    final_time_seconds = initial_time_seconds + date_range.shape[0] * 86400 # This is the number of seconds in a day
    time_array = np.arange(initial_time_seconds, final_time_seconds, 86400)

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

    # open all the ncs and select only the area in the weight table to an array
    # check if we need to merge a fourth dimension
    # We use fancy indexing to multiply the areas (This results in a big array, rows are each time step, columns are each reach id (which may be more than 1))
    # This is actaully faster than any numpy operations I've found. Make a dataframe, group by stream id, sum up groups, and return a numpy array
    dataset = xr.open_mfdataset(lsm_file_list).isel(latitude=lat_slice, longitude=lon_slice, time=slice(0, len(time_array)))['ro']
    
    # if the input nc files/file is not daily, get the time array directly from the dataset (probably could be done always but not sure it that's the case)
    if time_array.shape[0] != len(dataset['time']):
        time_array = dataset['time'].to_numpy()

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

        # time
        time_var = data_out_nc.createVariable('time', 'i4', ('time',))
        time_var.long_name = 'time'
        time_var.standard_name = 'time'
        time_var.units = f'seconds since {simulation_start_datetime}' # Must be seconds, otherwise it freaks out?
        time_var.axis = 'T'
        time_var.calendar = 'gregorian'
        time_var.bounds = 'time_bnds'

        time_var[:] = time_array

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
        comid_df = pd.read_csv(comid_lat_lon_z)
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