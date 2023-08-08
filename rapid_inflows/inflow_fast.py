import logging
import os
import sys
import re
from datetime import datetime

import netCDF4 as nc
import numpy as np
import pandas as pd
import psutil
import xarray as xr

logging.basicConfig(level=logging.INFO,
                    stream=sys.stdout,
                    format='%(asctime)s - %(levelname)s - %(message)s')


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
        raise MemoryError(
            f"Trying to allocate {psutil._common.bytes2human(num_bytes)} of "
            f"{psutil._common.bytes2human(available_mem)} available")
    if num_bytes >= available_mem * ram_buffer_percentage:
        print(f"WARNING: arrays will use ~{round(num_bytes / available_mem, 1)}% of \
        {psutil._common.bytes2human(available_mem)} available memory...")


def create_inflow_file(lsm_file_list: list,
                       weight_table: str = None,
                       comid_lat_lon_z: str = None,
                       inflow_file_path: str = None,
                       input_tuples: list = None,) -> None:
    """
    Generate inflow files for use with RAPID. The generated inflow file will sort the river ids in the order found in
    the comid_lat_lon_z csv. Either weight_table, comid_lat_lon_z, and inflow_file_path are defined explicitly, or 
    input_tuples is defined. 

    Parameters
    ----------
    lsm_file_list: list
        List of netcdf file paths. NCs should have dimensions order (time, latitude, longitude) or
        (time, expver, latitude, longitude)
    weight_table: str, list
        Path and name of the weight table
    comid_lat_lon_z: str
        Path to the comid_lat_lon_z.csv corresponding to the weight table
    inflow_file_path: str
        Path and name of the output netcdf
    input_tuples: list
        A list of iterable objects likle the following: [(weight.csv, comid.csv, out.nc),(weight_1.csv), comid_1.csv, out_1.nc), ...]
    """
    # Create the input_tuples object if not created already: This will be what will be iterated
    if input_tuples is None:
        if (weight_table is not None and comid_lat_lon_z is not None and inflow_file_path is not None):
            input_tuples = [(weight_table, comid_lat_lon_z, inflow_file_path)]
        else:
            raise ValueError("Either input_tuples OR all other inputs must be defined.")

    # Ensure that every input file exists
    for weight_table, comid_lat_lon_z, _ in input_tuples: 
        if not os.path.exists(weight_table):
            raise FileNotFoundError(f'{weight_table} does not exist')
        if not os.path.exists(comid_lat_lon_z):
            raise FileNotFoundError(f'{comid_lat_lon_z} does not exist')

    # open all the ncs and select only the area within the weight table
    logging.info('Opening LSM files multi-file dataset')
    lsm_dataset = xr.open_mfdataset(lsm_file_list)

    # Select the variable names
    runoff_variable = [x for x in ['ro', 'RO', 'runoff', 'RUNOFF'] if x in lsm_dataset.variables][0]
    lon_variable = [x for x in ['lon', 'longitude', 'LONGITUDE', 'LON'] if x in lsm_dataset.variables][0]
    lat_variable = [x for x in ['lat', 'latitude', 'LATITUDE', 'LAT'] if x in lsm_dataset.variables][0]

    # Check that the input table dimensions match the dataset dimensions
    # We ignore time dimension
    variable_dims = lsm_dataset[runoff_variable].dims
    dataset_shape = [lsm_dataset[runoff_variable].shape[variable_dims.index(lat_variable)], 
                    lsm_dataset[runoff_variable].shape[variable_dims.index(lon_variable)]]

    for weight_table, _ ,_ in input_tuples:
        matches = re.findall(r'(\d+)x(\d+)', weight_table)[0]
        if len(matches) == 2:
            if all(int(item) in dataset_shape for item in matches):
                continue
            raise ValueError(f"{weight_table} dimensions don't match the input dataset shape: {dataset_shape}")
        raise ValueError(f"Could not find a shape (###x####) in {weight_table}. Consider renaming")

    # Iterate over each weight table
    for weight_table, comid_lat_lon_z, inflow_file_path in input_tuples:
        # load in weight table and get some information
        logging.info('Reading weight table and comid_lat_lon_z csvs')
        weight_df = pd.read_csv(weight_table)
        comid_df = pd.read_csv(comid_lat_lon_z)

        sorted_rivid_array = comid_df.iloc[:, 0].to_numpy()

        min_lon = weight_df['lon'].min()
        max_lon = weight_df['lon'].max()
        min_lat = weight_df['lat'].min()
        max_lat = weight_df['lat'].max()

        min_lon_idx = weight_df.loc[weight_df['lon'] == min_lon, 'lon_index'].values[0].astype(int)
        max_lon_idx = weight_df.loc[weight_df['lon'] == max_lon, 'lon_index'].values[0].astype(int)
        min_lat_idx = weight_df.loc[weight_df['lat'] == min_lat, 'lat_index'].values[0].astype(int)
        max_lat_idx = weight_df.loc[weight_df['lat'] == max_lat, 'lat_index'].values[0].astype(int)

        if min_lon_idx > max_lon_idx:
            min_lon_idx, max_lon_idx = max_lon_idx, min_lon_idx
        if min_lat_idx > max_lat_idx:
            min_lat_idx, max_lat_idx = max_lat_idx, min_lat_idx

        # for readability, select certain cols from the weight table
        n_wt_rows = weight_df.shape[0]
        stream_ids = weight_df.iloc[:, 0].to_numpy()
        lat_indices = weight_df['lat_index'].values - min_lat_idx
        lon_indices = weight_df['lon_index'].values - min_lon_idx

        spatial_slices = {lon_variable: slice(min_lon_idx, max_lon_idx + 1),
                        lat_variable: slice(min_lat_idx, max_lat_idx + 1)}

        ds = (
            lsm_dataset
            .isel(**spatial_slices)
            [runoff_variable]
        )

        # Get approximate sizes of arrays and check if we have enough memory
        logging.info('Checking anticipated memory requirement')
        out_array_size = ds['time'].shape[0] * sorted_rivid_array.shape[0]
        in_array_size = ds['time'].shape[0] * n_wt_rows
        if ds.ndim == 4:
            in_array_size *= 2
        total_size = out_array_size + in_array_size
        _memory_check(total_size)

        # Get conversion factor
        logging.info('Getting conversion factor')
        conversion_factor = 1
        units = ds.attrs.get('units', False)
        if not units:
            logging.warning("No units attribute found. Assuming meters")
        elif ds.attrs['units'] == 'm':
            conversion_factor = 1
        elif ds.attrs['units'] == 'mm':
            conversion_factor = .001
        else:
            raise ValueError(f"Unknown units: {ds.attrs['units']}")

        # get the time array from the dataset
        logging.info('Reading Time values')
        datetime_array = ds['time'].to_numpy()

        logging.info('Creating inflow array')
        # todo more checks on names and order of dimensions
        if ds.ndim == 3:
            inflow_array = ds.values[:, lat_indices, lon_indices]
        elif ds.ndim == 4:
            inflow_array = ds.values[:, :, lat_indices, lon_indices]
            inflow_array = np.where(np.isnan(inflow_array[:, 0, :]), inflow_array[:, 1, :], inflow_array[:, 0, :]),
        else:
            raise ValueError(f"Unknown number of dimensions: {ds.ndim}")
        inflow_array = np.nan_to_num(inflow_array, nan=0)
        inflow_array[inflow_array < 0] = 0
        inflow_array = inflow_array * weight_df['area_sqm'].values * conversion_factor
        inflow_array = pd.DataFrame(inflow_array, columns=stream_ids)
        inflow_array = inflow_array.groupby(by=stream_ids, axis=1).sum()
        inflow_array = inflow_array[sorted_rivid_array].to_numpy()

        ds.close()

        # Create output inflow netcdf data
        logging.info("Writing inflows to file")
        os.makedirs(os.path.dirname(inflow_file_path), exist_ok=True)
        with nc.Dataset(inflow_file_path, "w", format="NETCDF3_CLASSIC") as inflow_nc:
            # create dimensions
            inflow_nc.createDimension('time', datetime_array.shape[0])
            inflow_nc.createDimension('rivid', sorted_rivid_array.shape[0])
            inflow_nc.createDimension('nv', 2)

            # m3_riv
            m3_riv_var = inflow_nc.createVariable('m3_riv', 'f4', ('time', 'rivid'), fill_value=0)
            m3_riv_var[:] = inflow_array
            m3_riv_var.long_name = 'accumulated inflow inflow volume in river reach boundaries'
            m3_riv_var.units = 'm3'
            m3_riv_var.coordinates = 'lon lat'
            m3_riv_var.grid_mapping = 'crs'
            m3_riv_var.cell_methods = "time: sum"

            # rivid
            rivid_var = inflow_nc.createVariable('rivid', 'i4', ('rivid',))
            rivid_var[:] = sorted_rivid_array
            rivid_var.long_name = 'unique identifier for each river reach'
            rivid_var.units = '1'
            rivid_var.cf_role = 'timeseries_id'

            # time
            reference_time = datetime_array[0]
            time_step = (datetime_array[1] - datetime_array[0]).astype('timedelta64[s]')
            time_var = inflow_nc.createVariable('time', 'i4', ('time',))
            time_var[:] = (datetime_array - reference_time).astype('timedelta64[s]').astype(int)
            time_var.long_name = 'time'
            time_var.standard_name = 'time'
            time_var.units = f'seconds since {reference_time.astype("datetime64[s]")}'  # Must be seconds
            time_var.axis = 'T'
            time_var.calendar = 'gregorian'
            time_var.bounds = 'time_bnds'

            # time_bnds
            time_bnds = inflow_nc.createVariable('time_bnds', 'i4', ('time', 'nv',))
            time_bnds_array = np.stack([datetime_array, datetime_array + time_step], axis=1)
            time_bnds_array = (time_bnds_array - reference_time).astype('timedelta64[s]').astype(int)
            time_bnds[:] = time_bnds_array

            # longitude
            lon_var = inflow_nc.createVariable('lon', 'f8', ('rivid',), fill_value=-9999.0)
            lon_var[:] = comid_df['lon'].values
            lon_var.long_name = 'longitude of a point related to each river reach'
            lon_var.standard_name = 'longitude'
            lon_var.units = 'degrees_east'
            lon_var.axis = 'X'

            # latitude
            lat_var = inflow_nc.createVariable('lat', 'f8', ('rivid',), fill_value=-9999.0)
            lat_var[:] = comid_df['lat'].values
            lat_var.long_name = 'latitude of a point related to each river reach'
            lat_var.standard_name = 'latitude'
            lat_var.units = 'degrees_north'
            lat_var.axis = 'Y'

            # crs
            crs_var = inflow_nc.createVariable('crs', 'i4')
            crs_var.grid_mapping_name = 'latitude_longitude'
            crs_var.epsg_code = 'EPSG:4326'  # WGS 84
            crs_var.semi_major_axis = 6378137.0
            crs_var.inverse_flattening = 298.257223563

            # add global attributes
            inflow_nc.Conventions = 'CF-1.6'
            inflow_nc.history = 'date_created: {0}'.format(datetime.utcnow())
            inflow_nc.featureType = 'timeSeries'
            inflow_nc.geospatial_lat_min = min_lat
            inflow_nc.geospatial_lat_max = max_lat
            inflow_nc.geospatial_lon_min = min_lon
            inflow_nc.geospatial_lon_max = max_lon

    lsm_dataset.close()
    return
