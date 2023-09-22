import argparse
import datetime
import glob
import logging
import os
import re
import sys

import netCDF4 as nc
import numpy as np
import pandas as pd
import psutil
import xarray as xr

logging.basicConfig(level=logging.INFO,
                    stream=sys.stdout,
                    format='%(asctime)s - %(levelname)s - %(message)s')

DESIRED_TIME_STEP = np.timedelta64(3,'h') # We want 3 hour time steps

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

def _is_cumulative(array: np.ndarray):
    return np.all(np.diff(array) >= 0)

def create_inflow_file(lsm_directory: str,
                       vpu_name: str,
                       inflows_dir: str,
                       weight_table: str,
                       comid_lat_lon_z: str, 
                       cumulative: bool = False, ) -> None:
    """
    Generate inflow files for use with RAPID.

    Parameters
    ----------
    lsm_directory: str
        Path to directory of LSM files which should end in .nc
    vpu_name: str
        Name of the vpu
    inflows_dir: str
        Path to directory where inflows will be saved
    weight_table: str, list
        Path and name of the weight table
    comid_lat_lon_z: str
        Path to the comid_lat_lon_z.csv corresponding to the weight table
    cumulative: bool, optional
        If true, we will process this as forecast data, meaning:
        1) We assume the input runoff is culmative and
        2) We will force the output time step to be 3 hours
        Default is False
    """

    # Ensure that every input file exists
    if not os.path.exists(weight_table):
        raise FileNotFoundError(f'{weight_table} does not exist')
    if not os.path.exists(comid_lat_lon_z):
        raise FileNotFoundError(f'{comid_lat_lon_z} does not exist')

    # open all the ncs and select only the area within the weight table
    logging.info('Opening LSM files multi-file dataset')
    lsm_dataset = xr.open_mfdataset(sorted(glob.glob(os.path.join(lsm_directory, '*.nc'))))

    # Select the variable names
    runoff_variable = [x for x in ['ro', 'RO', 'runoff', 'RUNOFF'] if x in lsm_dataset.variables][0]
    lon_variable = [x for x in ['lon', 'longitude', 'LONGITUDE', 'LON'] if x in lsm_dataset.variables][0]
    lat_variable = [x for x in ['lat', 'latitude', 'LATITUDE', 'LAT'] if x in lsm_dataset.variables][0]

    # Check that the input table dimensions match the dataset dimensions
    # This gets us the shape, while ignoring the time dimension
    variable_dims = lsm_dataset[runoff_variable].dims
    dataset_shape = [lsm_dataset[runoff_variable].shape[variable_dims.index(lat_variable)],
                     lsm_dataset[runoff_variable].shape[variable_dims.index(lon_variable)]]

    matches = re.findall(r'(\d+)x(\d+)', weight_table)[0]
    if len(matches) == 2:
        if all(int(item) in dataset_shape for item in matches):
            pass
        else:
            raise ValueError(f"{weight_table} dimensions don't match the input dataset shape: {dataset_shape}")
    else:
        raise ValueError(f"Could not validate the grid shape in {weight_table} filename")

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
    
    # Forecast may not be in 3 hr timesteps. Check if this is so, and if 'cumulative' is True, convert all to 3 hr timesteps
    time_diff = np.diff(datetime_array)
    expected_time_step = datetime_array[1] - datetime_array[0]

    if not np.all(time_diff == expected_time_step) and not cumulative: 
        logging.warning("Input datasets do NOT have consistent time steps!")

    # Forecast data is cumulative. Check if this is so and warn
    is_cumulative = _is_cumulative(inflow_array[:,-1])
    if is_cumulative and not cumulative:
        logging.warning("Input datasets are cumulative and you are not fixing them")

    if cumulative:
        if is_cumulative:
            # IMPORTANT: If data is cumulative, we fix this here
            diff_array = np.diff(inflow_array, axis=0)
            inflow_array = np.vstack((inflow_array[0,:], diff_array))

        # Interpolate data to fit 3 hr timesteps
        interp_array = time_diff // DESIRED_TIME_STEP
        datetime_array = np.arange(datetime_array[0], datetime_array[-1] + DESIRED_TIME_STEP, DESIRED_TIME_STEP)
        new_array = np.empty((datetime_array.shape[0], inflow_array.shape[1]))

        for i in range(inflow_array.shape[0] - 1):
            step_values = inflow_array[i, :] / interp_array[i]
            start_idx = sum(interp_array[:i])
            end_idx = start_idx + interp_array[i]
            new_array[start_idx:end_idx, :] = step_values

        # Copy the last row from the original array
        new_array[-1, :] = inflow_array[-1, :]

        # Update inflow_array
        inflow_array = None
        inflow_array = new_array.astype(np.float64)
        new_array = None

    inflow_array = np.nan_to_num(inflow_array, nan=0)
    inflow_array[inflow_array < 0] = 0
    inflow_array = inflow_array * weight_df['area_sqm'].values * conversion_factor
    inflow_array = pd.DataFrame(inflow_array, columns=stream_ids)
    inflow_array = inflow_array.groupby(by=stream_ids, axis=1).sum()
    inflow_array = inflow_array[sorted_rivid_array].to_numpy()

    ds.close()

    # Create output inflow netcdf data
    logging.info("Writing inflows to file")
    os.makedirs(os.path.join(inflows_dir, vpu_name), exist_ok=True)
    start_date = datetime.datetime.utcfromtimestamp(datetime_array[0].astype(float) / 1e9).strftime('%Y%m%d')
    end_date = datetime.datetime.utcfromtimestamp(datetime_array[-1].astype(float) / 1e9).strftime('%Y%m%d')
    inflow_file_path = os.path.join(inflows_dir,
                                    vpu_name,
                                    f'm3_{os.path.basename(inflows_dir)}_{start_date}_{end_date}.nc')

    with nc.Dataset(inflow_file_path, "w", format="NETCDF3_CLASSIC") as inflow_nc:
        # create dimensions
        inflow_nc.createDimension('time', datetime_array.shape[0])
        inflow_nc.createDimension('rivid', sorted_rivid_array.shape[0])
        inflow_nc.createDimension('nv', 2)

        # m3_riv
        # note to Riley: setting a fill value is not be a problem with netcdf4+. Howver, since we are saving with netcdf3, this creates masked arrays where 0 is 
        # nan, causing RAPID to freak out. By not setting a fill vlaue, it doesn't create masked arrays with nan values
        m3_riv_var = inflow_nc.createVariable('m3_riv', 'f4', ('time', 'rivid'), zlib=True, complevel=7)
        m3_riv_var[:] = inflow_array
        m3_riv_var.long_name = 'accumulated inflow inflow volume in river reach boundaries'
        m3_riv_var.units = 'm3'
        m3_riv_var.coordinates = 'lon lat'
        m3_riv_var.grid_mapping = 'crs'
        m3_riv_var.cell_methods = "time: sum"

        # rivid
        rivid_var = inflow_nc.createVariable('rivid', 'i4', ('rivid',), zlib=True, complevel=7)
        rivid_var[:] = sorted_rivid_array
        rivid_var.long_name = 'unique identifier for each river reach'
        rivid_var.units = '1'
        rivid_var.cf_role = 'timeseries_id'

        # time
        reference_time = datetime_array[0]
        time_step = (datetime_array[1] - datetime_array[0]).astype('timedelta64[s]')
        time_var = inflow_nc.createVariable('time', 'i4', ('time',), zlib=True, complevel=7)
        time_var[:] = (datetime_array - reference_time).astype('timedelta64[s]').astype(int)
        time_var.long_name = 'time'
        time_var.standard_name = 'time'
        time_var.units = f'seconds since {reference_time.astype("datetime64[s]")}'  # Must be seconds
        time_var.axis = 'T'
        time_var.calendar = 'gregorian'
        time_var.bounds = 'time_bnds'

        # time_bnds
        time_bnds = inflow_nc.createVariable('time_bnds', 'i4', ('time', 'nv',), zlib=True, complevel=7)
        time_bnds_array = np.stack([datetime_array, datetime_array + time_step], axis=1)
        time_bnds_array = (time_bnds_array - reference_time).astype('timedelta64[s]').astype(int)
        time_bnds[:] = time_bnds_array

        # longitude
        lon_var = inflow_nc.createVariable('lon', 'f8', ('rivid',), zlib=True, complevel=7)
        lon_var[:] = comid_df['lon'].values
        lon_var.long_name = 'longitude of a point related to each river reach'
        lon_var.standard_name = 'longitude'
        lon_var.units = 'degrees_east'
        lon_var.axis = 'X'

        # latitude
        lat_var = inflow_nc.createVariable('lat', 'f8', ('rivid',), zlib=True, complevel=7)
        lat_var[:] = comid_df['lat'].values
        lat_var.long_name = 'latitude of a point related to each river reach'
        lat_var.standard_name = 'latitude'
        lat_var.units = 'degrees_north'
        lat_var.axis = 'Y'

        # crs
        crs_var = inflow_nc.createVariable('crs', 'i4', zlib=True, complevel=7)
        crs_var.grid_mapping_name = 'latitude_longitude'
        crs_var.epsg_code = 'EPSG:4326'  # WGS 84
        crs_var.semi_major_axis = 6378137.0
        crs_var.inverse_flattening = 298.257223563

        # add global attributes
        inflow_nc.Conventions = 'CF-1.6'
        inflow_nc.history = 'date_created: {0}'.format(datetime.datetime.utcnow())
        inflow_nc.featureType = 'timeSeries'
        inflow_nc.geospatial_lat_min = min_lat
        inflow_nc.geospatial_lat_max = max_lat
        inflow_nc.geospatial_lon_min = min_lon
        inflow_nc.geospatial_lon_max = max_lon

    lsm_dataset.close()
    return


def main():
    parser = argparse.ArgumentParser(description='Create inflow file for LSM files and input directory.')

    # Define the command-line argument
    parser.add_argument('--lsmdir', type=str, help='Directory of LSM files')
    parser.add_argument('--inputdir', type=str, help='Inputs directory')
    parser.add_argument('--inflowdir', type=str, help='Inflows directory')

    args = parser.parse_args()

    # Access the parsed argument
    lsm_dir = args.lsmdir
    input_dir = args.inputdir
    inflows_dir = args.inflowdir

    if not all([lsm_dir, input_dir, inflows_dir]):
        raise ValueError('Missing required arguments')

    # Create the inflow file for each LSM file
    input_dir_name = os.path.basename(input_dir)
    create_inflow_file(lsm_dir,
                       input_dir_name,
                       inflows_dir,
                       os.path.join(input_dir, 'weight_era5_721x1440.csv'),
                       os.path.join(input_dir, 'comid_lat_lon_z.csv'), 
                       False,)


if __name__ == '__main__':
    main()
