import datetime
import glob
import math
import os

import netCDF4 as nc
import numpy as np
import pandas as pd
import xarray as xr


def inflow_df_to_nc(df: pd.DataFrame, save_path: str):
    inflow_nc = nc.Dataset(save_path, 'w', format='NETCDF3_CLASSIC')
    inflow_nc.createDimension('time', df.shape[0])
    inflow_nc.createDimension('rivid', df.shape[1])

    inflow_nc.createVariable('time', 'i8', ('time',))
    inflow_nc.variables['time'][:] = np.array([(d - inflow_df.index[0]).days for d in inflow_df.index]).astype(int)
    inflow_nc['time'].units = f'days since {df.index[0].strftime("%Y-%m-%d %H:%M:%S")}'
    inflow_nc['time'].calendar = 'gregorian'
    inflow_nc['time'].axis = 'T'
    inflow_nc['time'].standard_name = 'time'
    inflow_nc['time'].long_name = 'time'

    inflow_nc.createVariable('rivid', 'i8', ('rivid',))
    inflow_nc.variables['rivid'][:] = df.columns.values.astype(np.int32)
    inflow_nc['rivid'].units = '1'
    inflow_nc['rivid'].cf_role = 'timeseries_id'
    inflow_nc['rivid'].long_name = 'unique identifier for each river segment'

    inflow_nc.createVariable('m3_riv', 'f8', ('time', 'rivid',))
    inflow_nc.variables['m3_riv'][:] = df.values.astype(np.float64)
    inflow_nc['m3_riv'].units = 'm3'

    inflow_nc.createVariable('lat', 'i2', ('rivid',))
    inflow_nc.variables['lat'][:] = np.zeros(df.shape[1]).astype(np.int16)
    inflow_nc['lat'].units = 'degrees_north'
    inflow_nc['lat'].standard_name = 'latitude'
    inflow_nc['lat'].long_name = 'latitude of river segment'
    inflow_nc['lat'].axis = 'Y'

    inflow_nc.createVariable('lon', 'i2', ('rivid',))
    inflow_nc.variables['lon'][:] = np.zeros(df.shape[1]).astype(np.int16)
    inflow_nc['lon'].units = 'degrees_east'
    inflow_nc['lon'].standard_name = 'longitude'
    inflow_nc['lon'].long_name = 'longitude of river segment'
    inflow_nc['lon'].axis = 'X'

    inflow_nc.sync()
    inflow_nc.close()
    return


LSM_DATA_DIR = '/Users/rchales/Data/era5_1940_2022_daily_cumulative'
input_directory = '/Volumes/EB406_T7_2/GEOGLOWS2/rapid_files/'
inflow_directory = '/Volumes/EB406_T7_2/GEOGLOWS2/inflows'
year_interval = 10
first_year = 1940
last_year = 2010
start_years = np.linspace(first_year,
                          last_year,
                          int(math.ceil((last_year - first_year) / year_interval)) + 1,
                          endpoint=True,
                          dtype=int)

lsm_files = sorted(glob.glob(os.path.join(LSM_DATA_DIR, '*.nc')))
lsm_files = pd.Series(lsm_files)
lsm_dates = [datetime.datetime.strptime(os.path.basename(x).split('.')[0], "%Y%m%d") for x in lsm_files]
lsm_dates = np.array(lsm_dates)

for start_year in start_years:
    # set date ranges
    sim_start_date = datetime.datetime(start_year, 1, 1)
    sim_end_date = datetime.datetime(start_year + year_interval - 1, 12, 31)
    print(f'Processing {sim_start_date.strftime("%Y-%m-%d")} to {sim_end_date.strftime("%Y-%m-%d")}')

    # read subset of data from disc
    ds = xr.open_mfdataset(
        lsm_files[np.logical_and(lsm_dates >= sim_start_date, lsm_dates <= sim_end_date)].values
    )
    # ro = dsrp['ro'][:, min_lat_idx:max_lat_idx + 1, min_lon_idx:max_lon_idx + 1].values
    ro = ds['ro'][:].values
    ro[ro < 0] = 0
    ro[ro == np.nan] = 0
    ds.close()

    for inp in sorted(glob.glob(os.path.join(input_directory, '*'))):
        vpu_code = int(os.path.basename(inp))
        print(f'Processing VPU {vpu_code}')
        t1 = datetime.datetime.now()

        if not os.path.exists(os.path.join(inflow_directory, str(vpu_code))):
            os.makedirs(os.path.join(inflow_directory, str(vpu_code)))

        weight_table = os.path.join(input_directory, str(vpu_code), 'weight_era5_721x1440.csv')
        weight_df = pd.read_csv(weight_table)

        rivbasid = os.path.join(input_directory, str(vpu_code), 'riv_bas_id.csv')
        rivid_df = pd.read_csv(rivbasid, header=None)
        rivid_df = rivid_df.iloc[:, 0].astype(str).apply(lambda x: int(x[2:])).to_frame()

        if rivid_df.shape[0] != weight_df.iloc[:, 0].unique().shape[0]:
            # raise ValueError('Number of rivids in weight table does not match number of rivids in riv_bas_id.csv')
            print('Number of rivids in weight table does not match number of rivids in riv_bas_id.csv')
            continue

        # determine output file names to check if it exists already
        # todo embed the timestep in the inflow file name and make everything else expect that
        nc_name = f'm3_{vpu_code}_{sim_start_date.strftime("%Y%m%d")}_{sim_end_date.strftime("%Y%m%d")}_daily.nc'
        nc_path = os.path.join(inflow_directory, str(vpu_code), nc_name)

        # check for already completed dataset for this time interval
        if os.path.exists(nc_path):
            print(f'Inflow file for decade {start_year} already exists')
            continue

        # create empty dataframe to populate
        inflow_df = pd.DataFrame(
            columns=rivid_df.values.flatten(),
            index=pd.date_range(start=sim_start_date, end=sim_end_date),
            dtype=np.float64
        )

        for rivid in inflow_df.columns:
            inflow_series = []
            for index, row in weight_df[weight_df.iloc[:, 0] == rivid].iterrows():
                inflow_series.append(
                    ro[:, row['lat_index'].astype(int), row['lon_index'].astype(int)] * row['area_sqm']
                )
            inflow_series = np.array(inflow_series)
            if len(inflow_series) > 1:
                inflow_series = np.nansum(inflow_series, axis=0)
            inflow_df.loc[:, rivid] = inflow_series

        inflow_df_to_nc(inflow_df, nc_path)

        t2 = datetime.datetime.now()
        print(round((t2 - t1).total_seconds() / 60, 3), 'minutes')
