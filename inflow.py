import glob
import os
import datetime

import numpy as np
import pandas as pd
import xarray as xr

t1 = datetime.datetime.now()

LSM_DATA_DIR = '/Volumes/DrHalesT7/era5_1940_2022_daily_cumulative'
input_directory = '/Volumes/EB406_T7_2/GEOGLOWS2/rapid_files/718/'
vpu_code = 718

weight_table = os.path.join(input_directory, 'weight_era5_721x1440.csv')
weight_df = pd.read_csv(weight_table)

min_lon_idx = weight_df['lon_index'].min()
max_lon_idx = weight_df['lon_index'].max()
min_lat_idx = weight_df['lat_index'].min()
max_lat_idx = weight_df['lat_index'].max()

if min_lon_idx > max_lon_idx:
    min_lon_idx, max_lon_idx = max_lon_idx, min_lon_idx
if min_lat_idx > max_lat_idx:
    min_lat_idx, max_lat_idx = max_lat_idx, min_lat_idx

weight_df['lon_index'] -= min_lon_idx
weight_df['lat_index'] -= min_lat_idx

rivbasid = os.path.join(input_directory, 'riv_bas_id.csv')
rivid_df = pd.read_csv(rivbasid, header=None)
rivid_df = rivid_df.iloc[:, 0].astype(str).apply(lambda x: int(x[2:])).to_frame()

if rivid_df.shape[0] != weight_df.iloc[:, 0].unique().shape[0]:
    raise ValueError('Number of unique RIVID in weight table does not match number of unique RIVID in riv_bas_id.csv')

lsm_file_list = sorted(glob.glob(os.path.join(LSM_DATA_DIR, '*.nc')))
lsm_file_list = pd.Series(lsm_file_list)
lsm_dates = [datetime.datetime.strptime(os.path.basename(x).split('.')[0], "%Y%m%d") for x in lsm_file_list]
lsm_dates = np.array(lsm_dates)

for start_year in (1940, 1950, 1960, 1970, 1980, 1990, 2000, 2010):
    sim_start_datetime = datetime.datetime(start_year, 1, 1)
    sim_end_datetime = datetime.datetime(start_year + 9, 12, 31)

    inflow_df = pd.DataFrame(
        columns=rivid_df.values.flatten(),
        index=pd.date_range(start=sim_start_datetime, end=sim_end_datetime),
        dtype=np.float64
    )

    # read subset of data from disc
    ds = xr.open_mfdataset(
        lsm_file_list[np.logical_and(lsm_dates >= sim_start_datetime, lsm_dates <= sim_end_datetime)].values
    )
    ro = ds['ro'][:, min_lat_idx:max_lat_idx + 1, min_lon_idx:max_lon_idx + 1].values
    ds.close()

    for rivid in rivid_df.values.flatten():
        inflow_series = []
        for index, row in weight_df[weight_df.iloc[:, 0] == rivid].iterrows():
            inflow_series.append(
                ro[:, row['lat_index'].astype(int), row['lon_index'].astype(int)] * row['area_sqm']
            )
        inflow_series = np.array(inflow_series)
        if len(inflow_series) > 1:
            inflow_series = np.sum(inflow_series, axis=0)
        inflow_df.loc[:, rivid] = inflow_series

    pq_name = f'm3_{vpu_code}_{sim_start_datetime.strftime("%Y%m%d")}_{sim_end_datetime.strftime("%Y%m%d")}.parquet'
    inflow_df.to_parquet(os.path.join(input_directory, pq_name))

t2 = datetime.datetime.now()
print((t2 - t1).total_seconds())
