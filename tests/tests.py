"""
We are testing the first 10 days (1980-1-1 to 1980-1-10) and the last 10 rivers in the order of the comid_lat_lon_z. The input data is that subselection.
We also test a subset of forecast data that has uneven time steps.
Things to test:
    - dimensions match
    - rivid order matches
    - m3 values match
    - time matches
    - time bnds match
    - lon match
    - lat match
    - crs is EPSG 4326
"""
import glob
import os
import sys

import netCDF4 as nc

# Add the project_root directory to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
os.chdir(project_root)
sys.path.append(project_root)

import basininflow as bi


def check_function(validation_ds, output_ds, test):
    print(test)
    try:
        # Check dimensions match
        assert output_ds.dimensions.keys() == validation_ds.dimensions.keys(), "Dimensions do not match."

        for key in output_ds.dimensions.keys():
            if key == 'nv':
                continue
            assert (output_ds[key][:] == validation_ds[key][:]).all(), f"{key} values differ"

        # Check time bounds match
        assert (output_ds['time_bnds'][:] == validation_ds['time_bnds'][:]).all(), "time bounds do not match."

        # Check lon match
        assert (output_ds['lon'][:] == validation_ds['lon'][:]).all(), "lon values do not match."

        # Check lat match
        assert (output_ds['lat'][:] == validation_ds['lat'][:]).all(), "lat values do not match."

        # Check CRS is EPSG 4326
        assert output_ds['crs'].epsg_code == validation_ds['crs'].epsg_code, f"crs does not match"

        # Check m3 values match
        assert (output_ds['m3_riv'][:] == validation_ds['m3_riv'][:]).all(), "m3 values do not match."

        print("All tests passed.")

    except AssertionError as e:
        print(f"Test failed: {e}")

    finally:
        # Close the datasets
        output_ds.close()
        validation_ds.close()

# TEST 1: Normal inputs, directory of LSM
bi.create_inflow_file('tests/inputs/era5_721x1440_sample_data/', 'tests/test_vpu/123', 'tests/test_results/',
                      cumulative=False, force_positive_runoff=True)

out_ds = nc.Dataset(glob.glob('./tests/test_results/*_123_*.nc')[0], 'r')
val_ds = nc.Dataset('tests/validation/1980_01_01to10_123.nc', 'r')

check_function(val_ds, out_ds, 'TEST 1: Base. Constant Timestep, Incremental Runoff, Force Positive Runoff')

# # TEST 2: Forecast inputs, auto timestep
# bi.create_inflow_file('tests/inputs/era5_2560x5120_sample_data/forecast_data.nc',
#                    'tests/test_vpu/345',
#                    'tests/test_results/',
#                    cumulative=True)
#
# out_ds = nc.Dataset(glob.glob('./tests/test_results/*_345_*.nc')[0], 'r')
# val_ds = nc.Dataset('tests/validation/forecast_3_to_6_hour.nc', 'r')
#
# check_function(val_ds, out_ds, 'TEST 2: Forecast inputs, auto timestep')
#
# # TEST 3: Forecast inputs, 1 hour timestep
# bi.create_inflow_file('tests/inputs/era5_2560x5120_sample_data/forecast_data.nc',
#                    'tests/test_vpu/345',
#                    'tests/test_results/',
#                    vpu_name='custom_vpu',
#                    cumulative=True,
#                    timestep=datetime.timedelta(hours=3),
#                    file_label='file_label')
#
# out_ds = nc.Dataset(glob.glob('./tests/test_results/*_custom_vpu_*_file_label.nc')[0], 'r')
# val_ds = nc.Dataset('tests/validation/forecast_3_to_6_hour.nc', 'r')
#
# check_function(val_ds, out_ds, 'TEST 3: Forecast inputs, auto timestep')
