"""
We are testing the first 10 days (1980-1-1 to 1980-1-10) and the last 10 rivers in the order of the comid_lat_lon_z. The input data is that subselection
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
import netCDF4 as nc
import sys
import os

# Add the project_root directory to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

from rapid_inflows.inflow_fast import create_inflow_file

create_inflow_file(glob.glob('./tests/inputs/era5_721x1440_sample_data/*.nc'),'./tests/inputs/weight_era5_721x1440_last_10.csv','./tests/inputs/comid_lat_lon_z_last_10.csv','./tests/test.nc')

out_ds = nc.Dataset('./tests/test.nc', 'r')
val_ds = nc.Dataset('tests/validation/1980_01_01to10_last10.nc', 'r')

try:
    # Check dimensions match
    assert out_ds.dimensions.keys() == val_ds.dimensions.keys(), "Dimensions do not match."
    for key in out_ds.dimensions.keys():
        if key == 'nv':
            continue
        assert (out_ds[key][:] == val_ds[key][:]).all(), f"{key} values differ"
    
    # Check m3 values match
    assert (out_ds['m3_riv'][:] == val_ds['m3_riv'][:]).all(), "m3 values do not match."
    
    # Check time bounds match
    assert (out_ds['time_bnds'][:] == val_ds['time_bnds'][:]).all(), "time bounds do not match."
    
    # Check lon match
    assert (out_ds['lon'][:] == val_ds['lon'][:]).all(), "lon values do not match."
    
    # Check lat match
    assert (out_ds['lat'][:] == val_ds['lat'][:]).all(), "lat values do not match."
    
    # Check CRS is EPSG 4326
    assert out_ds['crs'].epsg_code == val_ds['crs'].epsg_code, "CRS is not EPSG 4326."
    
    print("All tests passed.")
    
except AssertionError as e:
    print(f"Test failed: {e}")

finally:
    # Close the datasets
    out_ds.close()
    val_ds.close()



