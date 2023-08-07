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
from inflows.inflow_fast import create_inflow_file

create_inflow_file(glob.glob('./tests/inputs/era5_721x1440_sample_data/*.nc'),'./tests/inputs/weight_era5_721x1440_last_10.csv','./tests/inputs/comid_lat_lon_z_last_10.csv','./tests/test.nc')

# Open the NetCDF file using the Dataset class
with nc.Dataset('./tests/validation/1980_01_01to10_last10.nc', 'r') as validation_ds:
    with nc.Dataset('./tests/test.nc', 'r') as test_dataset:
        # Access the variables, attributes, and dimensions in the dataset
        # For example, to get the dimensions:
        dimensions = test_dataset.dimensions
        print("Dimensions:", dimensions)
        
        # To access a specific variable:
        variable_name = 'temperature'  # Replace this with the variable name you want to access
        variable = test_dataset.variables[variable_name]
        print("Variable:", variable)
        
        # To access a specific attribute of a variable:
        attribute_name = 'units'  # Replace this with the attribute name you want to access
        attribute_value = variable.getncattr(attribute_name)
        print("Attribute Value:", attribute_value)



