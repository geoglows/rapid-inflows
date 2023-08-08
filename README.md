# rapid_inflows
Louis R. Rosas, Dr Riley Hales

Rapid Inflows is a python interface for RAPID that assists to prepare inflow data. Gridded runoff data is taken in, along with lists of basins and their area. The runoff for each basin is calculated for every time step, and the resulting flows are written to an output inflow NetCDF file, the main dataset used by RAPID. Inspired by ERDC's [RAPIDpy](https://github.com/erdc/RAPIDpy). More information about RAPID can be found [here](http://rapid-hub.org).

## Inputs
The `create_inflow_file` function takes in four parameters: 
    1) a list of gridded runoff netcdf datasets
    2) a weight table
    3) a comid_lat_lon_z file
    4) an output filename

### Gridded Runoff Datasets
The input netcdf datasets should follow the following conventions. It is expected that the datasets have 3 or 4 dimensions, in the following orders: `time, latitude, longitude` or `time, expver, latitude, longitude` (The 4th dimension is often present in the most recent releases of the ECMWF datasets, which has an 'experimental version' dimension. Internally, this dimension is flattened and the data summed. The name of this dimension does not matter). The runoff, longitude, and latitude variables may be any of the following, respectively: `'ro', 'RO', 'runoff', 'RUNOFF'`, `'lon', 'longitude', 'LONGITUDE', 'LON'`, `'lat', 'latitude', 'LATITUDE', 'LAT'`. Latitude and longitude is expected in even steps, from 90 to -90 degrees and 0 to 360 degrees repectively. Time is expected as `'time'`. The difference in time between each dataset should be equal (typically a time step of a day), and should be an interger.

The user may input as many datasets as desired. There is a built in memory check which warns the user if the amount of memory required exceeds 80% of the available RAM and will terminate the process if the amount of memory required exceeds all the available memory. 

Example netcdfs can be found in `/tests/inputs/era5_721x1440_sample_data`.

### Weight Table
A single csv file containing at least the following 5 columns of data with the following column names: 
    1) `'area_sqm'`, the area in square meters of a certain basin that intersects with the gridded runoff data. 
    2) `'lon'`,  longitude of the centroid of the gridded runoff data cell that intersects with a certain basin
    3) `'lat'`,  latitude of the centroid of the gridded runoff data cell that intersects with a certain basin
    4) `'lon_index'`,  x index of the gridded runoff data cell that intersects with a certain basin
    5) `'lat_index'`,  y index of the gridded runoff data cell that intersects with a certain basin

This file can be generated using either [these ArcGIS tools](https://github.com/Esri/python-toolbox-for-rapid) or [these python scripts](https://github.com/geoglows/tdxhydro-rapid).

An example weight table can be found in `/tests/inputs`.

### Comid csv
A csv which has the unique ids (COMIDs, reach ids, LINKNOs, etc.) of the basins/streams modeled in its first column, sorted in the order in which the user wants RAPID to process. Two other columns must be present, labeled `'lat'` and `'lon'`. These columns represent a point related to each river reach. This file can be generated using either [these ArcGIS tools](https://github.com/Esri/python-toolbox-for-rapid) or [these python scripts](https://github.com/geoglows/tdxhydro-rapid).

An example comid csv can be found in `/tests/inputs`.

## Outputs
The `create_inflow_file` outputs a new netcdf 3 (classic) dataset. It is desgined to be accepted by rapid and pass all of its internal tests. Example output datasets can be found in `/tests/validation`.