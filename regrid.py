import os
import subprocess as sp
from os.path import sep
import pandas as pd
import xarray as xr
import numpy as np
import multiprocessing as mp
import functools as ft


def get_nc_files(dir):
    files = os.listdir(dir)
    files = filter(lambda f: f.endswith(".nc"), files)
    return list(files)


def create_grid_file(file, xsize, ysize):
    # These should be the same for each file
    LON_MIN = -124.93800354003906
    LON_MAX = -67.06300354003906
    LAT_MIN = 25.062999725341797
    LAT_MAX = 52.9379997253418

    grid_text = f"""gridtype  = lonlat
gridsize  = {xsize * ysize}
datatype  = float
xsize     = {xsize}
ysize     = {ysize}
xname     = {LON_KEY}
xlongname = "longitude"
xunits    = "degrees_east"
yname     = {LAT_KEY}
ylongname = "latitude"
yunits    = "degrees_north"
xfirst    = {LON_MIN}
xinc      = {abs(LON_MAX - LON_MIN) / xsize}
yfirst    = {LAT_MIN}
yinc      = {abs(LAT_MAX - LAT_MIN) / ysize}"""
    with open(file, "w") as f:
        f.write(grid_text)


def regrid_file(grid_file, input, output):
    sp.run(f" cdo -remapdis,{grid_file} {input} {output}".split(), check=True)


def get_file_date(file):
    """Get date of file as string in mm/dd/yyyy format (leading zeroes omitted)"""
    raw_date = os.path.basename(file).split(".")[1][1:]
    # Conversion to int strips leading zeros
    month = int(raw_date[4:6])
    day = int(raw_date[-2:])
    year = raw_date[:4]
    return f"{month}/{day}/{year}"


def get_cell_boundaries(lat, lat_step, lon, lon_step):
    lat1 = lat - (lat_step / 2)
    lat2 = lat + (lat_step / 2)
    lon1 = lon - (lon_step / 2)
    lon2 = lon + (lon_step / 2)
    return (lat1, lat2, lon1, lon2)


def get_cells(dataset):
    """Return boundaries of grid cells in lat1, lat2, lon1, lon2 format"""
    lat = dataset.coords[LAT_KEY].values
    lon = dataset.coords[LON_KEY].values
    lat_step = abs(lat[1] - lat[0])
    lon_step = abs(lon[1] - lon[0])
    cross = ((la, lo) for la in lat for lo in lon)
    return (get_cell_boundaries(la, lat_step, lo, lon_step) for la, lo in cross)


def point_in_box(p_lat, p_lon, lat1, lat2, lon1, lon2):
    return (lat1 < p_lat < lat2) and (lon1 < p_lon < lon2)


def get_acres_burned(dataset, lat1, lat2, lon1, lon2):
    ACRES_KEY = "acres"
    LAT = "lat"
    LON = "long"
    # The number of fires per day is usually less than 10. Manual iteration on
    # each cell takes ~1 second. Query on each cell takes ~2 minutes
    fires_in_cell = (
        row[ACRES_KEY]
        for _, row in dataset.iterrows()
        if point_in_box(row[LAT], row[LON], lat1, lat2, lon1, lon2)
    )
    return sum(fires_in_cell)


def regrid_file_and_embed_data(dataset, input, output):
    regrid_file(GRID_FILE, input, output)

    date = get_file_date(input)
    wildfires_on_date = dataset.query("date == @date")
    ds = xr.load_dataset(output)
    cells = get_cells(ds)
    # It's easier to modify a flat array then reshape
    acres = np.zeros(YSIZE * XSIZE, dtype='float32')
    for i, (lat1, lat2, lon1, lon2) in enumerate(cells):
        acres[i] = get_acres_burned(wildfires_on_date, lat1, lat2, lon1, lon2)
    acres = acres.reshape((YSIZE, XSIZE))

    ds = ds.assign(acres_burned=((LAT_KEY, LON_KEY), acres))
    # ds = ds.assign(acres_burned=lambda x: debugging(x.lat_110, x.lon_110))
    # print(ds)
    ds.to_netcdf(output)


if __name__ == "__main__":
    DATA_DIR = "data"
    NLDAS_DIR = f"{DATA_DIR}{sep}NLDAS-2"
    OUTPUT_DATA_DIR = f"{NLDAS_DIR}{sep}regridded"
    GRID_FILE = f"{OUTPUT_DATA_DIR}{sep}grid.txt"
    WILDFIRE_FILE = f"{DATA_DIR}{sep}wildfire_and_index.csv"
    LAT_KEY = "lat_110"
    LON_KEY = "lon_110"
    XSIZE = 128
    YSIZE = XSIZE

    if not os.path.exists(OUTPUT_DATA_DIR):
        os.mkdir(OUTPUT_DATA_DIR)

    input_files = get_nc_files(NLDAS_DIR)
    input_file_paths = map(lambda f: f"{NLDAS_DIR}{sep}{f}", input_files)
    output_file_paths = map(
        lambda f: f"{OUTPUT_DATA_DIR}{sep}{XSIZE}x{YSIZE}-{f}", input_files
    )

    create_grid_file(GRID_FILE, XSIZE, YSIZE)
    wildfire_data = pd.read_csv(WILDFIRE_FILE)
    with mp.Pool(processes=mp.cpu_count()) as p:
        regridder = ft.partial(regrid_file_and_embed_data, wildfire_data)
        # for input, output in zip(input_file_paths, output_file_paths):
        # regridder(input, output)
        results = p.starmap(regridder, zip(input_file_paths, output_file_paths))
