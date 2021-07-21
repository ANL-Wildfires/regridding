import subprocess as sp
from os.path import sep
import os
import xarray as xr


def get_nc_files(dir):
    files = os.listdir(dir)
    files = filter(lambda f: f.endswith(".nc"), files)
    return list(files)


def get_bounding_box(ds):
    LON_KEY = "lon_110"
    LAT_KEY = "lat_110"
    lon_min = ds.coords[LON_KEY].Lo1
    lon_max = ds.coords[LON_KEY].Lo2
    lat_min = ds.coords[LAT_KEY].La1
    lat_max = ds.coords[LAT_KEY].La2
    return [lon_min, lon_max, lat_min, lat_max]


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
xname     = lon_110
xlongname = "longitude"
xunits    = "degrees_east"
yname     = lat_110
ylongname = "latitude"
yunits    = "degrees_north"
xfirst    = {LON_MIN}
xinc      = {abs(LON_MIN + LON_MAX) / xsize}
yfirst    = {LAT_MIN}
yinc      = {abs(LAT_MIN + LAT_MAX) / ysize}"""
    with open(file, "w") as f:
        f.write(grid_text)


def regrid_file(grid_file, input, output):
    sp.run(f"cdo -remapcon,{grid_file} {input} {output}".split(), check=True)


if __name__ == "__main__":
    DATA_DIR = f"data{sep}NLDAS-2"
    OUTPUT_DATA_DIR = DATA_DIR + f"{sep}regridded"
    GRID_FILE = OUTPUT_DATA_DIR + f"{sep}grid.txt"
    XSIZE = 128
    YSIZE = XSIZE

    if not os.path.exists(OUTPUT_DATA_DIR):
        os.mkdir(OUTPUT_DATA_DIR)

    input_file_names = get_nc_files(DATA_DIR)
    input_file_paths = list(map(lambda f: f"{DATA_DIR}{sep}{f}", input_file_names))
    output_file_paths = list(
        map(lambda f: f"{OUTPUT_DATA_DIR}{sep}{XSIZE}x{YSIZE}-{f}", input_file_names)
    )

    create_grid_file(GRID_FILE, XSIZE, YSIZE)
    for i, o in zip(input_file_paths, output_file_paths):
        regrid_file(GRID_FILE, i, o)
