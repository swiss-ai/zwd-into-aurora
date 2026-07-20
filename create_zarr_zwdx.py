from pathlib import Path
import xarray as xr
import os
from tqdm import tqdm
import pandas as pd
import numpy as np

dataset_path = os.path.join(os.environ.get("DATA_ROOT", "/path/to/data"), "ZWDX", "era5")
zwd_path = Path(dataset_path)

output_zarr_path = os.path.join(dataset_path, "zwd_data_1h_lead_time.zarr")

year_range = range(2010, 2025)  # Adjusted to include 2024, as the last year is exclusive in range

# First and last calendar years covered by the range
start_year = year_range.start                 # 2010
end_year   = year_range.stop - 1              # 2024

expected_times = pd.date_range(
    f"{start_year}-01-01 00:00",
    f"{end_year}-12-31 23:00",
    freq="1H"
)

valid_suffixes = ("00.zwd.nc", "01.zwd.nc", "02.zwd.nc", "03.zwd.nc",
                 "04.zwd.nc", "05.zwd.nc", "06.zwd.nc", "07.zwd.nc",
                 "08.zwd.nc", "09.zwd.nc", "10.zwd.nc", "11.zwd.nc",
                 "12.zwd.nc", "13.zwd.nc", "14.zwd.nc", "15.zwd.nc",
                 "16.zwd.nc", "17.zwd.nc", "18.zwd.nc", "19.zwd.nc",
                 "20.zwd.nc", "21.zwd.nc", "22.zwd.nc", "23.zwd.nc")
files = sorted([f for f in zwd_path.rglob("*.zwd.nc") if f.name.endswith(valid_suffixes)])



for i, yr in enumerate(year_range):
    print(f"\nProcessing year {yr}")
    # Filter files for the current year
    datasets = []
    all_times = []
    count_invalid = 0
    files_filtered_per_year = [f for f in files if f"{yr}" in f.name]
    print(f"num files at year {yr} = {len(files)}")
    print(f"Found {len(files_filtered_per_year)} files for year {yr}")
    for f in tqdm(files_filtered_per_year, desc="Checking files"):
        # Check if the file is valid and can be opened
        if not f.is_file():
            print(f"⚠️ Skipping {f} as it is not a file.")
            continue
        try:
            ds = xr.open_dataset(
                f,
                engine="h5netcdf",  # or "nested" if you use manual concat_dims
            )
            # ds.attrs  = {}
            times = ds.time.values
            if len(times) != 1:
                print(f"⚠️ File {f.name} has {len(times)} timestamps: {times}")
            all_times.append(ds.time.values[0])
            datasets.append(ds)
            ds.close()  # close to free resources
        except Exception as e:
            print(f"\nSkipping file {f} due to error:\n  {e}")
            count_invalid += 1

    # Save or inspect unique sorted times you actually gathered

    actual_times = np.sort(np.unique(all_times))
    print(f"number of invalid files for year {yr} = {count_invalid}")

    # Manually combine the datasets
    ds_all = xr.combine_by_coords(datasets, combine_attrs = "override")

    # Shift longitudes from [-180, 180] to [0, 360]
    if (ds_all.longitude < 0).any():
        ds_all = ds_all.assign_coords(longitude=(((ds_all.longitude + 360) % 360)))
        ds_all = ds_all.sortby("longitude")  # Ensure increasing order

    ds_all = ds_all.rename({"ZWD": "zenith_wet_delay"})

    ds_all.attrs = {}

    desired_chunks = {"time": 1, "latitude": 720, "longitude": 1440}

    # Determine mode and encoding
    if yr == 2010:
        mode = "w"
        encoding = {
            var: {
                "chunks": tuple(desired_chunks.get(dim, -1) for dim in ds_all[var].dims),
                "dtype": "float32"
            }
            for var in ds_all.data_vars
        }
    else:
        mode = "a"
        encoding = None
    print(f"Saving to {output_zarr_path} with encoding: {encoding}")

    ds_all.to_zarr(
        output_zarr_path,
        mode= mode,
        append_dim="time" if mode == "a" else None,
        consolidated=False,
        encoding=encoding,
        zarr_version=2
    )

import zarr 
zarr.consolidate_metadata(output_zarr_path)