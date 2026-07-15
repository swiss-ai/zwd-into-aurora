"""
Compute 6-hour accumulated precipitation from 3-hourly MSWEP data.

For each timestep t: acc_6h(t) = tp(t-3h) + tp(t)

This creates a new zarr with the same 3h temporal resolution as the input,
but with 6h accumulated values. The first timestep is NaN (no previous 3h value).

Strategy:
- First month: written via xarray to_zarr(mode="w") to set up zarr structure and metadata.
- Subsequent months: appended via xarray to_zarr(append_dim="time").

Usage:
    python compute_mswep_6h_accumulation.py
"""

import gc
import numpy as np
import pandas as pd
import xarray as xr
import zarr

INPUT_PATH = "/path/to/data/MSWEP-v280-720x1440-3h_new.zarr"
OUTPUT_PATH = "/path/to/data/MSWEP-v280-720x1440-6h_acc_3h_sampling.zarr"
VAR_NAME = "total_precipitation_MSWEP"

print(f"Opening {INPUT_PATH} ...", flush=True)
ds = xr.open_zarr(INPUT_PATH, chunks=None)

print(f"Input shape: {ds[VAR_NAME].shape}", flush=True)
print(f"Time range: {ds.time.values[0]} → {ds.time.values[-1]}", flush=True)

times = pd.DatetimeIndex(ds.time.values)
year_months = sorted(set(zip(times.year, times.month)))
print(f"Processing {len(year_months)} months ...", flush=True)

prev_last = None   # last spatial slice (lat, lon) of previous month
write_pos = 0      # current write position in output zarr

# Skip first 370 months (~1979-2009) since training data starts at 2010-01-01
for i, (year, month) in enumerate(year_months[370:]):
    print(f"  [{i+1}/{len(year_months)}] {year}-{month:02d}", flush=True)

    ds_month = ds.sel(time=f"{year}-{month:02d}").load()
    arr = ds_month[VAR_NAME].values  # (T, lat, lon), float32

    acc = np.empty_like(arr)
    acc[0] = np.nan if prev_last is None else prev_last + arr[0]
    acc[1:] = arr[:-1] + arr[1:]

    n = arr.shape[0]

    # Build output dataset for this month
    ds_out = ds_month.copy(deep=False)
    ds_out[VAR_NAME] = xr.DataArray(acc, dims=ds_month[VAR_NAME].dims,
                                    coords=ds_month[VAR_NAME].coords)

    if i == 0:
        # First month: create zarr store with full metadata
        ds_out.to_zarr(OUTPUT_PATH, mode="w", zarr_format=2)
    else:
        # Subsequent months: append along time dimension
        ds_out.to_zarr(OUTPUT_PATH, append_dim="time", zarr_format=2)

    prev_last = arr[-1].copy()
    write_pos += n

    # Verify time encoding is correct by round-tripping through xarray
    if i > 0 and i % 5 == 0:
        # Read back what was just written and decode via xarray.
        # consolidated=False forces reading live .zarray metadata instead of the
        # stale .zmetadata written by the first to_zarr() call.
        decoded = xr.open_zarr(OUTPUT_PATH, chunks=None, consolidated=False).time.values[write_pos - n : write_pos]
        expected = ds_month.time.values.astype("datetime64[ns]")
        assert decoded.shape == expected.shape, (
            f"Time shape mismatch in {year}-{month:02d}: "
            f"got {decoded.shape} vs expected {expected.shape}"
        )
        assert np.array_equal(decoded, expected), (
            f"Time encoding mismatch in {year}-{month:02d}: "
            f"first={decoded[0]} (expected {expected[0]}), "
            f"last={decoded[-1]} (expected {expected[-1]})"
        )

    del ds_month, arr, acc
    gc.collect()

zarr.consolidate_metadata(OUTPUT_PATH)
print("Done.", flush=True)
