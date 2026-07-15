import xarray as xr
import numpy as np

## open and save soil_type var to a npy file.
fname_soil_type_era5 = '/path/to/data/ERA5_Soiltype.nc'
ds_soil_type = xr.open_dataset(fname_soil_type_era5)
st0 = ds_soil_type['slt'][0,...].values
stN = ds_soil_type['slt'][-1,...].values
assert np.all(st0 == stN) # just to make sure it is static
assert len(np.unique(st0)) == 8 ## 8 soil types
assert np.all(np.unique(st0) == np.array([0, 1, 2, 3, 4, 5, 6, 7]))
st0 = st0.astype(np.int8)
np.save('/path/to/data/ERA5_Soiltype.npy', st0)