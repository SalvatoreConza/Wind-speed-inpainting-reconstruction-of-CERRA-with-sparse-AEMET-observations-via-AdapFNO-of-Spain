import xarray as xr
import cfgrib

ds = xr.open_dataset('data/2D/era5/u10_t2m.grib', engine='cfgrib')



