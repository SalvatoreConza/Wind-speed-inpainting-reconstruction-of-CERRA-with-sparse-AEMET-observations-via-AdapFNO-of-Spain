import xarray as xr
import cfgrib

ds = xr.open_dataset('data/3d/era5/wind/20240601.grib', engine='cfgrib')



