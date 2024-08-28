import os
import time
from typing import Dict, List, Any
import datetime as dt

import pandas as pd
import xarray as xr
import cdsapi


class WindByPressureLevels:

    """
    Data Page: https://cds-beta.climate.copernicus.eu/datasets/reanalysis-era5-pressure-levels?tab=download
    Authentication Page: https://cds-beta.climate.copernicus.eu/how-to-api
    """

    def __init__(self, destination_dir: str):
        self.client = cdsapi.Client()
        self.destination_dir: str = destination_dir
        os.makedirs(name=f'{self.destination_dir}', exist_ok=True)

    def retrieve(self, years: List[int]):
        years: List[str] = [str(y) for y in years]
        
        request: Dict[str, Any] = {
            'product_type': ['reanalysis'],
            'variable': ['geopotential', 'relative_humidity', 'temperature', 'u_component_of_wind', 'v_component_of_wind'],
            'year': years,
            'month': [
                '01', '02', '03', '04', '05', '06', 
                '07', '08', '09', '10', '11', '12'
            ],
            'day': [
                '01', '02', '03', '04', '05', '06', 
                '07', '08', '09', '10', '11', '12', 
                '13', '14', '15', '16', '17', '18', 
                '19', '20', '21', '22', '23', '24', 
                '25', '26', '27', '28', '29', '30', 
                '31',
            ],
            'time': ['00:00', '06:00', '12:00', '18:00'],
            'pressure_level': ['50', '500', '850', '1000'],
            'data_format': 'grib',
            'download_format': 'unarchived'
        }

        filename: str = f"{'_'.join(years)}.grib"
        self.client.retrieve(
            name="reanalysis-era5-pressure-levels", request=request
        ).download(f'{self.destination_dir}/{filename}')
        return f'{self.destination_dir}/{filename}'



if __name__ == '__main__':

    self = WindByPressureLevels(destination_dir='data/era5')
    self.retrieve(years=[2024, 2023])

