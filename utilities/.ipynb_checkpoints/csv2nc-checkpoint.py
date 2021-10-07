#Este programa toma los coeficientes de la regresion y hace de un .csv un .nc
import cartopy.crs as ccrs
import numpy as np
import cartopy.feature
from cartopy.util import add_cyclic_point
import matplotlib.path as mpath
import os
import glob
import pandas as pd
import xarray as xr
import netCDF4
import matplotlib
import matplotlib.pyplot as plt
#import psyplot.project as psy


#DataFrame to .nc----------------------------------------------------------------------------
def csv_to_nc(path):
    all_files = glob.glob(os.path.join(path, "*.csv"))
    for filename in all_files:
        namecsv = glob.glob(os.path.join(path,"*.csv"))
        colnames = ['coef','lat','lon']

    for filename in all_files:
        namecsv = os.path.splitext(os.path.basename(filename))[0]
        df = pd.read_csv(filename, header=0, error_bad_lines=False, names = colnames, sep=',')
        df = df.set_index(['lat','lon'])
        df = df[~df.index.duplicated(keep='first')]
        xr = df.to_xarray()

        # add variable attribute metadata
        #xr['time'].attrs={'units':'hours since 2018-01-01'}
        xr['lat'].attrs={'units':'degrees', 'long_name':'Latitude'}
        xr['lon'].attrs={'units':'degrees', 'long_name':'Longitude'}
        xr['coef'].attrs={'units':'coef', 'long_name':'coef'}


        # Save NetCDF file
        xr.to_netcdf(path+'/'+ namecsv + '.nc')
        del namecsv, df, xr
