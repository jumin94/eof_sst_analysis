#funciones.py

import numpy as np
import os
import glob
import pandas as pd
import xarray as xr
import os, fnmatch
import matplotlib.pyplot as plt
# Subplot number three for mean changes and other figures
import cartopy.crs as ccrs
import cartopy.feature
from cartopy.util import add_cyclic_point
import matplotlib.path as mpath
import netCDF4
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib as mpl
mpl.rcParams['hatch.linewidth'] = 0.5  # previous pdf hatch linewidth
import cartopy.util as cutil


def split_title_line(title_text, split_on='(', max_words=4):  # , max_words=None):
    """
    A function that splits any string based on specific character
    (returning it with the string), with maximum number of words on it
    """
    split_at = title_text.find (split_on)
    ti = title_text
    if split_at > 1:
        ti = ti.split (split_on)
        for i, tx in enumerate (ti[1:]):
            ti[i + 1] = split_on + tx
    if type (ti) == type ('text'):
        ti = [ti]
    for j, td in enumerate (ti):
        if td.find (split_on) > 0:
            pass
        else:
            tw = td.split ()
            t2 = []
            for i in range (0, len (tw), max_words):
                t2.append (' '.join (tw[i:max_words + i]))
            ti[j] = t2
    ti = [item for sublist in ti for item in sublist]
    ret_tex = []
    for j in range (len (ti)):
        for i in range(0, len(ti)-1, 2):
            if len (ti[i].split()) + len (ti[i+1].split ()) <= max_words:
                mrg = " ".join ([ti[i], ti[i+1]])
                ti = [mrg] + ti[2:]
                break

    if len (ti[-2].split ()) + len (ti[-1].split ()) <= max_words:
        mrg = " ".join ([ti[-2], ti[-1]])
        ti = ti[:-2] + [mrg]
    return '\n'.join (ti)

class my_dictionary(dict):
    # __init__ function 
    def __init__(self):
        self = dict()
    # Function to add key:value 
    def add(self, key, value):
        self[key] = value

def cargo_todo(scenarios,models,ruta,var):
    os.chdir(ruta)
    os.getcwd()
    dic = {}
    dic['historical'] = {}
    dic['ssp585'] = {}
    for scenario in dic.keys():
        listOfFiles = os.listdir(ruta+'/'+scenario+'/'+var)
        for model in models:
            dic[scenario][model] = {}
            if scenario == 'ssp585':
                periods = ['2070-2099']
            else:
                periods = ['1940-1969']
            for period in periods:
                dic[scenario][model][period] = []
                pattern = "*"+model+"*"+scenario+"*"+period+"*T42*"
                for entry in listOfFiles:
                    if fnmatch.fnmatch(entry,pattern):
                        dato = xr.open_dataset(ruta+'/'+scenario+'/'+var+'/'+entry)
                        dic[scenario][model][period].append(dato)
    return dic

def cargo_todo_crudos(scenarios,models,ruta,var):
    os.chdir(ruta)
    os.getcwd()
    dic = {}
    dic['historical'] = {}
    dic['ssp585'] = {}
    for scenario in dic.keys():
        listOfFiles = os.listdir(ruta+'/'+scenario+'/'+var)
        for model in models:
            dic[scenario][model] = {}
            if scenario == 'ssp585':
                periods = ['201501-209912']
            else:
                periods = ['195001-201412']
            for period in periods:
                dic[scenario][model][period] = []
                pattern = "*"+model+"*"+scenario+"*"
                for entry in listOfFiles:
                    if fnmatch.fnmatch(entry,pattern):
                        dato = xr.open_dataset(ruta+'/'+scenario+'/'+var+'/'+entry)
                        dic[scenario][model][period].append(dato)
    return dic


#Media del ensamble y desviacion------------------
def estadisticos(modelos,SST):
    mean = 0; mean_zonal = 0; mean_asym = 0
    dif = 0; dif_zonal = 0; dif_asym = 0
    for model in modelos:
        FULL = SST[model][0]; GW = FULL.mean(dim='lon').mean(dim='lat').values
        base = FULL/FULL
        ZONAL =  base * FULL.mean(dim='lon');  ASYM = (FULL - ZONAL)/GW
        mean = mean + FULL/GW; mean_zonal = mean_zonal + ZONAL/GW
        mean_asym = mean_asym + ASYM
        
    ensamble_means = [mean/len(modelos), mean_zonal/len(modelos), mean_asym/len(modelos)]
    for model in modelos:
        FULL = SST[model][0]; GW = FULL.mean(dim='lon').mean(dim='lat').values
        base = FULL/FULL
        ZONAL =  base * FULL.mean(dim='lon');  ASYM = (FULL - ZONAL)/GW
        dif = dif + (FULL/GW - ensamble_means[0])**2
        dif_zonal = dif_zonal + (ZONAL/GW - ensamble_means[1])**2
        dif_asym = dif_asym + (ASYM - ensamble_means[2])**2
        
    ensamble_stds =  [np.sqrt(dif/(len(modelos)-1)),np.sqrt(dif_zonal/(len(modelos)-1)),np.sqrt(dif_asym/(len(modelos)-1))]
    return ensamble_means, ensamble_stds

def estadisticos_not_scaled(modelos,SST):
    mean = 0; mean_zonal = 0; mean_asym = 0
    dif = 0; dif_zonal = 0; dif_asym = 0
    for model in modelos:
        FULL = SST[model][0]; GW = FULL.mean(dim='lon').mean(dim='lat').values
        base = FULL/FULL
        ZONAL =  base * FULL.mean(dim='lon');  ASYM = (FULL - ZONAL)
        mean = mean + FULL ; mean_zonal = mean_zonal + ZONAL
        mean_asym = mean_asym + ASYM
        
    ensamble_means = [mean/len(modelos), mean_zonal/len(modelos), mean_asym/len(modelos)]
    for model in modelos:
        FULL = SST[model][0]; GW = FULL.mean(dim='lon').mean(dim='lat').values
        dif = dif + (FULL - ensamble_means[0])**2
        
    base = dif/dif
    std = np.sqrt(dif/(len(modelos)-1)); std_zonal = base * std.mean(dim='lon')
    std_asym = std - std_zonal
    ensamble_stds =  [std,std_zonal,std_asym]
    return ensamble_means, ensamble_stds


#Componente zonal y asimetrica
def components(model,SST):
    FULL = SST[model][0]
    GW = FULL.mean(dim='lon').mean(dim='lat').values
    base = FULL/FULL
    ZONAL = base * FULL.mean(dim='lon')
    ASYM = FULL/GW - ZONAL/GW
    return FULL/GW, ZONAL/GW, ASYM

def plot_one_sst(sst,title,lons,lats,box,levels=np.arange(-2,2,0.1),cmap = 'RdBu_r'):

    fig = plt.figure(figsize=(10, 10),dpi=300,constrained_layout=True)
    data_crs = ccrs.PlateCarree()
    proj = ccrs.PlateCarree(central_longitude=180)
    ax1 = plt.subplot(1,1,1,projection=proj)
    im1=ax1.contourf(lons, lats, sst,levels,transform=data_crs,cmap=cmap,extend='both')
    ax1.set_title(split_title_line(title, max_words=5),fontsize=14)
    ax1.add_feature(cartopy.feature.COASTLINE,alpha=.5)
    ax1.add_feature(cartopy.feature.BORDERS, linestyle='-', alpha=.5)
    ax1.gridlines(crs=data_crs, linewidth=0.3, linestyle='-')
    ax1.set_extent([-180, 180, -90, 90], ccrs.PlateCarree(central_longitude=180))
    ax1.gridlines(draw_labels=True, crs=ccrs.PlateCarree())

    x_start, x_end, y_start, y_end = box[0],box[1],box[2],box[3]
    margin = 0.07
    margin_fractions = np.array([margin, 1.0 - margin])
    x_lower, x_upper = x_start + (x_end - x_start)*margin_fractions
    y_lower, y_upper = y_start + (y_end - y_start)*margin_fractions
    box_x_points = x_lower + (x_upper - x_lower)* np.array([0, 1, 1, 0, 0,])
    box_y_points = y_lower + (y_upper - y_lower)* np.array([0, 0, 1, 1, 0,])
    ax1.plot(box_x_points, box_y_points, transform=ccrs.PlateCarree(),linewidth=1.5, color='black',linestyle='-')

    plt1_ax = plt.gca()
    left, bottom, width, height = plt1_ax.get_position().bounds
    colorbar_axes = fig.add_axes([left + 0.9, bottom,0.02, height])
    cbar = fig.colorbar(im1, colorbar_axes, orientation='vertical')
    cbar.set_label(r'K',fontsize=14) #rotation= radianes
    cbar.ax.tick_params(axis='both',labelsize=14)

    return fig


def weighted_av(dato):
    w_a = dato.mean(dim='lon')
    w_a = w_a.fillna(w_a[-1]-1)
    lats = np.cos(w_a.lat.values*np.pi/180.)
    s = sum(lats)
    return sum(w_a*lats)/s

def pattern_correlation_index(data1,data2):
    data1_mean = weigthed_av(data1); data1_std = np.std(data1.values)
    data2_mean = weigthed_av(data2); data2_std = np.std(data2.values)
    data1 = (data1 - data1_mean)/data1_std; data2 = (data2 - data2_mean)/data2_std
    suma1 = sum(data1.values*data2.values)

    return eof_pattern