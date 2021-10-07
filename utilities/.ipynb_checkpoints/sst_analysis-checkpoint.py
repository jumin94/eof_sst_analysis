# Grafico todos los patrones de SST de todos los modelos para tener una idea general
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
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib as mpl
mpl.rcParams['hatch.linewidth'] = 0.5  # previous pdf hatch linewidth
import cartopy.util as cutil
import logging
import utilities.funciones
import os, fnmatch

#Defino y genero diccionario con datos
#Clase diccionarios --------------------
class my_dictionary(dict):
    # __init__ function 
    def __init__(self):
        self = dict()
    # Function to add key:value 
    def add(self, key, value):
        self[key] = value

def changes_list(datos,scenarios,models):
    SST = {}
    for i in range(len(models)):
        print(models[i])
        tos_hist = datos[scenarios[0]][models[i]]['1940-1969'][0]
        tos_rcp5 = datos[scenarios[1]][models[i]]['2070-2099'][0]
        tos_h = tos_hist.tos
        tos_h.attrs = tos_hist.tos.attrs
        tos_rcp = tos_rcp5.tos
        tos_rcp.attrs = tos_rcp5.tos.attrs
        seasonal = tos_h.groupby('time.season').mean(dim='time')
        tosDJF = seasonal.sel(season='DJF')
        tosDJF.attrs = tos_h.attrs
        seasonal_r = tos_rcp.groupby('time.season').mean(dim='time')
        tosDJF_r = seasonal_r.sel(season='DJF')
        tosDJF_r.attrs = tos_rcp.attrs
        sst_change = tosDJF_r - tosDJF
        SST[models[i]] = []
        SST[models[i]].append(sst_change)

    return SST

def components(model,SST):
    FULL = SST[model][0]
    GW = FULL.mean(dim='lon').mean(dim='lat').values
    base = FULL/FULL
    ZONAL = base * FULL.mean(dim='lon')
    ASYM = FULL/GW - ZONAL/GW
    return FULL/GW, ZONAL/GW, ASYM

def plot_sst_box(model_sst,model_sym,model_asym,mod_name,lons,lats,levels,alevels,asym_cmap,box1,box2,box3):

    fig = plt.figure(figsize=(10, 8),dpi=300,constrained_layout=True)
    data_crs = ccrs.PlateCarree()
    proj = ccrs.PlateCarree(central_longitude=180)
    ax1 = plt.subplot(3,1,1,projection=proj)
    im1=ax1.contourf(lons, lats, model_sst,levels,transform=data_crs,cmap='OrRd',extend='both')
    ax1.set_title(funciones.split_title_line(r'a) Full SST change - '+str(mod_name), max_words=5),fontsize=8)
    ax1.add_feature(cartopy.feature.COASTLINE,alpha=.5)
    ax1.add_feature(cartopy.feature.BORDERS, linestyle='-', alpha=.5)
    ax1.gridlines(crs=data_crs, linewidth=0.3, linestyle='-')
    ax1.set_extent([-180, 180, -90, 90], ccrs.PlateCarree(central_longitude=180))
    ax1.gridlines(draw_labels=True, crs=ccrs.PlateCarree())

    plt1_ax = plt.gca()
    left, bottom, width, height = plt1_ax.get_position().bounds
    colorbar_axes = fig.add_axes([left + 0.4, bottom,0.02, height])
    cbar = fig.colorbar(im1, colorbar_axes, orientation='vertical')
    cbar.set_label(r'K/K',fontsize=14) #rotation= radianes
    cbar.ax.tick_params(axis='both',labelsize=14)

    ax2 = plt.subplot(3,1,2,projection=proj)
    im2=ax2.contourf(lons, lats, model_sym,levels,transform=data_crs,cmap='OrRd',extend='both')
    ax2.set_title(funciones.split_title_line(r'b) Zonal SST change - '+str(mod_name), max_words=5),fontsize=8)
    ax2.add_feature(cartopy.feature.COASTLINE,alpha=.5)
    ax2.add_feature(cartopy.feature.BORDERS, linestyle='-', alpha=.5)
    ax2.gridlines(crs=data_crs, linewidth=0.3, linestyle='-')
    ax2.set_extent([-180, 180, -90, 90], ccrs.PlateCarree(central_longitude=180))
    ax2.gridlines(draw_labels=True, crs=ccrs.PlateCarree())

    plt2_ax = plt.gca()
    left, bottom, width, height = plt2_ax.get_position().bounds
    colorbar_axes = fig.add_axes([left + 0.4, bottom,0.02, height])
    cbar = fig.colorbar(im2, colorbar_axes, orientation='vertical')
    cbar.set_label('K/K',fontsize=14)
    cbar.ax.tick_params(axis='both',labelsize=14)

    ax3 = plt.subplot(3,1,3,projection=proj)
    im3=ax3.contourf(lons, lats, model_asym,alevels,transform=data_crs,cmap=asym_cmap,extend='both')
    ax3.set_title(funciones.split_title_line(r'c) Asymmetric SST change - '+str(mod_name), max_words=5),fontsize=8)
    ax3.add_feature(cartopy.feature.COASTLINE,alpha=.5)
    ax3.add_feature(cartopy.feature.BORDERS, linestyle='-', alpha=.5)
    ax3.gridlines(crs=data_crs, linewidth=0.3, linestyle='-')
    ax3.set_extent([-180, 180, -90, 90], ccrs.PlateCarree(central_longitude=180))
    ax3.gridlines(draw_labels=True, crs=ccrs.PlateCarree())
    
    x_start, x_end, y_start, y_end = box1[0],box1[1],box1[2],box1[3]
    margin = 0.07
    margin_fractions = np.array([margin, 1.0 - margin])
    x_lower, x_upper = x_start + (x_end - x_start)*margin_fractions
    y_lower, y_upper = y_start + (y_end - y_start)*margin_fractions
    box_x_points = x_lower + (x_upper - x_lower)* np.array([0, 1, 1, 0, 0,])
    box_y_points = y_lower + (y_upper - y_lower)* np.array([0, 0, 1, 1, 0,])
    ax3.plot(box_x_points, box_y_points, transform=ccrs.PlateCarree(),linewidth=1.5, color='black',linestyle='-')
    ax2.plot(box_x_points, box_y_points, transform=ccrs.PlateCarree(),linewidth=1.5, color='black',linestyle='-')
    ax1.plot(box_x_points, box_y_points, transform=ccrs.PlateCarree(),linewidth=1.5, color='black',linestyle='-')

    x_start, x_end, y_start, y_end = box2[0],box2[1],box2[2],box2[3]
    margin = 0.07
    margin_fractions = np.array([margin, 1.0 - margin])
    x_lower, x_upper = x_start + (x_end - x_start)*margin_fractions
    y_lower, y_upper = y_start + (y_end - y_start)*margin_fractions
    box_x_points = x_lower + (x_upper - x_lower)* np.array([0, 1, 1, 0, 0,])
    box_y_points = y_lower + (y_upper - y_lower)* np.array([0, 0, 1, 1, 0,])
    ax3.plot(box_x_points, box_y_points, transform=ccrs.PlateCarree(),linewidth=1.5, color='black',linestyle='-')
    ax2.plot(box_x_points, box_y_points, transform=ccrs.PlateCarree(),linewidth=1.5, color='black',linestyle='-')
    ax1.plot(box_x_points, box_y_points, transform=ccrs.PlateCarree(),linewidth=1.5, color='black',linestyle='-')

    x_start, x_end, y_start, y_end = box3[0],box3[1],box3[2],box3[3]
    margin = 0.07
    margin_fractions = np.array([margin, 1.0 - margin])
    x_lower, x_upper = x_start + (x_end - x_start)*margin_fractions
    y_lower, y_upper = y_start + (y_end - y_start)*margin_fractions
    box_x_points = x_lower + (x_upper - x_lower)* np.array([0, 1, 1, 0, 0,])
    box_y_points = y_lower + (y_upper - y_lower)* np.array([0, 0, 1, 1, 0,])
    ax3.plot(box_x_points, box_y_points, transform=ccrs.PlateCarree(),linewidth=1.5, color='black',linestyle='-')
    ax2.plot(box_x_points, box_y_points, transform=ccrs.PlateCarree(),linewidth=1.5, color='black',linestyle='-')
    ax1.plot(box_x_points, box_y_points, transform=ccrs.PlateCarree(),linewidth=1.5, color='black',linestyle='-')

    plt3_ax = plt.gca()
    left, bottom, width, height = plt3_ax.get_position().bounds
    colorbar_axes = fig.add_axes([left + 0.4, bottom,0.02, height])
    cbar = fig.colorbar(im3, colorbar_axes, orientation='vertical')
    cbar.set_label(r'K/K',fontsize=14) #rotation = radianes
    cbar.ax.tick_params(axis='both',labelsize=14)
    
    fig.subplots_adjust(hspace=0.5)
    return fig


def plot_sst_non_scaled(model_sst,model_sym,model_asym,mod_name,lons,lats,levels,alevels,asym_cmap):

    fig = plt.figure(figsize=(10, 8),dpi=300,constrained_layout=True)
    data_crs = ccrs.PlateCarree()
    proj = ccrs.PlateCarree(central_longitude=180)
    ax1 = plt.subplot(3,1,1,projection=proj)
    im1=ax1.contourf(lons, lats, model_sst,levels,transform=data_crs,cmap='OrRd',extend='both')
    ax1.set_title(funciones.split_title_line(r'a) Full SST change - '+str(mod_name), max_words=5),fontsize=8)
    ax1.add_feature(cartopy.feature.COASTLINE,alpha=.5)
    ax1.add_feature(cartopy.feature.BORDERS, linestyle='-', alpha=.5)
    ax1.gridlines(crs=data_crs, linewidth=0.3, linestyle='-')
    ax1.set_extent([-180, 180, -90, 90], ccrs.PlateCarree(central_longitude=180))
    ax1.gridlines(draw_labels=True, crs=ccrs.PlateCarree())

    plt1_ax = plt.gca()
    left, bottom, width, height = plt1_ax.get_position().bounds
    colorbar_axes = fig.add_axes([left + 0.4, bottom,0.02, height])
    cbar = fig.colorbar(im1, colorbar_axes, orientation='vertical')
    cbar.set_label(r'K',fontsize=14) #rotation= radianes
    cbar.ax.tick_params(axis='both',labelsize=14)

    ax2 = plt.subplot(3,1,2,projection=proj)
    im2=ax2.contourf(lons, lats, model_sym,levels,transform=data_crs,cmap='OrRd',extend='both')
    ax2.set_title(funciones.split_title_line(r'b) Zonal SST change - '+str(mod_name), max_words=5),fontsize=8)
    ax2.add_feature(cartopy.feature.COASTLINE,alpha=.5)
    ax2.add_feature(cartopy.feature.BORDERS, linestyle='-', alpha=.5)
    ax2.gridlines(crs=data_crs, linewidth=0.3, linestyle='-')
    ax2.set_extent([-180, 180, -90, 90], ccrs.PlateCarree(central_longitude=180))
    ax2.gridlines(draw_labels=True, crs=ccrs.PlateCarree())

    plt2_ax = plt.gca()
    left, bottom, width, height = plt2_ax.get_position().bounds
    colorbar_axes = fig.add_axes([left + 0.4, bottom,0.02, height])
    cbar = fig.colorbar(im2, colorbar_axes, orientation='vertical')
    cbar.set_label('K',fontsize=14)
    cbar.ax.tick_params(axis='both',labelsize=14)

    ax3 = plt.subplot(3,1,3,projection=proj)
    im3=ax3.contourf(lons, lats, model_asym,alevels,transform=data_crs,cmap=asym_cmap,extend='both')
    ax3.set_title(funciones.split_title_line(r'c) Asymmetric SST change - '+str(mod_name), max_words=5),fontsize=8)
    ax3.add_feature(cartopy.feature.COASTLINE,alpha=.5)
    ax3.add_feature(cartopy.feature.BORDERS, linestyle='-', alpha=.5)
    ax3.gridlines(crs=data_crs, linewidth=0.3, linestyle='-')
    ax3.set_extent([-180, 180, -90, 90], ccrs.PlateCarree(central_longitude=180))
    ax3.gridlines(draw_labels=True, crs=ccrs.PlateCarree())

    plt3_ax = plt.gca()
    left, bottom, width, height = plt3_ax.get_position().bounds
    colorbar_axes = fig.add_axes([left + 0.4, bottom,0.02, height])
    cbar = fig.colorbar(im3, colorbar_axes, orientation='vertical')
    cbar.set_label(r'K',fontsize=14) #rotation = radianes
    cbar.ax.tick_params(axis='both',labelsize=14)
    
    fig.subplots_adjust(hspace=0.5)
    return fig


def plot_sst_non_scaled_box(model_sst,model_sym,model_asym,mod_name,lons,lats,levels,alevels,asym_cmap,box):

    fig = plt.figure(figsize=(10, 8),dpi=300,constrained_layout=True)
    data_crs = ccrs.PlateCarree()
    proj = ccrs.PlateCarree(central_longitude=180)
    ax1 = plt.subplot(3,1,1,projection=proj)
    im1=ax1.contourf(lons, lats, model_sst,levels,transform=data_crs,cmap='OrRd',extend='both')
    ax1.set_title(funciones.split_title_line(r'a) Full SST change - '+str(mod_name), max_words=5),fontsize=8)
    ax1.add_feature(cartopy.feature.COASTLINE,alpha=.5)
    ax1.add_feature(cartopy.feature.BORDERS, linestyle='-', alpha=.5)
    ax1.gridlines(crs=data_crs, linewidth=0.3, linestyle='-')
    ax1.set_extent([-180, 180, -90, 90], ccrs.PlateCarree(central_longitude=180))
    ax1.gridlines(draw_labels=True, crs=ccrs.PlateCarree())

    plt1_ax = plt.gca()
    left, bottom, width, height = plt1_ax.get_position().bounds
    colorbar_axes = fig.add_axes([left + 0.4, bottom,0.02, height])
    cbar = fig.colorbar(im1, colorbar_axes, orientation='vertical')
    cbar.set_label(r'K',fontsize=14) #rotation= radianes
    cbar.ax.tick_params(axis='both',labelsize=14)

    ax2 = plt.subplot(3,1,2,projection=proj)
    im2=ax2.contourf(lons, lats, model_sym,levels,transform=data_crs,cmap='OrRd',extend='both')
    ax2.set_title(funciones.split_title_line(r'b) Zonal SST change - '+str(mod_name), max_words=5),fontsize=8)
    ax2.add_feature(cartopy.feature.COASTLINE,alpha=.5)
    ax2.add_feature(cartopy.feature.BORDERS, linestyle='-', alpha=.5)
    ax2.gridlines(crs=data_crs, linewidth=0.3, linestyle='-')
    ax2.set_extent([-180, 180, -90, 90], ccrs.PlateCarree(central_longitude=180))
    ax2.gridlines(draw_labels=True, crs=ccrs.PlateCarree())

    plt2_ax = plt.gca()
    left, bottom, width, height = plt2_ax.get_position().bounds
    colorbar_axes = fig.add_axes([left + 0.4, bottom,0.02, height])
    cbar = fig.colorbar(im2, colorbar_axes, orientation='vertical')
    cbar.set_label('K',fontsize=14)
    cbar.ax.tick_params(axis='both',labelsize=14)

    ax3 = plt.subplot(3,1,3,projection=proj)
    im3=ax3.contourf(lons, lats, model_asym,alevels,transform=data_crs,cmap=asym_cmap,extend='both')
    ax3.set_title(funciones.split_title_line(r'c) Asymmetric SST change - '+str(mod_name), max_words=5),fontsize=8)
    ax3.add_feature(cartopy.feature.COASTLINE,alpha=.5)
    ax3.add_feature(cartopy.feature.BORDERS, linestyle='-', alpha=.5)
    ax3.gridlines(crs=data_crs, linewidth=0.3, linestyle='-')
    ax3.set_extent([-180, 180, -90, 90], ccrs.PlateCarree(central_longitude=180))
    ax3.gridlines(draw_labels=True, crs=ccrs.PlateCarree())
    
    x_start, x_end, y_start, y_end = box[0],box[1],box[2],box[3]
    margin = 0.07
    margin_fractions = np.array([margin, 1.0 - margin])
    x_lower, x_upper = x_start + (x_end - x_start)*margin_fractions
    y_lower, y_upper = y_start + (y_end - y_start)*margin_fractions
    box_x_points = x_lower + (x_upper - x_lower)* np.array([0, 1, 1, 0, 0,])
    box_y_points = y_lower + (y_upper - y_lower)* np.array([0, 0, 1, 1, 0,])
    ax3.plot(box_x_points, box_y_points, transform=ccrs.PlateCarree(),linewidth=1.5, color='black',linestyle='-')
    ax2.plot(box_x_points, box_y_points, transform=ccrs.PlateCarree(),linewidth=1.5, color='black',linestyle='-')
    ax1.plot(box_x_points, box_y_points, transform=ccrs.PlateCarree(),linewidth=1.5, color='black',linestyle='-')

    plt3_ax = plt.gca()
    left, bottom, width, height = plt3_ax.get_position().bounds
    colorbar_axes = fig.add_axes([left + 0.4, bottom,0.02, height])
    cbar = fig.colorbar(im3, colorbar_axes, orientation='vertical')
    cbar.set_label(r'K',fontsize=14) #rotation = radianes
    cbar.ax.tick_params(axis='both',labelsize=14)
    
    fig.subplots_adjust(hspace=0.5)
    return fig


import matplotlib.patches as mpatches

def plot_eofs(model_sst,model_sym,model_asym,mod_name,lons,lats,levels,alevels,colormap):

    fig = plt.figure(figsize=(10, 8),dpi=300,constrained_layout=True)
    data_crs = ccrs.PlateCarree()
    proj = ccrs.PlateCarree(central_longitude=180)
    ax1 = plt.subplot(3,1,1,projection=proj)
    im1=ax1.contourf(lons, lats, model_sst,levels,transform=data_crs,cmap=colormap,extend='both')
    ax1.set_title(funciones.split_title_line(r'a) EOF 1 - '+str(mod_name), max_words=5),fontsize=14)
    ax1.add_feature(cartopy.feature.COASTLINE,alpha=.5)
    ax1.add_feature(cartopy.feature.BORDERS, linestyle='-', alpha=.5)
    ax1.gridlines(crs=data_crs, linewidth=0.3, linestyle='-')
    ax1.set_extent([-180, 180, -90, 90], ccrs.PlateCarree(central_longitude=180))
    ax1.gridlines(draw_labels=True, crs=ccrs.PlateCarree())

    plt1_ax = plt.gca()
    left, bottom, width, height = plt1_ax.get_position().bounds
    colorbar_axes = fig.add_axes([left + 0.4, bottom,0.02, height])
    cbar = fig.colorbar(im1, colorbar_axes, orientation='vertical')
    cbar.set_label(r'K/K',fontsize=14) #rotation= radianes
    cbar.ax.tick_params(axis='both',labelsize=14)

    ax2 = plt.subplot(3,1,2,projection=proj)
    im2=ax2.contourf(lons, lats, model_sym,levels,transform=data_crs,cmap='OrRd',extend='both')
    ax2.set_title(funciones.split_title_line(r'b) EOF 2 - '+str(mod_name), max_words=5),fontsize=14)
    ax2.add_feature(cartopy.feature.COASTLINE,alpha=.5)
    ax2.add_feature(cartopy.feature.BORDERS, linestyle='-', alpha=.5)
    ax2.gridlines(crs=data_crs, linewidth=0.3, linestyle='-')
    ax2.set_extent([-180, 180, -90, 90], ccrs.PlateCarree(central_longitude=180))
    ax2.gridlines(draw_labels=True, crs=ccrs.PlateCarree())

    plt2_ax = plt.gca()
    left, bottom, width, height = plt2_ax.get_position().bounds
    colorbar_axes = fig.add_axes([left + 0.4, bottom,0.02, height])
    cbar = fig.colorbar(im2, colorbar_axes, orientation='vertical')
    cbar.set_label('K/K',fontsize=14)
    cbar.ax.tick_params(axis='both',labelsize=14)

    ax3 = plt.subplot(3,1,3,projection=proj)
    im3=ax3.contourf(lons, lats, model_asym,alevels,transform=data_crs,cmap=asym_cmap,extend='both')
    ax3.set_title(funciones.split_title_line(r'c) EOF 3 - '+str(mod_name), max_words=5),fontsize=14)
    ax3.add_feature(cartopy.feature.COASTLINE,alpha=.5)
    ax3.add_feature(cartopy.feature.BORDERS, linestyle='-', alpha=.5)
    ax3.gridlines(crs=data_crs, linewidth=0.3, linestyle='-')
    ax3.set_extent([-180, 180, -90, 90], ccrs.PlateCarree(central_longitude=180))
    ax3.gridlines(draw_labels=True, crs=ccrs.PlateCarree())

    plt3_ax = plt.gca()
    left, bottom, width, height = plt3_ax.get_position().bounds
    colorbar_axes = fig.add_axes([left + 0.4, bottom,0.02, height])
    cbar = fig.colorbar(im3, colorbar_axes, orientation='vertical')
    cbar.set_label(r'K/K',fontsize=14) #rotation = radianes
    cbar.ax.tick_params(axis='both',labelsize=14)
    
    fig.subplots_adjust(hspace=0.5)
    return fig



