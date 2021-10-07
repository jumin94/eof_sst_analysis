#Calculo el EOF de precipitacion en SESA
import cartopy.crs as ccrs
import cartopy.feature
import matplotlib.path as mpath
import os
import glob
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import xarray as xr
import cartopy.crs as ccrs
from cartopy.util import add_cyclic_point
import math
from eofs.xarray import Eof
import logging
import utilities.funciones
import os, fnmatch

#----------------------------------------------------------------
#Funciones-------------------------------------------------------
#Anomalias 
def anomalias(datos,scenarios,models):
    pr_hist = {}; pr_ssp = {}
    for model in models:
        precip = datos[scenarios[0]][model]['1940-1969'][0]
        pr = precip.pr
        pr.attrs = precip.pr.attrs
        climatology = pr.groupby('time.month').mean('time')
        pr_anom = pr.groupby('time.month') - climatology
        pr =  pr.sel(lat=slice(-16,-39)).sel(lon=slice(296,329))
        lat = pr.lat
        lon = pr.lon
        time = precip.time

        prDJF = pr.sel(time=pr['time.season']=='DJF')
        pr_hist[model] = []
        pr_hist[model].append(prDJF)
    for model in models:
        precip = datos[scenarios[1]][model]['2070-2099'][0]
        pr = precip.pr
        pr.attrs = precip.pr.attrs
        climatology = pr.groupby('time.month').mean('time')
        pr_anom = pr.groupby('time.month') - climatology
        pr =  pr.sel(lat=slice(-16,-39)).sel(lon=slice(296,329))
        lat = pr.lat
        lon = pr.lon
        time = precip.time

        prDJF = pr.sel(time=pr['time.season']=='DJF')
        pr_ssp[model] = []
        pr_ssp[model].append(prDJF)

    return pr_hist, pr_ssp

def eof_solver(data):
    # Create an EOF solver to do the EOF analysis. Square-root of cosine of
    # latitude weights are applied before the computation of EOFs.
    coslat = np.cos(np.deg2rad(data['lat'].values))
    coslat[0]=0
    coslat[len(data['lat'])-1] = 0
    wgts = np.sqrt(coslat)[..., np.newaxis]

    #Calculo componente principal   
    solver = Eof(data, weights=wgts)
    # Retrieve the leading EOF, expressed as the correlation between the leading
    # PC time series and the input anomalies at each grid point, and the
    # leading PC time series itself.
    eof1 = solver.eofsAsCorrelation(neofs=2)
    pc1 = solver.pcs(npcs=2, pcscaling=1)

    return eof1, pc1

def eof_solver_manual(dato):
    #Create an EOF solver through SVD. Square-root of cosine of
    # latitude weights are applied before the computation of EOFs.
    M = np.zeros([int(len(dato.lat)*len(dato.lon)), len(dato.time)])
    cont = 0
    for i in range(len(dato.lat)):
        for j in range(len(dato.lon)):
            if (dato.lat[i].values == 0) or (dato.lat[i].values == 90):
                coslat = 0
                time_evolution_ij = np.nan_to_num(dato.tos.values[:,i,j],0) 
                M[cont] =  signal.detrend(time_evolution_ij) * np.sqrt(coslat)
                cont += 1
            else:
                coslat = np.cos(np.deg2rad(dato.lat[i].values))
                time_evolution_ij = np.nan_to_num(dato.tos.values[:,i,j],0) 
                M[cont] =  signal.detrend(time_evolution_ij) * np.sqrt(coslat)
                cont += 1

    N = M.T
    U, s, VT = np.linalg.svd(N)
    S = np.zeros((N.shape[0], N.shape[1]))
    S[:N.shape[0], :N.shape[0]] = np.diag(s)
    PC = U.dot(S)
    return VT, PC


def VT_to_EOF_pattern(data,vt,n):
    eof_pattern = np.zeros((len(data.lat),len(data.lon)))
    cont = 0
    for i in range(len(data.lat)):
        for j in range(len(data.lon)):
            eof_pattern[i,j] = vt[n-1,cont]
            cont += 1

    return eof_pattern


def main():
    logging.basicConfig(filename='rainfall_EOF1.log', level=logging.INFO)
    logging.info('Empieza el programa')
    #Data paths
    path = '/datos/julia.mindlin/CMIP6_ensambles/preprocesados'
    path_rean = '/datos/ERA5/mon'
    #Output figure paths
    path_fig = '/home/julia.mindlin/Tesis/Capitulo3/Figuras/EOFs_SSTs'
    #Open datasets
    modelos = [
        'ACCESS-CM2', 'ACCESS-ESM1-5', 'BCC-CSM2-MR', 'CAMS-CSM1-0','CanESM5',          'CESM2_', 'CESM2-WACCM','CMCC-CM2-SR5','CNRM-CM6-1',
        'CNRM-ESM2-1','EC-Earth3', 'FGOALS-g3', 'GFDL-ESM4',
        'HadGEM3-GC31-LL','HadGEM3-GC31-MM','INM-CM4-8','INM-CM5-0',
        'KACE-1-0-G','MIROC6','MIROC-ES2L', 'MPI-ESM1-2-HR',
        'MPI-ESM1-2-LR','MRI-ESM2-0', 'NESM3', 'NorESM2-LM', 'NorESM2-MM',
        'UKESM1-0-LL'
        ]
    var = 'mon/pr'
    scenarios = ['historical','ssp585']
    variables = ['pr']
    dato = funciones.cargo_todo(scenarios,modelos,path,var)
    
    pr_hist, pr_ssp = anomalias(dato,scenarios,modelos)
    logging.info('Calcule anomalias')
    for model in modelos:
        #eof1 = eof_solver(pr_hist[model][0])
        dato = pr_hist[model][0]
        eofs_VT, eofs_PC = eof_solver_manual(dato)
        eof1 = VT_to_EOF_pattern(dato,eofs_VT,1); pc1 = eofs_PC.T[0][:]
        eof2 = VT_to_EOF_pattern(dato,eofs_VT,2); pc2 = eofs_PC.T[1][:]
        eof3 = VT_to_EOF_pattern(dato,eofs_VT,3); pc3 = eofs_PC.T[2][:]
        logging.info('Calcule bien el EOF1')
        levels = np.arange(-2,2,.1); alevels = np.arange(-.5,.6,.1)
        figure = plot_eofs(eof1,eof2,eof3,model,dato.lon,dato.lat,levels,levels,'RdBu_r')
        plt.savefig(path_fig+'/SST_eofs_'+model+'.png',bbox_inches='tight')
        plt.close('all')


if __name__ == '__main__':
    main()

