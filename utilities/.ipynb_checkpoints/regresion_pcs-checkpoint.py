# Regresion de los campos completos de SSTs contra las componentes principales
#Imports
import numpy as np
import pandas as pd
import xarray as xr
import math
from sklearn import linear_model
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
import os, fnmatch
import glob
import utilities.csv2nc


#Principal component regression class

class EOF_regression(object):
    def __init__(self):
        self.what_is_this = 'This performs a regression patterns for principal component analysis'
    
    def regression_data_models(self,data_dic,scenarios,model):

        #----------------------------------------
        self.model = model
        ssts_hist = data_dic[scenarios[0]][model][0]
        self.dat_hist = ssts_hist.sel(time=self.cross_year_season(ssts_hist.tos['time.month'])).sel(lat=slice(20,-20)).sel(lon=slice(120,290)).sel(time=slice('1950-01','1999-12'))
        ssts_ssp = data_dic[scenarios[1]][model][0]
        self.dat_ssp = ssts_ssp.sel(time=self.cross_year_season(ssts_ssp.tos['time.month'])).sel(lat=slice(20,-20)).sel(lon=slice(120,290)).sel(time=slice('2050-01','2099-12'))     
        self.time_hist = ssts_hist.sel(time=slice('1950-01','1999-12')).time
        self.time_ssp = ssts_ssp.sel(time=slice('2050-01','2099-12')).time
        self.lat = ssts_ssp.sel(lat=slice(20,-20)).lat; self.lon = ssts_ssp.sel(lon=slice(120,290)).lon

    def regression_data_obs(self,data_name,data,domain,time_domain,season):

        #----------------------------------------
        self.model = data_name
        if season == 'None':
            self.dat = data.sel(lat=slice(domain[0],domain[1])).sel(lon=slice(domain[2],domain[3]))
        else:
            self.dat = data.sel(time=self.cross_year_season(ssts_hist.sst['time.month'],season)).sel(lat=slice(domain[0],domain[1])).sel(lon=slice(domain[2],domain[3]))
            
        self.dat = self.dat.sel(time=slice(time_domain[0],time_domain[1]))    
        self.time = self.dat.time
        self.lat = self.dat.lat; self.lon = self.dat.lon
      
    def cross_year_season(self,month,season):
        if season == 'DJF':
            return (month >= 12) | (month <= 2)
        elif season == 'JJA':
            return (month >= 6) & (month <= 8)
        #return (month >= 9) & (month <= 11)

    def stand(self,serie):
        return (serie - np.mean(serie))/np.std(serie)

    def perform_regression_obs(self,pcs,path,eof_num,time_out,regressors_in='None'): 

        
        EOF1ij = pd.DataFrame(columns=['eof1','lat','lon'])
        EOF2ij = pd.DataFrame(columns=['eof2','lat','lon'])
        if eof_num >= 3:
            EOF3ij = pd.DataFrame(columns=['eof3','lat','lon'])
            EOF4ij = pd.DataFrame(columns=['eof4','lat','lon'])
        else:
            a = 0
            
        EOF1ij_t = pd.DataFrame(columns=['eof1_t','lat','lon'])
        EOF2ij_t = pd.DataFrame(columns=['eof2_t','lat','lon'])
        if eof_num >=3:
            EOF3ij_t = pd.DataFrame(columns=['eof3_t','lat','lon'])
            EOF4ij_t = pd.DataFrame(columns=['eof4_t','lat','lon'])
        else:
            a = 0
        R2ij = pd.DataFrame(columns=['r2','lat','lon'])
        x = np.array([])
        
        #Generate indices and regressors diccionary 
        if eof_num >=3:
            regressors = pd.DataFrame({'pc1':self.stand(pcs['pc1']),
                                       'pc2':self.stand(pcs['pc2']),
                                       'pc3':self.stand(pcs['pc3']),
                                       'pc4':self.stand(pcs['pc4'])})
        else:
            regressors = pd.DataFrame({'pc1':self.stand(pcs['pc1']),
                                       'pc2':self.stand(pcs['pc2'])})
            
        if regressors_in == 'None':
            self.regressors = regressors
        else:
            self.regressors = regressors_in
        #Regresion lineal
        y = regressors.values
        #y = sm.add_constant(regressors.values)
        lat = self.lat
        lon = self.lon
        reg = linear_model.LinearRegression()
        
        campo = self.dat.sst
        if eof_num >= 3:
            for i in range(len(lat)):
                for j in range(len(lon)):
                    if np.isnan(self.dat.sst[:,i-1,j-1].values).any():
                        eof1 = pd.DataFrame({'eof1':[np.nan],'lat':[lat[i-1].values.tolist()],'lon':[lon[j-1].values.tolist()]})
                        EOF1ij = pd.concat([EOF1ij,eof1],axis=0)
                        eof2 = pd.DataFrame({'eof2':[np.nan],'lat':[lat[i-1].values.tolist()],'lon':[lon[j-1].values.tolist()]})
                        EOF2ij = pd.concat([EOF2ij,eof2],axis=0)
                        eof3 = pd.DataFrame({'eof3':[np.nan],'lat':[lat[i-1].values.tolist()],'lon':[lon[j-1].values.tolist()]})
                        EOF3ij = pd.concat([EOF3ij,eof3],axis=0)
                        eof4 = pd.DataFrame({'eof4':[np.nan],'lat':[lat[i-1].values.tolist()],'lon':[lon[j-1].values.tolist()]})
                        EOF4ij = pd.concat([EOF4ij,eof4],axis=0)
                        eof1_t = pd.DataFrame({'eof1_t':[np.nan],'lat':[lat[i-1].values.tolist()],'lon':[lon[j-1].values.tolist()]})
                        EOF1ij_t = pd.concat([EOF1ij_t,eof1_t],axis=0)
                        eof2_t = pd.DataFrame({'eof2_t':[np.nan],'lat':[lat[i-1].values.tolist()],'lon':[lon[j-1].values.tolist()]})
                        EOF2ij_t = pd.concat([EOF2ij_t,eof2_t],axis=0)
                        eof3_t = pd.DataFrame({'eof3_t':[np.nan],'lat':[lat[i-1].values.tolist()],'lon':[lon[j-1].values.tolist()]})
                        EOF3ij_t = pd.concat([EOF3ij_t,eof3_t],axis=0)
                        eof4_t = pd.DataFrame({'eof4_t':[np.nan],'lat':[lat[i-1].values.tolist()],'lon':[lon[j-1].values.tolist()]})
                        EOF4ij_t = pd.concat([EOF4ij_t,eof4_t],axis=0)
                        del eof1,eof2,eof3,eof4,eof1_t,eof2_t,eof3_t,eof4_t
                        x = np.array([])
                        continue
                    x = self.create_x(i,j,campo)
                    res = sm.OLS(x,y).fit()
                    eof1 = res.params[0]
                    eof2 = res.params[1]
                    eof3 = res.params[2]
                    eof4 = res.params[3]

                    eof1_t = res.pvalues[0]
                    eof2_t = res.pvalues[1]
                    eof3_t = res.pvalues[2]
                    eof4_t = res.pvalues[3]

                    r2 = res.rsquared
                    mse = res.conf_int(alpha=0.05, cols=None) #mse_model
                    eof1 = pd.DataFrame({'eof1':eof1,'lat':[lat[i-1].values.tolist()],'lon':[lon[j-1].values.tolist()]})
                    EOF1ij = pd.concat([EOF1ij,eof1],axis=0)
                    eof2 = pd.DataFrame({'eof2':eof2,'lat':[lat[i-1].values.tolist()],'lon':[lon[j-1].values.tolist()]})
                    EOF2ij = pd.concat([EOF2ij,eof2],axis=0)
                    eof3 = pd.DataFrame({'eof3':eof3,'lat':[lat[i-1].values.tolist()],'lon':[lon[j-1].values.tolist()]})
                    EOF3ij = pd.concat([EOF3ij,eof3],axis=0)
                    eof4 = pd.DataFrame({'eof4':eof4,'lat':[lat[i-1].values.tolist()],'lon':[lon[j-1].values.tolist()]})
                    EOF4ij = pd.concat([EOF4ij,eof4],axis=0)

                    r2  = pd.DataFrame({'r2':r2,'lat':[lat[i-1].values.tolist()],'lon':[lon[j-1].values.tolist()]})
                    R2ij = pd.concat([R2ij,r2],axis=0)

                    eof1_t = pd.DataFrame({'eof1_t':eof1_t,'lat':[lat[i-1].values.tolist()],'lon':[lon[j-1].values.tolist()]})
                    EOF1ij_t = pd.concat([EOF1ij_t,eof1_t],axis=0)
                    eof2_t = pd.DataFrame({'eof2_t':eof2_t,'lat':[lat[i-1].values.tolist()],'lon':[lon[j-1].values.tolist()]})
                    EOF2ij_t = pd.concat([EOF2ij_t,eof2_t],axis=0)
                    eof3_t = pd.DataFrame({'eof3_t':eof3_t,'lat':[lat[i-1].values.tolist()],'lon':[lon[j-1].values.tolist()]})
                    EOF3ij_t = pd.concat([EOF3ij_t,eof3_t],axis=0)
                    eof4_t = pd.DataFrame({'eof4_t':eof4_t,'lat':[lat[i-1].values.tolist()],'lon':[lon[j-1].values.tolist()]})
                    EOF4ij_t = pd.concat([EOF4ij_t,eof4_t],axis=0)                

                    del r2, res, eof1,eof2,eof3,eof4,eof1_t,eof2_t,eof3_t,eof4_t
                    x = np.array([])
        else:
            for i in range(len(lat)):
                for j in range(len(lon)):
                    if np.isnan(self.dat.sst[:,i-1,j-1].values).any():
                        eof1 = pd.DataFrame({'eof1':[np.nan],'lat':[lat[i-1].values.tolist()],'lon':[lon[j-1].values.tolist()]})
                        EOF1ij = pd.concat([EOF1ij,eof1],axis=0)
                        eof2 = pd.DataFrame({'eof2':[np.nan],'lat':[lat[i-1].values.tolist()],'lon':[lon[j-1].values.tolist()]})
                        EOF2ij = pd.concat([EOF2ij,eof2],axis=0)
                        eof1_t = pd.DataFrame({'eof1_t':[np.nan],'lat':[lat[i-1].values.tolist()],'lon':[lon[j-1].values.tolist()]})
                        EOF1ij_t = pd.concat([EOF1ij_t,eof1_t],axis=0)
                        eof2_t = pd.DataFrame({'eof2_t':[np.nan],'lat':[lat[i-1].values.tolist()],'lon':[lon[j-1].values.tolist()]})
                        EOF2ij_t = pd.concat([EOF2ij_t,eof2_t],axis=0)
                        del eof1,eof2,eof1_t,eof2_t
                        x = np.array([])
                        continue
                    x = self.create_x(i,j,campo)
                    res = sm.OLS(x,y).fit()
                    eof1 = res.params[0]
                    eof2 = res.params[1]

                    eof1_t = res.pvalues[0]
                    eof2_t = res.pvalues[1]

                    r2 = res.rsquared
                    mse = res.conf_int(alpha=0.05, cols=None) #mse_model
                    
                    eof1 = pd.DataFrame({'eof1':eof1,'lat':[lat[i-1].values.tolist()],'lon':[lon[j-1].values.tolist()]})
                    EOF1ij = pd.concat([EOF1ij,eof1],axis=0)
                    eof2 = pd.DataFrame({'eof2':eof2,'lat':[lat[i-1].values.tolist()],'lon':[lon[j-1].values.tolist()]})
                    EOF2ij = pd.concat([EOF2ij,eof2],axis=0)

                    r2  = pd.DataFrame({'r2':r2,'lat':[lat[i-1].values.tolist()],'lon':[lon[j-1].values.tolist()]})
                    R2ij = pd.concat([R2ij,r2],axis=0)

                    eof1_t = pd.DataFrame({'eof1_t':eof1_t,'lat':[lat[i-1].values.tolist()],'lon':[lon[j-1].values.tolist()]})
                    EOF1ij_t = pd.concat([EOF1ij_t,eof1_t],axis=0)
                    eof2_t = pd.DataFrame({'eof2_t':eof2_t,'lat':[lat[i-1].values.tolist()],'lon':[lon[j-1].values.tolist()]})
                    EOF2ij_t = pd.concat([EOF2ij_t,eof2_t],axis=0)             

                    del r2, res, eof1,eof2,eof1_t,eof2_t
                    x = np.array([])
                
                
        #Guardo resultados en DataFrames
        EOF1 = {'eof1':EOF1ij.iloc[:,0],'lat':EOF1ij.iloc[:,1],'lon':EOF1ij.iloc[:,2]}
        EOF1ij = pd.DataFrame(EOF1).fillna(0)
        eof1 = self.df_to_xr(EOF1ij,'eof1')
        EOF2 = {'eof2':EOF2ij.iloc[:,0],'lat':EOF2ij.iloc[:,1],'lon':EOF2ij.iloc[:,2]}
        EOF2ij = pd.DataFrame(EOF2).fillna(0)
        eof2 = self.df_to_xr(EOF2ij,'eof2')
        if eof_num >= 3:
            EOF3 = {'eof3':EOF3ij.iloc[:,0],'lat':EOF3ij.iloc[:,1],'lon':EOF3ij.iloc[:,2]}
            EOF3ij = pd.DataFrame(EOF3).fillna(0)
            eof3 = self.df_to_xr(EOF3ij,'eof3')
            EOF4 = {'eof4':EOF4ij.iloc[:,0],'lat':EOF4ij.iloc[:,1],'lon':EOF4ij.iloc[:,2]}
            EOF4ij = pd.DataFrame(EOF4).fillna(0)
            eof4 = self.df_to_xr(EOF4ij,'eof4')
        else:
            a=0
        
        EOF1p = {'eof1_pval':EOF1ij_t.iloc[:,0],'lat':EOF1ij_t.iloc[:,1],'lon':EOF1ij_t.iloc[:,2]}
        EOF1pij = pd.DataFrame(EOF1p).fillna(10)
        eof1_p = self.df_to_xr(EOF1pij,'eof1_pval')
        EOF2p = {'eof2_pval':EOF2ij_t.iloc[:,0],'lat':EOF2ij_t.iloc[:,1],'lon':EOF2ij_t.iloc[:,2]}
        EOF2pij = pd.DataFrame(EOF2p).fillna(10)    
        eof2_p = self.df_to_xr(EOF2pij,'eof2_pval')
        if eof_num >= 3:
            EOF3p = {'eof3_pval':EOF3ij_t.iloc[:,0],'lat':EOF3ij_t.iloc[:,1],'lon':EOF3ij_t.iloc[:,2]}
            EOF3pij = pd.DataFrame(EOF3p).fillna(10)
            eof3_p = self.df_to_xr(EOF3pij,'eof3_pval')
            EOF4p = {'eof4_pval':EOF4ij_t.iloc[:,0],'lat':EOF4ij_t.iloc[:,1],'lon':EOF4ij_t.iloc[:,2]}
            EOF4pij = pd.DataFrame(EOF4p).fillna(10)
            eof4_p = self.df_to_xr(EOF4pij,'eof4_pval')
        else:
            a=0
        
        R2 = {'r2':R2ij.iloc[:,0],'lat':R2ij.iloc[:,1],'lon':R2ij.iloc[:,2]}
        R2ij = pd.DataFrame(R2).fillna(0)
        r2 = self.df_to_xr(R2ij,'r2')

        if eof_num >=3:
            ds_all = xr.merge([eof1,eof2,eof3,eof4,eof1_p,
                               eof2_p,eof3_p,eof4_p,r2])
        else:
            ds_all = xr.merge([eof1,eof2,eof1_p,eof2_p,r2])
            
        ds_all.to_netcdf(path+'/'+self.model+'_PC_regression_pattern_'+time_out+'_detrended.nc')
    
    
        EOF1ij.to_csv(path+'/'+self.model+'_EOF1ij_PC_regression_pattern_'+time_out+'_detrended.csv', float_format='%g')
        EOF2ij.to_csv(path+'/'+self.model+'_EOF2ij_PC_regression_pattern_'+time_out+'_detrended.csv', float_format='%g')
        if eof_num >= 3:
            EOF3ij.to_csv(path+'/'+self.model+'_EOF3ij_PC_regression_pattern_'+time_out+'_detrended.csv', float_format='%g')
            EOF4ij.to_csv(path+'/'+self.model+'_EOF4ij_PC_regression_pattern_'+time_out+'_detrended.csv', float_format='%g')
        else: 
            a=0
        EOF1pij.to_csv(path+'/'+self.model+'_EOF1ij_p_PC_regression_pattern_'+time_out+'_detrended.csv', float_format='%g')
        EOF2pij.to_csv(path+'/'+self.model+'_EOF2ij_p_PC_regression_pattern_'+time_out+'_detrended.csv', float_format='%g')
        
        if eof_num >= 3:
            EOF3pij.to_csv(path+'/'+self.model+'_EOF3ij_p_PC_regression_pattern_'+time_out+'_detrended.csv', float_format='%g')
            EOF4pij.to_csv(path+'/'+self.model+'_EOF4ij_p_PC_regression_pattern_'+time_out+'_detrended.csv', float_format='%g')
        else:
            a=0
        
        R2ij.to_csv(path+'/'+self.model+'_R2ij_PC_regression_pattern_'+time_out+'_detrended.csv', float_format='%g')
                
        
    def perform_regression_historical(self,pcs,path): 

        
        EOF1ij = pd.DataFrame(columns=['eof1','lat','lon'])
        EOF2ij = pd.DataFrame(columns=['eof2','lat','lon'])
        EOF3ij = pd.DataFrame(columns=['eof3','lat','lon'])
        EOF4ij = pd.DataFrame(columns=['eof4','lat','lon'])
        EOF1ij_t = pd.DataFrame(columns=['eof1_t','lat','lon'])
        EOF2ij_t = pd.DataFrame(columns=['eof2_t','lat','lon'])
        EOF3ij_t = pd.DataFrame(columns=['eof3_t','lat','lon'])
        EOF4ij_t = pd.DataFrame(columns=['eof4_t','lat','lon'])
        R2ij = pd.DataFrame(columns=['r2','lat','lon'])
        x = np.array([])
        
        #Generate indices and regressors diccionary 
        regressors = pd.DataFrame({'pc1':self.stand(pcs['pc1']),
                                   'pc2':self.stand(pcs['pc2']),
                                   'pc3':self.stand(pcs['pc3']),
                                   'pc4':self.stand(pcs['pc4'])})
        
        self.regressors = regressors
        #Regresion lineal
        y = regressors.values
        #y = sm.add_constant(regressors.values)
        lat = self.dat_hist.tos[0].lat
        lon = self.dat_hist.tos[0].lon
        reg = linear_model.LinearRegression()
        
        campo = self.dat_hist.tos
        for i in range(len(lat)):
            for j in range(len(lon)):
                if np.isnan(self.dat_hist.tos[:,i-1,j-1].values).any():
                    eof1 = pd.DataFrame({'eof1':[np.nan],'lat':[lat[i-1].values.tolist()],'lon':[lon[j-1].values.tolist()]})
                    EOF1ij = pd.concat([EOF1ij,eof1],axis=0)
                    eof2 = pd.DataFrame({'eof2':[np.nan],'lat':[lat[i-1].values.tolist()],'lon':[lon[j-1].values.tolist()]})
                    EOF2ij = pd.concat([EOF2ij,eof2],axis=0)
                    eof3 = pd.DataFrame({'eof3':[np.nan],'lat':[lat[i-1].values.tolist()],'lon':[lon[j-1].values.tolist()]})
                    EOF3ij = pd.concat([EOF3ij,eof3],axis=0)
                    eof4 = pd.DataFrame({'eof4':[np.nan],'lat':[lat[i-1].values.tolist()],'lon':[lon[j-1].values.tolist()]})
                    EOF4ij = pd.concat([EOF4ij,eof4],axis=0)
                    eof1_t = pd.DataFrame({'eof1_t':[np.nan],'lat':[lat[i-1].values.tolist()],'lon':[lon[j-1].values.tolist()]})
                    EOF1ij_t = pd.concat([EOF1ij_t,eof1_t],axis=0)
                    eof2_t = pd.DataFrame({'eof2_t':[np.nan],'lat':[lat[i-1].values.tolist()],'lon':[lon[j-1].values.tolist()]})
                    EOF2ij_t = pd.concat([EOF2ij_t,eof2_t],axis=0)
                    eof3_t = pd.DataFrame({'eof3_t':[np.nan],'lat':[lat[i-1].values.tolist()],'lon':[lon[j-1].values.tolist()]})
                    EOF3ij_t = pd.concat([EOF3ij_t,eof3_t],axis=0)
                    eof4_t = pd.DataFrame({'eof4_t':[np.nan],'lat':[lat[i-1].values.tolist()],'lon':[lon[j-1].values.tolist()]})
                    EOF4ij_t = pd.concat([EOF4ij_t,eof4_t],axis=0)
                    del eof1,eof2,eof3,eof4,eof1_t,eof2_t,eof3_t,eof4_t
                    x = np.array([])
                    continue
                x = self.create_x(i,j,campo)
                res = sm.OLS(x,y).fit()
                eof1 = res.params[0]
                eof2 = res.params[1]
                eof3 = res.params[2]
                eof4 = res.params[3]

                eof1_t = res.pvalues[0]
                eof2_t = res.pvalues[1]
                eof3_t = res.pvalues[2]
                eof4_t = res.pvalues[3]

                r2 = res.rsquared
                mse = res.conf_int(alpha=0.05, cols=None) #mse_model
                eof1 = pd.DataFrame({'eof1':eof1,'lat':[lat[i-1].values.tolist()],'lon':[lon[j-1].values.tolist()]})
                EOF1ij = pd.concat([EOF1ij,eof1],axis=0)
                eof2 = pd.DataFrame({'eof2':eof2,'lat':[lat[i-1].values.tolist()],'lon':[lon[j-1].values.tolist()]})
                EOF2ij = pd.concat([EOF2ij,eof2],axis=0)
                eof3 = pd.DataFrame({'eof3':eof3,'lat':[lat[i-1].values.tolist()],'lon':[lon[j-1].values.tolist()]})
                EOF3ij = pd.concat([EOF3ij,eof3],axis=0)
                eof4 = pd.DataFrame({'eof4':eof4,'lat':[lat[i-1].values.tolist()],'lon':[lon[j-1].values.tolist()]})
                EOF4ij = pd.concat([EOF4ij,eof4],axis=0)
                 
                r2  = pd.DataFrame({'r2':r2,'lat':[lat[i-1].values.tolist()],'lon':[lon[j-1].values.tolist()]})
                R2ij = pd.concat([R2ij,r2],axis=0)
                
                eof1_t = pd.DataFrame({'eof1_t':eof1_t,'lat':[lat[i-1].values.tolist()],'lon':[lon[j-1].values.tolist()]})
                EOF1ij_t = pd.concat([EOF1ij_t,eof1_t],axis=0)
                eof2_t = pd.DataFrame({'eof2_t':eof2_t,'lat':[lat[i-1].values.tolist()],'lon':[lon[j-1].values.tolist()]})
                EOF2ij_t = pd.concat([EOF2ij_t,eof2_t],axis=0)
                eof3_t = pd.DataFrame({'eof3_t':eof3_t,'lat':[lat[i-1].values.tolist()],'lon':[lon[j-1].values.tolist()]})
                EOF3ij_t = pd.concat([EOF3ij_t,eof3_t],axis=0)
                eof4_t = pd.DataFrame({'eof4_t':eof4_t,'lat':[lat[i-1].values.tolist()],'lon':[lon[j-1].values.tolist()]})
                EOF4ij_t = pd.concat([EOF4ij_t,eof4_t],axis=0)                

                del r2, res, eof1,eof2,eof3,eof4,eof1_t,eof2_t,eof3_t,eof4_t
                x = np.array([])
                
        #Guardo resultados en DataFrames
        EOF1 = {'eof1':EOF1ij.iloc[:,0],'lat':EOF1ij.iloc[:,1],'lon':EOF1ij.iloc[:,2]}
        EOF1ij = pd.DataFrame(EOF1).fillna(0)
        eof1 = self.df_to_xr(EOF1ij,'eof1')
        EOF2 = {'eof2':EOF2ij.iloc[:,0],'lat':EOF2ij.iloc[:,1],'lon':EOF2ij.iloc[:,2]}
        EOF2ij = pd.DataFrame(EOF2).fillna(0)
        eof2 = self.df_to_xr(EOF2ij,'eof2')
        EOF3 = {'eof3':EOF3ij.iloc[:,0],'lat':EOF3ij.iloc[:,1],'lon':EOF3ij.iloc[:,2]}
        EOF3ij = pd.DataFrame(EOF3).fillna(0)
        eof3 = self.df_to_xr(EOF3ij,'eof3')
        EOF4 = {'eof4':EOF4ij.iloc[:,0],'lat':EOF4ij.iloc[:,1],'lon':EOF4ij.iloc[:,2]}
        EOF4ij = pd.DataFrame(EOF4).fillna(0)
        eof4 = self.df_to_xr(EOF4ij,'eof4')
        
        EOF1p = {'eof1_pval':EOF1ij_t.iloc[:,0],'lat':EOF1ij_t.iloc[:,1],'lon':EOF1ij_t.iloc[:,2]}
        EOF1pij = pd.DataFrame(EOF1p).fillna(10)
        eof1_p = self.df_to_xr(EOF1pij,'eof1_pval')
        EOF2p = {'eof2_pval':EOF2ij_t.iloc[:,0],'lat':EOF2ij_t.iloc[:,1],'lon':EOF2ij_t.iloc[:,2]}
        EOF2pij = pd.DataFrame(EOF2p).fillna(10)    
        eof2_p = self.df_to_xr(EOF2pij,'eof2_pval')
        EOF3p = {'eof3_pval':EOF3ij_t.iloc[:,0],'lat':EOF3ij_t.iloc[:,1],'lon':EOF3ij_t.iloc[:,2]}
        EOF3pij = pd.DataFrame(EOF3p).fillna(10)
        eof3_p = self.df_to_xr(EOF3pij,'eof3_pval')
        EOF4p = {'eof4_pval':EOF4ij_t.iloc[:,0],'lat':EOF4ij_t.iloc[:,1],'lon':EOF4ij_t.iloc[:,2]}
        EOF4pij = pd.DataFrame(EOF4p).fillna(10)
        eof4_p = self.df_to_xr(EOF4pij,'eof4_pval')
        
        R2 = {'r2':R2ij.iloc[:,0],'lat':R2ij.iloc[:,1],'lon':R2ij.iloc[:,2]}
        R2ij = pd.DataFrame(R2).fillna(0)
        r2 = self.df_to_xr(R2ij,'r2')

        ds_all = xr.merge([eof1,eof2,eof3,eof4,eof1_p,
                          eof2_p,eof3_p,eof4_p])
        ds_all.to_netcdf(path+'/eof_patterns/'+self.model+'_historical_eof1_1950-1999_DJF_detrended.nc')
    
        
        
        EOF1ij.to_csv(path+'/'+self.model+'_EOF1ij_historical_reg_patterns_1950-1999_DJF_detreded.csv', float_format='%g')
        EOF2ij.to_csv(path+'/'+self.model+'_EOF2ij_historical_reg_patterns_1950-1999_DJF_detreded.csv', float_format='%g')
        EOF3ij.to_csv(path+'/'+self.model+'_EOF3ij_historical_reg_patterns_1950-1999_DJF_detreded.csv', float_format='%g')
        EOF4ij.to_csv(path+'/'+self.model+'_EOF4ij_historical_reg_patterns_1950-1999_DJF_detreded.csv', float_format='%g')
        EOF1pij.to_csv(path+'/'+self.model+'_EOF1ij_p_historical_reg_patterns_1950-1999_DJF_detreded.csv', float_format='%g')
        EOF2pij.to_csv(path+'/'+self.model+'_EOF2ij_p_historical_reg_patterns_1950-1999_DJF_detreded.csv', float_format='%g')
        EOF3pij.to_csv(path+'/'+self.model+'_EOF3ij_p_historical_reg_patterns_1950-1999_DJF_detreded.csv', float_format='%g')
        EOF4pij.to_csv(path+'/'+self.model+'_EOF4ij_p_historical_reg_patterns_1950-1999_DJF_detreded.csv', float_format='%g')
        
        R2ij.to_csv(path+'/'+self.model+'_R2ij_historical_reg_patterns_1950-1999_DJF_detreded.csv', float_format='%g')
                
        
    def perform_regression_ssp(self,pcs,path): 

        
        EOF1ij = pd.DataFrame(columns=['eof1','lat','lon'])
        EOF2ij = pd.DataFrame(columns=['eof2','lat','lon'])
        EOF3ij = pd.DataFrame(columns=['eof3','lat','lon'])
        EOF4ij = pd.DataFrame(columns=['eof4','lat','lon'])
        EOF1ij_t = pd.DataFrame(columns=['eof1_t','lat','lon'])
        EOF2ij_t = pd.DataFrame(columns=['eof2_t','lat','lon'])
        EOF3ij_t = pd.DataFrame(columns=['eof3_t','lat','lon'])
        EOF4ij_t = pd.DataFrame(columns=['eof4_t','lat','lon'])
        R2ij = pd.DataFrame(columns=['r2','lat','lon'])
        x = np.array([])
        
        #Generate indices and regressors diccionary 
        regressors = pd.DataFrame({'pc1':self.stand(pcs['pc1']),
                                   'pc2':self.stand(pcs['pc2']),
                                   'pc3':self.stand(pcs['pc3']),
                                   'pc4':self.stand(pcs['pc4'])})
        
        self.regressors = regressors
        #Regresion lineal
        y = regressors.values
        #y = sm.add_constant(regressors.values)
        lat = self.dat_ssp.tos[0].lat
        lon = self.dat_ssp.tos[0].lon
        reg = linear_model.LinearRegression()
        
        campo = self.dat_hist.tos
        for i in range(len(lat)):
            for j in range(len(lon)):
                if np.isnan(self.dat_hist.tos[:,i-1,j-1].values).any():
                    eof1 = pd.DataFrame({'eof1':[np.nan],'lat':[lat[i-1].values.tolist()],'lon':[lon[j-1].values.tolist()]})
                    EOF1ij = pd.concat([EOF1ij,eof1],axis=0)
                    eof2 = pd.DataFrame({'eof2':[np.nan],'lat':[lat[i-1].values.tolist()],'lon':[lon[j-1].values.tolist()]})
                    EOF2ij = pd.concat([EOF2ij,eof2],axis=0)
                    eof3 = pd.DataFrame({'eof3':[np.nan],'lat':[lat[i-1].values.tolist()],'lon':[lon[j-1].values.tolist()]})
                    EOF3ij = pd.concat([EOF3ij,eof3],axis=0)
                    eof4 = pd.DataFrame({'eof4':[np.nan],'lat':[lat[i-1].values.tolist()],'lon':[lon[j-1].values.tolist()]})
                    EOF4ij = pd.concat([EOF4ij,eof4],axis=0)
                    eof1_t = pd.DataFrame({'eof1_t':[np.nan],'lat':[lat[i-1].values.tolist()],'lon':[lon[j-1].values.tolist()]})
                    EOF1ij_t = pd.concat([EOF1ij_t,eof1_t],axis=0)
                    eof2_t = pd.DataFrame({'eof2_t':[np.nan],'lat':[lat[i-1].values.tolist()],'lon':[lon[j-1].values.tolist()]})
                    EOF2ij_t = pd.concat([EOF2ij_t,eof2_t],axis=0)
                    eof3_t = pd.DataFrame({'eof3_t':[np.nan],'lat':[lat[i-1].values.tolist()],'lon':[lon[j-1].values.tolist()]})
                    EOF3ij_t = pd.concat([EOF3ij_t,eof3_t],axis=0)
                    eof4_t = pd.DataFrame({'eof4_t':[np.nan],'lat':[lat[i-1].values.tolist()],'lon':[lon[j-1].values.tolist()]})
                    EOF4ij_t = pd.concat([EOF4ij_t,eof4_t],axis=0)
                    del eof1,eof2,eof3,eof4,eof1_t,eof2_t,eof3_t,eof4_t
                    x = np.array([])
                    continue
                x = self.create_x(i,j,campo)
                res = sm.OLS(x,y).fit()
                eof1 = res.params[0]
                eof2 = res.params[1]
                eof3 = res.params[2]
                eof4 = res.params[3]

                eof1_t = res.pvalues[0]
                eof2_t = res.pvalues[1]
                eof3_t = res.pvalues[2]
                eof4_t = res.pvalues[3]

                r2 = res.rsquared
                mse = res.conf_int(alpha=0.05, cols=None) #mse_model
                eof1 = pd.DataFrame({'eof1':eof1,'lat':[lat[i-1].values.tolist()],'lon':[lon[j-1].values.tolist()]})
                EOF1ij = pd.concat([EOF1ij,eof1],axis=0)
                eof2 = pd.DataFrame({'eof2':eof2,'lat':[lat[i-1].values.tolist()],'lon':[lon[j-1].values.tolist()]})
                EOF2ij = pd.concat([EOF2ij,eof2],axis=0)
                eof3 = pd.DataFrame({'eof3':eof3,'lat':[lat[i-1].values.tolist()],'lon':[lon[j-1].values.tolist()]})
                EOF3ij = pd.concat([EOF3ij,eof3],axis=0)
                eof4 = pd.DataFrame({'eof4':eof4,'lat':[lat[i-1].values.tolist()],'lon':[lon[j-1].values.tolist()]})
                EOF4ij = pd.concat([EOF4ij,eof4],axis=0)
                 
                r2  = pd.DataFrame({'r2':r2,'lat':[lat[i-1].values.tolist()],'lon':[lon[j-1].values.tolist()]})
                R2ij = pd.concat([R2ij,r2],axis=0)
                
                eof1_t = pd.DataFrame({'eof1_t':eof1_t,'lat':[lat[i-1].values.tolist()],'lon':[lon[j-1].values.tolist()]})
                EOF1ij_t = pd.concat([EOF1ij_t,eof1_t],axis=0)
                eof2_t = pd.DataFrame({'eof2_t':eof2_t,'lat':[lat[i-1].values.tolist()],'lon':[lon[j-1].values.tolist()]})
                EOF2ij_t = pd.concat([EOF2ij_t,eof2_t],axis=0)
                eof3_t = pd.DataFrame({'eof3_t':eof3_t,'lat':[lat[i-1].values.tolist()],'lon':[lon[j-1].values.tolist()]})
                EOF3ij_t = pd.concat([EOF3ij_t,eof3_t],axis=0)
                eof4_t = pd.DataFrame({'eof4_t':eof4_t,'lat':[lat[i-1].values.tolist()],'lon':[lon[j-1].values.tolist()]})
                EOF4ij_t = pd.concat([EOF4ij_t,eof4_t],axis=0)                

                del r2, res, eof1,eof2,eof3,eof4,eof1_t,eof2_t,eof3_t,eof4_t
                x = np.array([])
                
        #Guardo resultados en DataFrames
        EOF1 = {'eof1':EOF1ij.iloc[:,0],'lat':EOF1ij.iloc[:,1],'lon':EOF1ij.iloc[:,2]}
        EOF1ij = pd.DataFrame(EOF1).fillna(0)
        eof1 = self.df_to_xr(EOF1ij,'eof1')
        EOF2 = {'eof2':EOF2ij.iloc[:,0],'lat':EOF2ij.iloc[:,1],'lon':EOF2ij.iloc[:,2]}
        EOF2ij = pd.DataFrame(EOF2).fillna(0)
        eof2 = self.df_to_xr(EOF2ij,'eof2')
        EOF3 = {'eof3':EOF3ij.iloc[:,0],'lat':EOF3ij.iloc[:,1],'lon':EOF3ij.iloc[:,2]}
        EOF3ij = pd.DataFrame(EOF3).fillna(0)
        eof3 = self.df_to_xr(EOF3ij,'eof3')
        EOF4 = {'eof4':EOF4ij.iloc[:,0],'lat':EOF4ij.iloc[:,1],'lon':EOF4ij.iloc[:,2]}
        EOF4ij = pd.DataFrame(EOF4).fillna(0)
        eof4 = self.df_to_xr(EOF4ij,'eof4')
        
        EOF1p = {'eof1_pval':EOF1ij_t.iloc[:,0],'lat':EOF1ij_t.iloc[:,1],'lon':EOF1ij_t.iloc[:,2]}
        EOF1pij = pd.DataFrame(EOF1p).fillna(10)
        eof1_p = self.df_to_xr(EOF1pij,'eof1_pval')
        EOF2p = {'eof2_pval':EOF2ij_t.iloc[:,0],'lat':EOF2ij_t.iloc[:,1],'lon':EOF2ij_t.iloc[:,2]}
        EOF2pij = pd.DataFrame(EOF2p).fillna(10)    
        eof2_p = self.df_to_xr(EOF2pij,'eof2_pval')
        EOF3p = {'eof3_pval':EOF3ij_t.iloc[:,0],'lat':EOF3ij_t.iloc[:,1],'lon':EOF3ij_t.iloc[:,2]}
        EOF3pij = pd.DataFrame(EOF3p).fillna(10)
        eof3_p = self.df_to_xr(EOF3pij,'eof3_pval')
        EOF4p = {'eof4_pval':EOF4ij_t.iloc[:,0],'lat':EOF4ij_t.iloc[:,1],'lon':EOF4ij_t.iloc[:,2]}
        EOF4pij = pd.DataFrame(EOF4p).fillna(10)
        eof4_p = self.df_to_xr(EOF4pij,'eof4_pval')
        
        R2 = {'r2':R2ij.iloc[:,0],'lat':R2ij.iloc[:,1],'lon':R2ij.iloc[:,2]}
        R2ij = pd.DataFrame(R2).fillna(0)
        r2 = self.df_to_xr(R2ij,'r2')

        ds_all = xr.merge([eof1,eof2,eof3,eof4,eof1_p,
                          eof2_p,eof3_p,eof4_p])
        ds_all.to_netcdf(path+'/eof_patterns/'+self.model+'_historical_eof1_1950-1999_DJF_detrended.nc')
    
        
        EOF1ij.to_csv(path+'/'+self.model+'_EOF1ij_ssp585_reg_patterns_2050-2099_DJF_detreded.csv', float_format='%g')
        EOF2ij.to_csv(path+'/'+self.model+'_EOF2ij_ssp585_reg_patterns_2050-2099_DJF_detreded.csv', float_format='%g')
        EOF3ij.to_csv(path+'/'+self.model+'_EOF3ij_ssp585_reg_patterns_2050-2099_DJF_detreded.csv', float_format='%g')
        EOF4ij.to_csv(path+'/'+self.model+'_EOF4ij_ssp585_reg_patterns_2050-2099_DJF_detreded.csv', float_format='%g')
        EOF1pij.to_csv(path+'/'+self.model+'_EOF1ij_p_ssp585_reg_patterns_2050-2099_DJF_detreded.csv', float_format='%g')
        EOF2pij.to_csv(path+'/'+self.model+'_EOF2ij_p_ssp585_reg_patterns_2050-2099_DJF_detreded.csv', float_format='%g')
        EOF3pij.to_csv(path+'/'+self.model+'_EOF3ij_p_ssp585_reg_patterns_2050-2099_DJF_detreded.csv', float_format='%g')
        EOF4pij.to_csv(path+'/'+self.model+'_EOF4ij_p_ssp585_reg_patterns_2050-2099_DJF_detreded.csv', float_format='%g')
        
        R2ij.to_csv(path+'/'+self.model+'_R2ij_ssp585_reg_patterns_2050-2099_DJF_detreded.csv', float_format='%g')
                
            
    def convert_csv_files(self,path):
        file_names_nc = csv2nc.csv_to_nc(path)
        return file_names_nc
    
    def df_to_xr(self,df_in,var):
        df = df_in.set_index(['lat','lon'])
        df = df[~df.index.duplicated(keep='first')]
        xr_out = df.to_xarray()
        xr_out['lat'].attrs={'units':'degrees', 'long_name':'Latitude'}
        xr_out['lon'].attrs={'units':'degrees', 'long_name':'Longitude'}
        xr_out[var].attrs={'units':var, 'long_name':var}
        return xr_out
                
    
    def create_x(self,i,j,dato):
        x = dato[:,i-1,j-1].values
        return x
    