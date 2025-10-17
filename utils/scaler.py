#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  9 11:38:56 2022

@author: lollier


"""
# +---------------------------------------------------------------------------------------+ #
# |                                                                                       | #
# |                                         Scaler                                        | #
# |                                                                                       | #
# +---------------------------------------------------------------------------------------+ #
import numpy as np
import xarray as xr
import dask as da


# +---------------------------------------------------------------------------------------+ #
# |                                                                                       | #
# |                             THIS FILE SHOULD NOT BE MODIFIED                          | #
# |                   UTILS FUNCTIONS TO PROCESS DATA, SAVE AND LOAD RESULT               | #
# |                                                                                       | #
# +---------------------------------------------------------------------------------------+ #

class StandardScaler:

    def __init__(self, mean=None, std=None, epsilon=1e-7):
        """Standard Scaler.
        Rewrite for numpy (inspired from https://gist.github.com/farahmand-m/8a416f33a27d73a149f92ce4708beb40)
        
        The class can be used to normalize PyTorch Tensors using native functions. The module does not expect the
        tensors to be of any specific shape; as long as the features are the last dimension in the tensor, the module
        will work fine.
        :param mean: The mean of the features. The property will be set after a call to fit.
        :param std: The standard deviation of the features. The property will be set after a call to fit.
        :param epsilon: Used to avoid a Division-By-Zero exception.
        """
        self.mean = mean
        self.std = std
        self.epsilon = epsilon

    def fit(self, values):
        dims = tuple(range(len(values.shape) - 1))
        self.mean = np.nanmean(values, axis=dims)
        self.std = np.nanstd(values, axis=dims)

    def transform(self, values):
        return (values - self.mean) / (self.std + self.epsilon)

    def fit_transform(self, values):
        self.fit(values)
        return self.transform(values)
    

class StandardScaler_xarray:
    def __init__(self, mean=None, std=None, epsilon=1e-7):
        """
        Standard Scaler for xarray DataArrays.
        
        The scaler normalizes `xarray.DataArray` objects by their mean and standard deviation along specified dimensions.
        
        :param mean: Mean of the features. Set after calling `fit`.
        :param std: Standard deviation of the features. Set after calling `fit`.
        :param epsilon: Small value to avoid division by zero.
        """
        self.mean = mean
        self.std = std
        self.epsilon = epsilon

    def fit(self, data_array, dims_first=True):
        if dims_first :
            dims = tuple(range(1,len(data_array.shape)))
        else :
            dims=None
        self.mean = da.array.nanmean(data_array,axis=dims).compute()
        self.std = da.array.nanstd(data_array,axis=dims).compute()

    def transform(self, data_array, dims_first=True):

        if self.mean is None or self.std is None:
            raise ValueError("The scaler has not been fitted yet. Call `fit` or `fit_transform` first.")
        
        if dims_first:
            #mdr la ligne est absolument ignoble à lire, mais c'est compact jsuis trop heureux
            return (data_array - self.mean[:,*[np.newaxis for i in range(len(data_array.shape)-1)]]) / (self.std[:,*[np.newaxis for i in range(len(data_array.shape)-1)]] + self.epsilon)
        else :
            return (data_array - self.mean) / (self.std + self.epsilon) #c'est pas si propre du coup 


    def fit_transform(self, data_array, dims_first=True):

        self.fit(data_array, dims_first=dims_first)
        return self.transform(data_array, dims_first=dims_first)

    

class MinMaxScaler:

    def __init__(self, maximum=None, minimum=None):
        """Standard Scaler.
        Write for numpy (inspired from https://gist.github.com/farahmand-m/8a416f33a27d73a149f92ce4708beb40)
        The class can be used to normalize PyTorch Tensors using native functions. The module does not expect the
        tensors to be of any specific shape; as long as the features are the last dimension in the tensor, the module
        will work fine.
        :param max: The max of the features. The property will be set after a call to fit.
        :param min: The min of the features. The property will be set after a call to fit.
        """
        self.maximum = maximum
        self.minimum = minimum

    def fit(self, values):
        dims = tuple(range(len(values.shape) - 1))
        self.maximum = np.nanmax(values, axis=dims)
        self.minimum = np.nanmin(values, axis=dims)

    def transform(self, values):
        return (values - self.minimum) / (self.minimum - self.minimum)

    def fit_transform(self, values):
        self.fit(values)
        return self.transform(values)


    
#########################################Scaler######################################

#c'est clean sauf le clipping
class scaler():
    """
    scaler for datasets, can apply 3 differents scalers min_max and z-score(standard)
    """
    
    def __init__(self, mode='standard',data_format='numpy',channel_pos=-1):
        """
        Parameters
        ----------
        mode : str, optional
            min_max or standard. The default is 'standard'.
        """
        
        if mode in ('min_max', 'standard', 'clipping'):
            self.mode = mode
        else:
            raise ValueError("\"{}\" is not a valid mode for "
                             "splitting. Only \"min_max\" , \"standard\"and "
                             "\"clipping\" are allowed.".format(mode))   
        if data_format in ('numpy', 'xarray'):
            self.data_format = data_format
        else:
            raise ValueError("\"{}\" is not a valid mode for "
                             "splitting. Only \"numpy\" , \"xarray\" are allowed.".format(data_format))  
        if mode=='standard':
            if data_format=='numpy':
                self.sc=StandardScaler()
            else :
                self.sc=StandardScaler_xarray()
        elif mode=='min_max':
            self.sc=MinMaxScaler()

    

    
if __name__=="__main__":
    
    path="/home/luther/Documents/npy_data/"


    physics=xr.open_dataset(path+"physics/physics_completion/processed_1998_2023_reanalysis_100km_daily_seaice.nc", chunks={'time': 100}).to_array(dim='variables').data
    scaler=StandardScaler_xarray()
    physics_scaled=scaler.fit_transform(physics)
    
