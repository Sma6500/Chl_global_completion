#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 15 15:34:22 2023

@author: lollier
"""

# +---------------------------------------------------------------------------------------+ #
# |                                                                                       | #
# |                                         Plot                                          | #
# |                                                                                       | #
# +---------------------------------------------------------------------------------------+ #

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, LogNorm, SymLogNorm
from matplotlib.cm import ScalarMappable

import numpy as np
import os
import xarray as xr
import cmocean.cm as cm
import torch

import cartopy
import cartopy.feature as cfeature
import cartopy.crs as ccrs
import xarray as xr
import numpy as np
import pymannkendall as mk
import matplotlib.pyplot as plt
import scipy as sc
from tqdm import tqdm
from scipy import signal
from matplotlib import cm as mcm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from scipy.sparse import coo_matrix
import scipy as sc
# +---------------------------------------------------------------------------------------+ #
# |                                                                                       | #
# |                             THIS FILE SHOULD NOT BE MODIFIED                          | #
# |                           PLOT FUNCTIONS TO SHOW DATA AND RESULT                      | #
# |                                                                                       | #
# +---------------------------------------------------------------------------------------+ #


"""
TO DO
--> ajouter un affichage en log, ie un param qui prend le log de la carte d'entrée mais sans changer la colorbar
--> pour les PSC gérer un triple affichage ?
         
"""

def check_shape(data):
    """
    Parameters
    ----------
    tensor : tensor or array
        check the size and the format of inputs and return the axes for plot.

    Returns
    -------
    tensor : TYPE
        DESCRIPTION.
    axes_label : TYPE
        DESCRIPTION.

    """
    if type(data) not in [np.ndarray, np.array, torch.Tensor]:
        raise ValueError('type not valid')
    
    if type(data)==torch.Tensor:
        data=data.numpy()
        
    if data.shape==(120,360):
        axes_label=[-179.5,
                    179.5,
                    59.5,
                    -59.5]   
        return data, axes_label
    
    if data.shape==(100,360):
        axes_label=[-179.5,
                    179.5,
                    49.5,
                    -49.5]   
        return data, axes_label
    
    if data.shape==(180,360):
        axes_label=[-179.5,
                    179.5,
                    89.5,
                    -89.5]   
        return data, axes_label

        
    if data.shape==(4320,8640):
        axes_label=[-179.97,
                    179.97,
                    89.97,
                    -89.97]
        return data, axes_label
    
    if data.shape==(481,1440):
        axes_label=[0,
                   360,
                   59.97,
                   -59.97]
        return data, axes_label
    
    if data.shape==(720,1440):
        axes_label=[-179.97,
                    179.97,
                    89.97,
                    -89.97]
        return data, axes_label
    
    
    else :
        raise ValueError('shape not valid')
        


def imshow_area(array, cmap='jet', fig=None, ax=None, 
                vmin=None, vmax=None, log=False, title=False,
                colorbar=True, symlog=False, save=None,
                contour=None):

    if (fig is None and not(ax is None)) or (ax is None and not(fig is None)):
        raise ValueError("You need to specify both ax and fig params")
        
    array, axes_label = check_shape(array)
    
    lon = np.linspace(axes_label[0], axes_label[1], array.shape[1])
    lat = np.linspace(axes_label[2], axes_label[3], array.shape[0])
    proj = ccrs.Robinson(central_longitude=200)

    if fig is None and ax is None:
        fig = plt.figure(figsize=(20, 15), dpi=200) 
        ax = plt.subplot(1, 1, 1, projection=proj)
        show = True
    else:
        fig, ax = fig, ax
        show = False
        
    if log:
        array = np.log10(array)
        if vmin is None and vmax is None:
            vmax, vmin = np.nanmax(array), np.nanmin(array)
        cb = LogNorm(10**vmin, 10**vmax)
        
    elif symlog:
        if vmin is None and vmax is None:
            vmax, vmin = np.nanmax(array), np.nanmin(array)
        cb = SymLogNorm(0.001, vmin=vmin, vmax=vmax)

    img = ax.pcolormesh(lon, lat, 
                        np.roll(array, int(array.shape[1] / 2.25), axis=1), 
                        transform=ccrs.PlateCarree(central_longitude=200), 
                        cmap=cmap, vmin=vmin, vmax=vmax)
        
    ax.coastlines(alpha=0.5)
    gl = ax.gridlines(crs=ccrs.PlateCarree(),
                      draw_labels=True,
                      linewidth=0.5, 
                      color='gray', 
                      alpha=0.2, 
                      linestyle='solid')
    gl.xlabels_top=False
    
    ax.add_feature(cfeature.LAND, edgecolor='black', facecolor='grey', zorder=1)  
    ax.set_extent(axes_label, crs=ccrs.PlateCarree(central_longitude=200))
    
    if colorbar:
        if isinstance(colorbar, str):
            colorbar_label = colorbar
        else:
            colorbar_label = None
        if log:
            fig.colorbar(ScalarMappable(cb, cmap=cmap), ax=ax, 
                         orientation='vertical', shrink=0.4, label=colorbar_label)
        elif symlog:
            fig.colorbar(ScalarMappable(cb, cmap=cmap), ax=ax, 
                         orientation='vertical', shrink=0.4, label=colorbar_label)
        else:
            if vmin is None and vmax is None:
                vmax, vmin = np.nanmax(array), np.nanmin(array)
            cb = Normalize(vmin, vmax)
            fig.colorbar(ScalarMappable(cb, cmap=cmap), ax=ax, 
                         orientation='vertical', shrink=0.4, label=colorbar_label)
    
    if contour:
        # Extract contour levels and additional parameters from the contour argument
        levels = contour.get('levels', None)
        colors = contour.get('colors', 'k')
        linewidths = contour.get('linewidths', 1)
        linestyles = contour.get('linestyles', 'solid')
        contour_array=contour.get('contour_data',array)

        # Add contour lines
        cs = ax.contour(lon, lat, 
                        np.roll(contour_array, int(contour_array.shape[1] / 2.25), axis=1), 
                        levels=levels, colors=colors, linewidths=linewidths, 
                        linestyles=linestyles, transform=ccrs.PlateCarree(central_longitude=200))
        ax.clabel(cs, inline=True, fontsize=8, fmt='%1.1f')  # Add labels to contours
    
    ax.set(xlabel='Longitude', ylabel='Latitude')
    
    if title:
        ax.set_title(title)
    if save:
        plt.savefig(save,bbox_inches='tight')
    if show:
        plt.show()




