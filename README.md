![GitHub last commit](https://img.shields.io/github/last-commit/Sma6500/Chl_global_completion) 
![GitHub top language](https://img.shields.io/github/languages/top/Sma6500/Chl_global_completion)
![GitHub language count](https://img.shields.io/github/languages/count/Sma6500/Chl_global_completion)


# SmaAT_UNet for gap completion of satellite chlorophyll-a and pigment based phytoplankton classes.


SmaAt_UNet is a neural network to
reconstruct missing data in satellite observations which is described in the following open access paper:
https://doi.org/10.48550/arXiv.2007.04417

The reconstruction use physical variables from GLorysV12 and Era5 reanalysis 


## Installation

Python > 3.7 with the modules:
* numpy (https://docs.scipy.org/doc/numpy/user/install.html)
* xarray (https://docs.xarray.dev/en/stable/)
* pytorch (https://pytorch.org/get-started/locally/)
* dask (https://docs.dask.org/en/stable/)

Tested versions:

* Python 3.11.8


You can install those packages either with `pip3` or with `conda`.


## Documentation



## Input format

The input data should be in netCDF with the variables:
* `lon`: longitude (degrees East)
* `lat`: latitude (degrees North)
* `time`: time (days since 1900-01-01 00:00:00)
* `mask`: boolean mask where true means the data location is valid

To update.. 


This is the example output from `ncdump -h`:

```

```



## Running SmaAt_UNet







## Example results

North-South asymmetry in subtropical phytoplankton response to recent warming