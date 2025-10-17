#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  3 15:13:27 2022

@author: lollier

model config file 

Instantiate all the configs and parameters. This is the only file that should be modified to reproduce the experiments.
All configuration files can be named and placed in the configs directory, then run_configs.sh should be executed.
"""
# +---------------------------------------------------------------------------------------+ #
# |                                                                                       | #
# |                                         CONFIG                                        | #
# |                                                                                       | #
# +---------------------------------------------------------------------------------------+ #



######################################### DATALOADER ##########################################


path=""

dataloader_config={'dataset_path_inputs':path+"inputs.nc", # for variables names in the netcdf check the dataloaders
                   'dataset_path_chl_psc':path+"chl_psc.nc", # for variables names in the netcdf check the dataloaders
                   'hplc_path':False, #path+"insitu/hplc_merged_new_glob_100km.csv",
                   'transform':None,
                   'batch_size': 16,
                   'norm_chl':True,
                   'norm_mode':'standard',
                   'log_chl':True,
                   'anomalies':0, #0 or False or None for classic behavior, 1 for chl anomalies, 2 for physics anomalies --> corresponds to a different set of experiments, here 0 is enough
                   'completion':'zeros',
                   'stations_coord':{'SWAtl':(-55,-43),# stations withdrawed of the dataset for control
                                     'NAtl':(-37,36),
                                     'NPac':(156,24),
                                     'SIO':(60,-32),
                                     'SCTR':(80,-3)}}


########################################### TRAIN #############################################


train_config = {
    'nb_epochs' : 300, # arbitrary high
    'checkpoints_path': '', 
    'verbose': 0.,
    'checkpoint':None, #save the weights every x epochs
    'name':'SmaAt_completion_avw_cmems_0',
    'patience_early_stopping':50, #needs to be >>> patience of lr scheduler
    'delta_early_stopping':0.00000001,
    'device':0,
    'nb_training':1,
    'optim':False, #if True, trainer will return mean valid loss and std valid loss for optimization
    'comment':''
}

########################################### Model #############################################

        
model_config = {
    'in_channels' : 12,
    'model':'SmaAt-UNet',
    'depth':5,
    'merge_mode':'concat',
    'activation':'SiLU',
    'weight_load':None,
    'freeze_key':'0',
    'kernels_per_layer':2,
    'chl':0, # 0: predict both CHL and PSC (keeps compatibility with previous configurations)
# 1: predict PSC only (optionally using CHL as input, depending on the dataloader)
# 2: predict CHL only (Marina/Stéphane study)
# 3: predict CHL only, with n_classes=2 so that the output represents both sign and amplitude

    'nb_layers':64,
    'time2vec':False,
}


########################################### criterion config #############################################


criterion_config = {
    'MSE_psc' : 1., # eventual features and weights of losses (put None instead of 0 if you don't want the loss to be compute)
    'MSE_chl':1.,
    'Under_chl':0.,
    'KL_div':0.,
    'quantile_loss':0.,
    'HingeLoss':0.,
    'details':True
}

########################################### criterion config #############################################

optimizer_config = {
    'optimizer' : 'AdamW', # Adam, AdamW, SGD, SparseAdam
    'learning_rate' : 0.001
}

######################################### SCHEDULER ###########################################


scheduler_config = {
    'scheduler': 'ROP', # ROP or ELR, OC not available currently
    'mode': 'min', # we want to detect a decrease and not an increase. 
    'factor': 0.75, # when loss has stagnated for too long, new_lr = factor*lr
    'patience': 10,# how long to wait before updating lr
    'threshold': 0.000000001, 
    'max_lr':0.001,
    'verbose':True,
    'steps_per_epoch':None,#1+(int((dataloader_config['split_index']['train'][1]-dataloader_config['split_index']['train'][0]+1)*46/dataloader_config['batch_size']))
}
    
