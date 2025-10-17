
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  3 15:42:54 2022

@author: lollier
"""



# +---------------------------------------------------------------------------------------+ #
# |                                                                                       | #
# |                                         MAIN                                          | #
# |                                                                                       | #
# +---------------------------------------------------------------------------------------+ #

from torch import load, save 
import os
from Trainer import Trainer
from config import model_config, dataloader_config, train_config, criterion_config, scheduler_config, optimizer_config


import numpy as np 


import sys
sys.path.append("/home/luther/Documents/scripts_training/models/")
sys.path.append("/usr/home/lollier/Documents/scripts_training/models/")


from models.UNet import UNet
from models.UNet_DSC import UNet_DSC
from models.UNet_CBAM import UNet_CBAM
from models.SmaAt_UNet import SmaAt_UNet
from utils.functions import timer, saving_tool#, process_results

import torch
import torch.nn as nn
# +---------------------------------------------------------------------------------------+ #
# |                                                                                       | #
# |                             THIS FILE SHOULD NOT BE MODIFIED                          | #
# |                   ALL HYPER PARAMETERS SHOULD BE CONFIGURED IN config.py              | #
# |                                                                                       | #
# +---------------------------------------------------------------------------------------+ #

"""
This function takes hyperparameters from config.py. 
It creates an object from the Model class and then uses it to define an object 
from the Trainer class.
The training is launched by the call to the run() method from the trainer object. 
This call is inside a try: block in order to handle exceptions.
For now, the only exception handled is a KeyboardInterrupt: 
the current network will be saved.
"""

def main(model_config, dataloader_config, train_config, criterion_config, optimizer_config, scheduler_config):

    print('Building Model...')
    
    """
    Architecture configuration
    """
    
    ######################parameters####################################
    if 'nb_layers' in model_config.keys():
        nb_layers=model_config['nb_layers']
    else : nb_layers=64
    
    if model_config['activation'] in ('ReLU', 'SiLU'):
        if model_config['activation']=='ReLU' :
            activation = nn.ReLU()
        else : 
            activation = nn.SiLU()
    else:
        raise ValueError("\"{}\" is not a valid mode for "
                         "activation. Only \"SiLU\" and "
                         "\"ReLU\" are allowed.".format(model_config['activation']))

    if 'time2vec' in model_config.keys() and model_config['time2vec']:
        time_encoded=model_config['time2vec']
    else :
        time_encoded=False
        

    n_classes=model_config.get('n_classes', 4)
            
#Model configurations available
########################################################################
    if model_config['model']=='UNet':
               
        net=UNet(in_channels=model_config['in_channels'],
                 out_channels=model_config['n_classes'],
                  depth=model_config['depth'],
                  start_filts=nb_layers,
                  merge_mode=model_config['merge_mode'], 
                  activation=model_config['activation'],
                  chl=model_config['chl'])
        
    elif model_config['model']=='UNet_DSC':
        
        net=UNet_DSC(in_channels=model_config['in_channels'],
                     out_channels=model_config['n_classes'],
                     depth=model_config['depth'],
                     start_filts=nb_layers,
                     merge_mode=model_config['merge_mode'], 
                     activation=model_config['activation'],
                     chl=model_config['chl'],
                     kernels_per_layer=model_config['kernels_per_layer'])
        
    elif model_config['model']=='UNet_CBAM':
                
        net=UNet_CBAM(n_channels=model_config['in_channels'],
                     n_classes=model_config['n_classes'],#tjrs 4 pour PSC et CHL
                     depth=model_config['depth'],
                     activation=activation,
                     nb_layers=nb_layers,
                     chl=model_config['chl'],)
        
    elif model_config['model']=='SmaAt-UNet':

        net=SmaAt_UNet(n_channels=model_config['in_channels'], 
                       n_classes=n_classes,
                       activation=activation,
                       kernels_per_layer=model_config['kernels_per_layer'],
                       chl=model_config['chl'],
                       nb_layers=nb_layers,
                       time_encoded=time_encoded)
        
    elif model_config['model']=='SmaAt-BAM':

        net=SmaAt_UNet(n_channels=model_config['in_channels'], 
                       n_classes=n_classes,
                       activation=activation,
                       kernels_per_layer=model_config['kernels_per_layer'],
                       chl=model_config['chl'],
                       nb_layers=nb_layers,
                       time_encoded=time_encoded)

    else :
        raise ValueError("\"{}\" is not a valid mode for model.".format(model_config['model']))
    

    if model_config['weight_load'] is not(None):
        net.load_state_dict(load(model_config['weight_load']))
        print("\nweights loaded \n")
    
    print('\nSet up the saving folder')
    #initiate the saver and create a directory to store results, copy dataloader and architecture in the directory
    saver=saving_tool(train_config['checkpoints_path'], train_config['name'], model_config['model'])
    train_config['checkpoints_path']=saver.result_dir #update the saving directory
    trainer = Trainer(net, train_config, dataloader_config, criterion_config, optimizer_config, scheduler_config)
    
    try:
        print(trainer)
        trainer.run()
    except KeyboardInterrupt:

        filename = 'interrupted_'+ train_config['name']
        path = os.path.join(train_config['checkpoints_path'],filename)
        save(trainer.model.net.state_dict(), path)
        print()
        print(80*'_')
        print('Training Interrupted')
        print('Current State saved.')
    
    path = os.path.join(train_config['checkpoints_path'],'training_finished_'+train_config['name']+'_.pt')
    save(trainer.model.net.state_dict(),path)
    
    print('\nTraining finished, running prediction on test set..')
    trainer.model.net.load_state_dict(load(os.path.join(train_config['checkpoints_path'], 
                                                        train_config['name'] + 'best_valid_loss.pt')))
    print("\nweights loaded \n")
    loss, psc_predictions, chl_predictions=trainer.test_step()
    saver.save_pred(chl_predictions, psc_predictions, train_config['checkpoints_path'])

    print(f'\nTest loss : {loss}')
    print('\nSaving config and results .....')    
    
    saver.save_glob(loss, model_config['model'], trainer.state['epoch'],
                    trainer.state['epoch']-trainer.early_stopping.counter,
                    (optimizer_config['optimizer'],scheduler_config['scheduler'],optimizer_config['learning_rate']),
                    sum(p.numel() for p in net.parameters()),train_config['comment'])
                    
    dataloader_config.pop('transform', None)
    
    saver.save_config([model_config, dataloader_config, train_config, criterion_config, scheduler_config, optimizer_config],
                train_config['checkpoints_path'],
                train_config['name']+'_config')

        

    

if __name__ == '__main__':


    if 'nb_training' in train_config.keys():
        nb_training=train_config['nb_training']-1
    else :
        nb_training=0
    
    print(f'\nTraining {nb_training+1}\n')

    main(model_config, dataloader_config, train_config, criterion_config, optimizer_config, scheduler_config)


    train_config['checkpoints_path']=train_config['checkpoints_path'][:-len(train_config['name'])]
    
    for i in range(nb_training):
        print(f'\nTraining {i+1}\n')

        train_config['name']+='_'+str(i)
        dataloader_config['transform']=None
        main(model_config, dataloader_config, train_config, criterion_config, optimizer_config, scheduler_config)
        train_config['checkpoints_path']=train_config['checkpoints_path'][:-len(train_config['name'])]

                
