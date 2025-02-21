# -*- coding: utf-8 -*-
"""
Demo for histogram layer networks (HistRes_B)
Current script is only for experiments on
single cpu/gpu. If you want to run the demo
on multiple gpus (two were used in paper), 
please contact me at jpeeples@ufl.edu 
for the parallel version of 
demo.py
@author: jpeeples
"""

## Python standard libraries
from __future__ import print_function
from __future__ import division
import numpy as np
import os

## PyTorch dependencies
import torch
import torch.nn as nn
import torch.optim as optim

## Local external libraries
from Utils.Network_functions import initialize_model, train_model,test_model
from Utils.RBFHistogramPooling import HistogramLayer as RBFHist
from Utils.LinearHistogramPooling import HistogramLayer as LinearHist
from Utils.Save_Results import save_results, save_params
from Demo_Parameters import Network_parameters
from Prepare_Data import Prepare_DataLoaders

# Comet ML logging package
try:
    from comet_ml import Experiment
except:
    Experiment = None

#Name of dataset
Dataset = Network_parameters['Dataset']

#Model(s) to be used
model_name = Network_parameters['Model_names'][Dataset]

#Number of classes in dataset
num_classes = Network_parameters['num_classes'][Dataset]
                                 
#Number of runs and/or splits for dataset
numRuns = Network_parameters['Splits'][Dataset]

#Number of bins and input convolution feature maps after channel-wise pooling
numBins = Network_parameters['numBins']
num_feature_maps = Network_parameters['out_channels'][model_name]

#Local area of feature map after histogram layer
feat_map_size = Network_parameters['feat_map_size']

# Detect if we have a GPU available
gpu_dev = Network_parameters['GPU_Num']
device = torch.device(f"cuda:{gpu_dev}" if torch.cuda.is_available() else "cpu")

#Location to store trained models
current_directory = os.path.dirname(__file__)  # abs path of runfile
final_directory = os.path.join(current_directory, Network_parameters['folder'])
Network_parameters['data_dir'] = os.path.join(current_directory, Network_parameters['data_dir'])
best_val_stats = np.zeros(numRuns)

for split in range(0, numRuns):

    if Experiment is None:
        experiment = None
    else:  # Not yet sure whether to also use this for logging HistRes inits
        if Network_parameters['histogram']:
            proj_name = 'histogram-layer'
        else:
            proj_name = 'vanilla-layer'

        experiment = Experiment(
            api_key="cf2AdIgBb4jLjQZHyCyWoo2k2",
            project_name=proj_name,
            workspace="changspencer",
        )
        experiment.set_name(f"{Dataset}-{model_name}-{split+1}")
    
    print('Starting Experiments...')
    if experiment is not None:
        experiment.log_parameters(Network_parameters)
        save_params(Network_parameters)
    else:
        save_params(Network_parameters)
    # exit()  # For debugging purposes
    
    #Keep track of the bins and widths as these values are updated each
    #epoch
    saved_bins = np.zeros((Network_parameters['num_epochs']+1,
                           numBins*int(num_feature_maps/(feat_map_size*numBins))))
    saved_widths = np.zeros((Network_parameters['num_epochs']+1,
                             numBins*int(num_feature_maps/(feat_map_size*numBins))))
    
    #Initialize histogram layer based on type
    if Network_parameters['histogram_type'] == 'RBF':
        histogram_layer = RBFHist(int(num_feature_maps/(feat_map_size*numBins)),
                                  Network_parameters['kernel_size'][model_name],
                                  num_bins=numBins,stride=Network_parameters['stride'],
                                  normalize_count=Network_parameters['normalize_count'],
                                  normalize_bins=Network_parameters['normalize_bins'])
    elif Network_parameters['histogram_type'] == 'Linear': 
        histogram_layer = LinearHist(int(num_feature_maps/(feat_map_size*numBins)),
                                  Network_parameters['kernel_size'][model_name],
                                  num_bins=numBins,stride=Network_parameters['stride'],
                                  normalize_count=Network_parameters['normalize_count'],
                                  normalize_bins=Network_parameters['normalize_bins'])
    else:
        raise RuntimeError('Invalid type for histogram layer')
    
    # Initialize the histogram model for this run - Include Comet logging here?
    model_ft, input_size = initialize_model(model_name, num_classes,
                                            Network_parameters['in_channels'][model_name],
                                            num_feature_maps, 
                                            feature_extract = Network_parameters['feature_extraction'], 
                                            histogram= Network_parameters['histogram'],
                                            histogram_layer=histogram_layer,
                                            parallel=Network_parameters['parallel'], 
                                            use_pretrained=Network_parameters['use_pretrained'],
                                            add_bn = Network_parameters['add_bn'],
                                            scale = Network_parameters['scale'],
                                            feat_map_size=feat_map_size,
                                            dataset=Dataset)

    if experiment is not None:
        experiment.set_model_graph(model_ft)

    # Send the model to GPU if available
    model_ft = model_ft.to(device)
    
    #Print number of trainable parameters
    num_params = sum(p.numel() for p in model_ft.parameters() if p.requires_grad)
    print("Number of parameters: %d" % (num_params))    
    print("Initializing Datasets and Dataloaders...")
    
    # Create training, validation, and test dataloaders
    dataloaders_dict = Prepare_DataLoaders(Network_parameters,
                                           split,
                                           input_size=input_size,
                                           comet_exp=experiment)
    
    #Save the initial values for bins and widths of histogram layer
    #Set optimizer for model
    if(Network_parameters['histogram']):
        reduced_dim = int((num_feature_maps/feat_map_size)/(numBins))
        if (Network_parameters['in_channels'][model_name]==reduced_dim):
            dim_reduced = False
            saved_bins[0,:] = model_ft.histogram_layer.centers.detach().cpu().numpy()
            saved_widths[0,:] = model_ft.histogram_layer.widths.reshape(-1).detach().cpu().numpy()
        else:
            dim_reduced = True
            saved_bins[0,:] = model_ft.histogram_layer[-1].centers.detach().cpu().numpy()
            saved_widths[0,:] = model_ft.histogram_layer[-1].widths.reshape(-1).detach().cpu().numpy()
        if(Network_parameters['add_bn']):
            optimizer_ft = optim.SGD([
                    {'params': model_ft.backbone.conv1.parameters()},
                    {'params': model_ft.backbone.bn1.parameters()},
                    {'params': model_ft.backbone.layer1.parameters()},
                    {'params': model_ft.backbone.layer2.parameters()},
                    {'params': model_ft.backbone.layer3.parameters()},
                    {'params': model_ft.backbone.layer4.parameters()},                
                    {'params': model_ft.histogram_layer.parameters(), 'lr': Network_parameters['new_lr']},
                    {'params': model_ft.fc.parameters(), 'lr': Network_parameters['new_lr']},
                    {'params': model_ft.bn_norm.parameters(), 'lr': Network_parameters['new_lr']}
                ], lr=Network_parameters['pt_lr'], momentum=Network_parameters['momentum'],
            weight_decay=Network_parameters['wgt_decay'])
        else:
            optimizer_ft = optim.SGD([
                    {'params': model_ft.backbone.conv1.parameters()},
                    {'params': model_ft.backbone.bn1.parameters()},
                    {'params': model_ft.backbone.layer1.parameters()},
                    {'params': model_ft.backbone.layer2.parameters()},
                    {'params': model_ft.backbone.layer3.parameters()},
                    {'params': model_ft.backbone.layer4.parameters()},                
                    {'params': model_ft.histogram_layer.parameters(), 'lr': Network_parameters['new_lr']},
                    {'params': model_ft.fc.parameters(), 'lr': Network_parameters['new_lr']},
                ], lr=Network_parameters['pt_lr'], momentum=Network_parameters['momentum'],
            weight_decay=Network_parameters['wgt_decay'])
    else:
        saved_bins = None
        saved_widths = None
        dim_reduced = False
        optimizer_ft = optim.SGD([
                {'params': model_ft.conv1.parameters()},
                {'params': model_ft.bn1.parameters()},
                {'params': model_ft.layer1.parameters()},
                {'params': model_ft.layer2.parameters()},
                {'params': model_ft.layer3.parameters()},
                {'params': model_ft.layer4.parameters()},
                {'params': model_ft.fc.parameters(), 'lr': Network_parameters['new_lr']},
            ], lr=Network_parameters['pt_lr'], momentum = Network_parameters['momentum'],
            weight_decay=Network_parameters['wgt_decay'])
   
    # Setup the loss fxn
    criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.StepLR(optimizer_ft,
                                          step_size=Network_parameters['step_size'],
                                          gamma= Network_parameters['gamma'])

    # first_step = Network_parameters['step_size']
    # sched_milestones = [
    #     first_step,
    #     first_step + first_step // 2,
    #     first_step + first_step // 2 + first_step // 4
    # ]
    # scheduler = optim.lr_scheduler.MultiStepLR(optimizer_ft,
    #                                       milestones=sched_milestones,
    #                                       gamma=Network_parameters['gamma'])

    # if Dataset == 'cifar10':  # TODO - Maybe try this later...
    #     scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer_ft, T_max=125)
    #     print("SCHEDULER IS COSINE ANNEALING WITH TMAX 125")
    
    # Train and evaluate
    train_dict = train_model(
            model_ft, dataloaders_dict, criterion, optimizer_ft, device,
            saved_bins=saved_bins, saved_widths=saved_widths, histogram=Network_parameters['histogram'],
            num_epochs=Network_parameters['num_epochs'], scheduler=scheduler,
            dim_reduced=dim_reduced,
            comet_exp=experiment)
    test_dict = test_model(dataloaders_dict['test'], model_ft, device, comet_exp=experiment)

    best_val_stats[split] = max(train_dict["val_acc_track"])
    
    # Save results
    if(Network_parameters['save_results']):
        save_results(train_dict,test_dict,split,Network_parameters,num_params)
        del train_dict,test_dict
        torch.cuda.empty_cache()
        
    if(Network_parameters['histogram']):
        print('**********Run ' + str(split + 1) + ' For ' + Network_parameters['hist_model'] + ' Finished**********') 
    else:
        print('**********Run ' + str(split + 1) + ' For GAP_' + model_name + ' Finished**********') 

experiment.end()
print("\nBest Val-Acc Stats (mean, dev): {:.2f}, {:.2f}".format(
            100*best_val_stats.mean(), 100*best_val_stats.std()))

