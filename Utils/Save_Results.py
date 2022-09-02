# -*- coding: utf-8 -*-
"""
Created on Sat Feb  1 15:39:28 2020
Save results from training/testing model
@author: jpeeples
"""
## Python standard libraries
from __future__ import print_function
from __future__ import division
import numpy as np
import pickle
import os

## PyTorch dependencies
import torch

def save_results(train_dict,test_dict,split,Network_parameters,num_params):
    if(Network_parameters['histogram']):
        if(Network_parameters['parallel']):
            filename = (Network_parameters['folder'] + '/' + Network_parameters['mode'] 
                        + '/' + Network_parameters['Dataset'] + '/' 
                        + Network_parameters['hist_model'] +'_'+ 
                        Network_parameters['histogram_type']
                        +'/Parallel/Run_' + str(split + 1) + '/')
        else:
            filename = (Network_parameters['folder'] + '/' + Network_parameters['mode'] 
                        + '/' + Network_parameters['Dataset'] + '/' 
                        + Network_parameters['hist_model'] +'_'+ 
                         Network_parameters['histogram_type']
                        + '/Inline/Run_' + str(split + 1) + '/')
    #Baseline model
    else:
        filename = (Network_parameters['folder'] + '/'+ Network_parameters['mode'] 
                    + '/' + Network_parameters['Dataset'] + '/GAP_' +
                    Network_parameters['Model_names'][Network_parameters['Dataset']] + '/Run_' + str(split + 1) + '/')    
        
    if not os.path.exists(filename):
        os.makedirs(filename)
    with open((filename + 'Test_Accuracy.txt'), "w") as output:
        output.write(str(test_dict['test_acc']))
    if not os.path.exists(filename):
        os.makedirs(filename)
    with open((filename + 'Num_parameters.txt'), "w") as output:
        output.write(str(num_params))
    
    #Save training and testing dictionary, save model using torch
    torch.save(train_dict['best_model_wts'], filename + 'Best_Weights.pt')
   
    #Remove model from training dictionary
    train_dict.pop('best_model_wts')
    output_train = open(filename + 'train_dict.pkl','wb')
    pickle.dump(train_dict,output_train)
    output_train.close()
    
    output_test = open(filename + 'test_dict.pkl','wb')
    pickle.dump(test_dict,output_test)
    output_test.close()


def save_params(Network_parameters):
    '''
    Print the network parameters to stdout and write to a file
    in the Saved_Results subfolders.
    '''
    if(Network_parameters['histogram']):
        if(Network_parameters['parallel']):
            filename = "{}/{}/{}/{}_{}/Parallel/".format(
                           Network_parameters['folder'],
                           Network_parameters['mode'], 
                           Network_parameters['Dataset'],
                           Network_parameters['hist_model'],
                           Network_parameters['histogram_type']
                       )
        else:
            filename = "{}/{}/{}/{}_{}/Inline/".format(
                           Network_parameters['folder'],
                           Network_parameters['mode'], 
                           Network_parameters['Dataset'],
                           Network_parameters['hist_model'],
                           Network_parameters['histogram_type']
                       )
    #Baseline model
    else:
        filename = "{}/{}/{}/GAP_{}/".format(
                        Network_parameters['folder'],
                        Network_parameters['mode'],
                        Network_parameters['Dataset'],
                        Network_parameters['Model_names'][Network_parameters['Dataset']]
                   )    

    if not os.path.exists(filename):
        os.makedirs(filename)

    dataset_params = ['Data_dirs', 'Model_names', 'num_classes', 'Splits']
    model_params = ['kernel_size', "in_channels", "out_channels"]
    with open(filename + "network_params.txt", "w") as out_file:
        print('Network Parameters are as follows:')
        out_file.write('Network Parameters are as follows:\n')
        for key in Network_parameters.keys():
            key_attr = getattr(Network_parameters[key], 'keys', None)
            if callable(key_attr):
                key_dict = Network_parameters[key]
                if key in dataset_params:
                    dataset = Network_parameters['Dataset']
                    print(f"   {key}: {key_dict[dataset]}")
                    out_file.write(f"   {key}: {key_dict[dataset]}\n")
                elif key in model_params:
                    model_name = Network_parameters['Model_names'][Network_parameters['Dataset']]
                    print(f"   {key}: {key_dict[model_name]}")
                    out_file.write(f"   {key}: {key_dict[model_name]}\n")
                else:
                    print(f"   {key}:")
                    out_file.write(f"   {key}:\n")
                    for sub_key in key_dict.keys():
                        print(f"      {sub_key}: {key_dict[sub_key]}")
                        out_file.write(f"      {sub_key}: {key_dict[sub_key]}\n")
            else:
                print(f"   {key}: {Network_parameters[key]}")
                out_file.write(f"   {key}: {Network_parameters[key]}\n")
    print("Save dir: " + filename)
