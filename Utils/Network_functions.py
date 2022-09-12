# -*- coding: utf-8 -*-
"""
Created on Tue Aug  6 10:26:08 2019

@author: jpeeples
"""
## Python standard libraries
from __future__ import print_function
from __future__ import division
import numpy as np
import time
import copy

## PyTorch dependencies
import torch
import torch.nn as nn
from torchvision import models

## Local external libraries
from Utils.Histogram_Model import HistRes
from barbar import Bar
from PIL import Image


def train_model(model, dataloaders, criterion, optimizer, device,
                          saved_bins=None, saved_widths=None, histogram=True,
                          num_epochs=25, scheduler=None, dim_reduced=True,
                          comet_exp=None):
    since = time.time()

    val_acc_history = []
    train_acc_history = []
    train_error_history = []
    val_error_history = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    train_dataloader = dataloaders['train']
    val_dataloader = dataloaders['val']
    
    if comet_exp is not None:
        def index_to_example(index):
            if not hasattr(val_dataloader.dataset, 'texture') and \
               not hasattr(val_dataloader.dataset, 'texture_dir'):
                # copied from comet.com tutorial "Image Data"
                image_array = val_dataloader.dataset.dataset[index][0]
                image_name = f"val-confusemat-{index:5d}.png"
                results = comet_exp.log_image(image_array, name=image_name)
        
                # Return sample, assetId (index is added automatically)
                return {"sample": image_name, "assetId": results["imageId"]}
            else:
                dir_name = val_dataloader.dataset.files[index]['img']
                f_name = dir_name.split('/')[-1].split('.')[0]
                img = Image.open(dir_name).convert('RGB')

                image_name = f"val-matrix-sample-{f_name}.png"
                results = comet_exp.log_image(img, name=image_name)

                return {"sample": dir_name, "assetId": results["imageId"]}

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        #! ---TRAINING PHASE--- !#
        with None if comet_exp is None else comet_exp.train() as exp:
            model.train()  # Set model to training mode
            
            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for idx, (inputs, labels, index) in enumerate(Bar(train_dataloader)):
                inputs = inputs.to(device)
                labels = labels.to(device)
                index = index.to(device)
    
                # zero the parameter gradients
                optimizer.zero_grad()
    
                # forward
                # track history if only in train
                with torch.set_grad_enabled(True):
                    # Get model outputs and calculate loss
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
    
                    _, preds = torch.max(outputs, 1)
    
                    # backward + optimize only in training phase
                    loss.backward()
                    optimizer.step()

                    # if exp is not None:
                    #     comet_exp.log_metric(loss.item(), "batch_loss")
    
                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
        
            epoch_loss = running_loss / (len(dataloaders['train'].sampler))
            epoch_acc = running_corrects.double() / (len(dataloaders['train'].sampler))
            
            train_error_history.append(epoch_loss)
            train_acc_history.append(epoch_acc)
            if(histogram):
                if dim_reduced:
                    #save bins and widths
                    saved_bins[epoch+1,:] = model.histogram_layer[-1].centers.detach().cpu().numpy()
                    saved_widths[epoch+1,:] = model.histogram_layer[-1].widths.reshape(-1).detach().cpu().numpy()
                else:
                    #save bins and widths
                    saved_bins[epoch+1,:] = model.histogram_layer.centers.detach().cpu().numpy()
                    saved_widths[epoch+1,:] = model.histogram_layer.widths.reshape(-1).detach().cpu().numpy()

                print(saved_bins.shape)
                # if exp is not None:
                #     comet_exp.log_others({
                #         "histo_bins": saved_bins[epoch + 1, :],
                #         "histo_widths": saved_widths[epoch + 1, :]
                #     })

            print('\ntrain Loss: {:.4f} Acc: {:.4f}\n'.format(epoch_loss, epoch_acc))
            if exp is not None:
                comet_exp.log_metrics({
                    "epoch_loss": epoch_loss,
                    "epoch_acc": epoch_acc
                }, epoch=epoch)

        #! ---VALIDATION PHASE--- !#
        with None if comet_exp is None else comet_exp.validate() as exp:
            model.eval()   # Set model to evaluate mode
            
            running_loss = 0.0
            running_corrects = 0
            running_preds = []
            running_targets = []

            # Iterate over data.
            for idx, (inputs, labels, index) in enumerate(Bar(val_dataloader)):
                inputs = inputs.to(device)
                labels = labels.to(device)
                index = index.to(device)
    
                # zero the parameter gradients
                optimizer.zero_grad()
    
                # forward
                # track history if only in train
                with torch.set_grad_enabled(False):
                    # Get model outputs and calculate loss
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
    
                    _, preds = torch.max(outputs, 1)

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                running_preds.extend(preds.cpu().detach().flatten().numpy())
                running_targets.extend(labels.cpu().flatten().numpy())
        
            epoch_loss = running_loss / (len(val_dataloader.sampler))
            epoch_acc = running_corrects.double() / (len(val_dataloader.sampler))

            # deep copy the model
            if epoch_acc > best_acc:
                best_epoch = epoch
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                print("Model Deep-copied")

            if exp is not None:
                comet_exp.log_confusion_matrix(
                    y_true=running_targets,
                    y_predicted=running_preds,
                    index_to_example_function=index_to_example,
                    file_name=f'val_confusion_mat.json',
                    step=epoch
                )

            val_error_history.append(epoch_loss)
            val_acc_history.append(epoch_acc)
            print('\nval Loss: {:.4f} Acc: {:.4f}\n'.format(epoch_loss, epoch_acc))

            if exp is not None:
                comet_exp.log_metrics({
                    "epoch_loss": epoch_loss,
                    "epoch_acc": epoch_acc
                }, epoch=epoch)

        if scheduler is not None:
            prev_lr = scheduler.state_dict()['_last_lr'][-1]
            scheduler.step()
            # Reload the best model weights when stepping the LR down
            # Start the model back at its best location so far...
            scheduler_state = scheduler.state_dict()
            if (prev_lr - scheduler_state['_last_lr'][-1]) > 1e-8:
                print('\nStepping Down LR...Reload best model\n')
                model.load_state_dict(best_model_wts)

            if comet_exp is not None:
                comet_exp.log_others(scheduler.state_dict())

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc), end='\n\n')

    if comet_exp is not None:
        comet_exp.log_confusion_matrix(
            y_true=running_targets,
            y_predicted=running_preds,
            index_to_example_function=index_to_example,
            file_name=f'final_val_confusion_mat.json'
        )

    # load best model weights
    model.load_state_dict(best_model_wts)
    
    #Returning error (unhashable), need to fix
    train_dict = {'best_model_wts': best_model_wts, 'val_acc_track': val_acc_history, 
                  'val_error_track': val_error_history,'train_acc_track': train_acc_history, 
                  'train_error_track': train_error_history,'best_epoch': best_epoch, 
                  'saved_bins': saved_bins, 'saved_widths': saved_widths}
    
    return train_dict

def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False
          
def test_model(dataloader, model, device, comet_exp=None):
    #Initialize and accumalate ground truth, predictions, and image indices
    GT = np.array(0)
    Predictions = np.array(0)
    Index = np.array(0)
    
    running_corrects = 0
    model.eval()
    
    # Iterate over data
    print('Testing Model...')
    with torch.no_grad():
        for idx, (inputs, labels,index) in enumerate(Bar(dataloader)):
            inputs = inputs.to(device)
            labels = labels.to(device)
            index = index.to(device)
    
            # forward
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
    
            #If test, accumulate labels for confusion matrix
            GT = np.concatenate((GT,labels.detach().cpu().numpy()),axis=None)
            Predictions = np.concatenate((Predictions,preds.detach().cpu().numpy()),axis=None)
            Index = np.concatenate((Index,index.detach().cpu().numpy()),axis=None)
            
        
            running_corrects += torch.sum(preds == labels.data)

    test_acc = running_corrects.double() / (len(dataloader.sampler))
    print('Test Accuracy: {:4f}'.format(test_acc))
    
    test_dict = {'GT': GT[1:], 'Predictions': Predictions[1:], 'Index':Index[1:],
                 'test_acc': np.round(test_acc.cpu().numpy()*100,2)}
    
    return test_dict
       
def initialize_model(model_name, num_classes,in_channels,out_channels, 
                     feature_extract=False, histogram=True,histogram_layer=None,
                     parallel=True, use_pretrained=True,add_bn=True,scale=5,
                     feat_map_size=4, dataset=None):
    # Initialize these variables which will be set in this if statement. Each of these
    #   variables is model specific.
    model_ft = None
    input_size = 0
    if(histogram):
        # Initialize these variables which will be set in this if statement. Each of these
        # variables is model specific.
        model_ft = HistRes(histogram_layer,parallel=parallel,
                           model_name=model_name,add_bn=add_bn,scale=scale,
                           use_pretrained=use_pretrained,
                           dataset=dataset)
        set_parameter_requires_grad(model_ft.backbone, feature_extract)
        
        #Reduce number of conv channels from input channels to input channels/number of bins*feat_map size (2x2)
        reduced_dim = int((out_channels/feat_map_size)/(histogram_layer.numBins))
        if (in_channels==reduced_dim): #If input channels equals reduced/increase, don't apply 1x1 convolution
            model_ft.histogram_layer = histogram_layer
        else:
            conv_reduce = nn.Conv2d(in_channels,reduced_dim,(1,1))
            model_ft.histogram_layer = nn.Sequential(conv_reduce,histogram_layer)
        if(parallel):
            num_ftrs = model_ft.fc.in_features*2
        else:
            num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    # Baseline model
    else:
        if model_name == "resnet18":
            """ Resnet18
            """
            model_ft = models.resnet18(pretrained=use_pretrained)
            set_parameter_requires_grad(model_ft, feature_extract)
            num_ftrs = model_ft.fc.in_features
            model_ft.fc = nn.Linear(num_ftrs, num_classes)
            input_size = 224
            
            if dataset == 'mnist' or dataset == 'fashionmnist':
                input_size = 28
                model_ft.conv1 = nn.Conv2d(1, model_ft.conv1.out_channels,
                        kernel_size=3, stride=1, padding=1, bias=False)
    
        elif model_name == "resnet50":
            """ Resnet50
            """
            model_ft = models.resnet50(pretrained=use_pretrained)
            set_parameter_requires_grad(model_ft, feature_extract)
            num_ftrs = model_ft.fc.in_features
            model_ft.fc = nn.Linear(num_ftrs, num_classes)
            input_size = 224

            if dataset == 'mnist' or dataset == 'fashionmnist':
                input_size = 28
                model_ft.conv1 = nn.Conv2d(1, model_ft.conv1.out_channels,
                        kernel_size=3, stride=1, padding=1, bias=False)
    # TODO - Change this to be dependent on the dataset parameters 
    if dataset == 'mnist' or dataset == 'fashionmnist':
        input_size = 27
    return model_ft, input_size

