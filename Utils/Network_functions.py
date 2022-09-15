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

    confuse_mat_step = num_epochs // 10  # Max of 10 Valid. Confusion mat's
    confuse_mat_step = 1 if confuse_mat_step == 0 else confuse_mat_step

    if comet_exp is not None:
        conf_mat = comet_exp.create_confusion_matrix(max_categories=50)

        if saved_bins is not None and saved_widths is not None:
            with comet_exp.train() as exp:
                comet_exp.log_histogram_3d(saved_bins[0, :],
                        name=f"histo_means", step=0)
                comet_exp.log_histogram_3d(saved_widths[0, :],
                    name=f"histo_widths", step=0)

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

                if exp is not None:
                    comet_exp.log_histogram_3d(saved_bins[epoch + 1, :],
                            name=f"histo_means", step=epoch + 1)
                    comet_exp.log_histogram_3d(saved_widths[epoch + 1, :],
                        name=f"histo_widths", step=epoch + 1)

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
                val_in = inputs if idx == 0 else torch.cat((val_in, inputs))

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
            
            val_error_history.append(epoch_loss)
            val_acc_history.append(epoch_acc)

            # deep copy the model
            if epoch_acc > best_acc:
                best_epoch = epoch
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                print('\nval Loss: {:.4f} Acc: {:.4f}'.format(epoch_loss, epoch_acc))
                print(" --- Model Deep-copied --- \n")
            else:
                print('\nval Loss: {:.4f} Acc: {:.4f}\n'.format(epoch_loss, epoch_acc))

            if exp is not None:
                comet_exp.log_metrics({
                    "epoch_loss": epoch_loss,
                    "epoch_acc": epoch_acc
                }, epoch=epoch)

            if exp is not None and epoch % confuse_mat_step == 0:
                # Comet ML experiment logging
                conf_mat.compute_matrix(running_targets, running_preds,
                        images=val_in.permute((0, 2, 3, 1)),
                )

                comet_exp.log_confusion_matrix(
                        matrix=conf_mat,
                        step=epoch,
                        title=f'Validation Confusion Mat, Epoch {epoch}',
                        file_name=f'val_confusion_mat_{epoch:03d}.json'
                )

        if scheduler is not None:
            # if comet_exp is not None and comet_exp.name.startswith("cifar10"):
            #     prev_lr = scheduler._last_lr[0]
            #     scheduler.step()  # Only call stepping for the CosineAnnealing Scheduler
            # else:
            prev_lr = scheduler.state_dict()['_last_lr'][-1]
            scheduler.step()
            # Reload the best model weights when stepping the LR down
            # Start the model back at its best location so far...
            scheduler_state = scheduler.state_dict()
            if (prev_lr - scheduler_state['_last_lr'][-1]) > 1e-8:
                print('\nStepping Down LR...Reload best model\n')
                model.load_state_dict(best_model_wts)

            if comet_exp is not None:
                comet_exp.log_metric("LR", prev_lr)

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc), end='\n\n')

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
        for idx, (inputs, labels, index) in enumerate(Bar(dataloader)):
            test_in = inputs if idx == 0 else torch.cat((test_in, inputs))

            inputs = inputs.to(device)
            labels = labels.to(device)
            index = index.to(device)
    
            # forward
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
    
            #If test, accumulate labels for confusion matrix
            GT = np.concatenate((GT, labels.detach().cpu().numpy()), axis=None)
            Predictions = np.concatenate((Predictions, preds.detach().cpu().numpy()), axis=None)
            Index = np.concatenate((Index, index.detach().cpu().numpy()), axis=None)
            
            running_corrects += torch.sum(preds == labels.data)

    test_acc = running_corrects.double() / (len(dataloader.sampler))
    print('Test Accuracy: {:4f}'.format(test_acc))

    if comet_exp is not None:
        comet_exp.log_metric("test_accuracy", test_acc)
        conf_mat = comet_exp.create_confusion_matrix(max_categories=50)
        conf_mat.compute_matrix(GT[1:].flatten(), Predictions[1:].flatten(),
                images=test_in.permute((0, 2, 3, 1)),
        )
        comet_exp.log_confusion_matrix(
                matrix=conf_mat,
                title=f'Test Confusion Mat',
                file_name=f'test_confusion_mat.json'
        )
    
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

        # Change this to be dependent on the dataset parameters 
        if dataset == 'mnist' or dataset == 'fashionmnist':
            input_size = 28
        elif dataset == 'cifar10':
            input_size = 32

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
                model_ft.conv1 = nn.Conv2d(1, model_ft.conv1.out_channels,
                        kernel_size=3, stride=1, padding=1, bias=False)
                # ResNet 18 for Github/kuangliu removes maxpool
                model_ft.maxpool = nn.Identity()
                input_size = 28
            elif dataset == 'cifar10':
                model_ft.conv1 = nn.Conv2d(3, model_ft.conv1.out_channels,
                        kernel_size=3, stride=1, padding=1, bias=False)
                # ResNet 18 for Github/kuangliu removes maxpool
                model_ft.maxpool = nn.Identity()
                input_size = 32
    
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
            elif dataset == 'cifar10':
                model_ft.conf1 = nn.Conv2d(3, model_ft.conv1.out_channels,
                        kernel_size=3, stride=1, padding=1, bias=False)
                input_size = 32

    return model_ft, input_size

