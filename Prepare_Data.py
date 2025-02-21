# -*- coding: utf-8 -*-
"""
Created on Thu Dec 19 18:07:33 2019
Load datasets for models
@author: jpeeples
"""
## Python standard libraries
from __future__ import print_function
from __future__ import division
import numpy as np
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit

## PyTorch dependencies
import torch
from torchvision import transforms, datasets

## Local external libraries
from Datasets.Subset_Wrapper import Subset_Wrapper
from Datasets.DTD_loader import DTD_data
from Datasets.MINC_2500 import MINC_2500_data
from Datasets.GTOS_mobile_single_size import GTOS_mobile_single_data


def Prepare_DataLoaders(Network_parameters, split,input_size=224, comet_exp=None):
    
    Dataset = Network_parameters['Dataset']
    data_dir = Network_parameters['data_dir']
    # For Comet ML logging
    dataset_info = {"Dataset": Dataset}
    
    # Data augmentation and normalization for training
    # Just normalization and resize for test
    # Data transformations as described in:
    # http://openaccess.thecvf.com/content_cvpr_2018/papers/Xue_Deep_Texture_Manifold_CVPR_2018_paper.pdf
    if not(Network_parameters['rotation']):
        data_transforms = {
            'train': transforms.Compose([
                transforms.Resize(Network_parameters['resize_size']),
                transforms.RandomResizedCrop(input_size,scale=(.8,1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            'val': transforms.Compose([
                transforms.Resize(Network_parameters['center_size']),
                transforms.CenterCrop(input_size),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
        }
    else:
        data_transforms = {
            'train': transforms.Compose([
                transforms.Resize(Network_parameters['resize_size']),
                transforms.RandomResizedCrop(input_size,scale=(.8,1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            'val': transforms.Compose([
                transforms.Resize(Network_parameters['center_size']),
                transforms.CenterCrop(input_size),
                transforms.RandomAffine(Network_parameters['degrees']),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
        }
    
        # Create training and test datasets
    if Dataset=='DTD':
        train_dataset = DTD_data(data_dir, data='train',
                                           numset = split + 1,
                                           img_transform=data_transforms['train'])
        
        test_dataset = DTD_data(data_dir, data = 'test',
                                           numset = split + 1,
                                           img_transform=data_transforms['val'])   
        
        # #Combine training and test datasets
        if Network_parameters['val_split']:
            validation_dataset = DTD_data(data_dir, data = 'val',
                                               numset = split + 1,
                                               img_transform=data_transforms['val'])
        else:
            validation_dataset = DTD_data(data_dir, data = 'val',
                                               numset = split + 1,
                                               img_transform=data_transforms['train'])
            train_dataset = torch.utils.data.ConcatDataset((train_dataset,validation_dataset))

            dataset_info['Train_Transform'] = data_transforms['train']
            dataset_info['Test-Val_Transform'] = data_transforms['val']
            dataset_info['Dataset_Sizes'] = [len(train_dataset),
                                             len(validation_dataset),
                                             len(test_dataset)]
        
    elif Dataset == 'MINC_2500':
        train_dataset = MINC_2500_data(data_dir, data='train',
                                           numset = split + 1,
                                           img_transform=data_transforms['train'])
        
        validation_dataset = MINC_2500_data(data_dir, data='val',
                                           numset = split + 1,
                                           img_transform=data_transforms['val'])
        
        test_dataset = MINC_2500_data(data_dir, data = 'test',
                                           numset = split + 1,
                                           img_transform=data_transforms['val'])

        dataset_info['Train_Transform'] = data_transforms['train']
        dataset_info['Test-Val_Transform'] = data_transforms['val']
        dataset_info['Dataset_Sizes'] = [len(train_dataset),
                                         len(validation_dataset),
                                         len(test_dataset)]
    elif Dataset == 'GTOS-mobile':
        # Create training and test datasets
        dataset = GTOS_mobile_single_data(data_dir, train = True,
                                          image_size=Network_parameters['resize_size'],
                                          img_transform=data_transforms['train']) 
        X = np.ones(len(dataset))
        Y = dataset.targets
        train_indices = []
        val_indices = []
    
        skf = StratifiedKFold(n_splits=Network_parameters['Splits'][Dataset],shuffle=True,
                   random_state=Network_parameters['random_state'])
        
        for train_index, val_index in skf.split(X, Y):
             train_indices.append(train_index)
             val_indices.append(val_index)
        
        validation_dataset = torch.utils.data.Subset(dataset, val_indices[split])
        train_dataset = torch.utils.data.Subset(dataset, train_indices[split])
        test_dataset = GTOS_mobile_single_data(data_dir, train = False,
                                           img_transform=data_transforms['val'])

        dataset_info['Train_Transform'] = data_transforms['train']
        dataset_info['Test-Val_Transform'] = data_transforms['val']
        dataset_info['Dataset_Sizes'] = [len(train_dataset),
                                         len(validation_dataset),
                                         len(test_dataset)]
    elif Dataset == 'mnist':
        mnist_tr = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.1307], [0.3081])
        ])
        train_dataset = datasets.MNIST(data_dir, train=True,
                                    transform=mnist_tr,
                                    download=True)
        X = np.ones(len(train_dataset))
        Y = train_dataset.targets
        train_indices = []
        val_indices = []
    
        # split = 0
        # sss = StratifiedShuffleSplit(n_splits=Network_parameters['Splits'][Dataset], test_size=0.1,
        #                       random_state=Network_parameters['random_state'])
        skf = StratifiedKFold(n_splits=Network_parameters['Splits'][Dataset],shuffle=True,
                   random_state=Network_parameters['random_state'])
        
        for train_index, val_index in skf.split(X, Y):
             train_indices.append(train_index)
             val_indices.append(val_index)
        
        validation_dataset = Subset_Wrapper(train_dataset, val_indices[split])
        train_dataset = Subset_Wrapper(train_dataset, train_indices[split])
        test_dataset = datasets.MNIST(data_dir, train=False,
                                      transform=mnist_tr,
                                      download=True)
        test_dataset = Subset_Wrapper(test_dataset, np.random.permutation(len(test_dataset)))

        dataset_info['Train_Transform'] = mnist_tr
        dataset_info['Test-Val_Transform'] = mnist_tr
        dataset_info['Dataset_Sizes'] = [len(train_dataset),
                                         len(validation_dataset),
                                         len(test_dataset)]
    elif Dataset == 'fashionmnist':
        fashion_tr = [
            transforms.ToTensor(),
            transforms.Normalize([0.2861], [0.3530]),
        ]
        extra_tr = [
            transforms.RandomCrop(28, padding=4),
            transforms.RandomHorizontalFlip()
        ]
        train_dataset = datasets.FashionMNIST(data_dir, train=True,
                                           transform=transforms.Compose(fashion_tr+extra_tr),
                                           download=True)
        X = np.ones(len(train_dataset))
        Y = train_dataset.targets
        train_indices = []
        val_indices = []
    
        # split = 0
        # sss = StratifiedShuffleSplit(n_splits=Network_parameters['Splits'][Dataset], test_size=0.1,
        #                       random_state=Network_parameters['random_state'])
        skf = StratifiedKFold(n_splits=Network_parameters['Splits'][Dataset],shuffle=True,
                   random_state=Network_parameters['random_state'])
        
        for train_index, val_index in skf.split(X, Y):
             train_indices.append(train_index)
             val_indices.append(val_index)
        
        validation_dataset = Subset_Wrapper(train_dataset, val_indices[split])
        train_dataset = Subset_Wrapper(train_dataset, train_indices[split])
        test_dataset = datasets.FashionMNIST(data_dir, train=False,
                                             transform=transforms.Compose(fashion_tr),
                                             download=True)
        test_dataset = Subset_Wrapper(test_dataset, np.random.permutation(len(test_dataset)))

        dataset_info['Train_Transform'] = fashion_tr + extra_tr
        dataset_info['Test-Val_Transform'] = fashion_tr
        dataset_info['Dataset_Sizes'] = [len(train_dataset),
                                         len(validation_dataset),
                                         len(test_dataset)]
    elif Dataset == 'cifar10':
        cifar10_tr = [
            transforms.ToTensor(),
            transforms.Normalize([0.4914, 0.4822, 0.4465],
                                 [0.2023, 0.1994, 0.2010]),
                                 # [0.2470, 0.2435, 0.2616]),
        ]
        extra_tr = [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip()
        ]
        train_dataset = datasets.CIFAR10(data_dir, train=True,
                                      transform=transforms.Compose(cifar10_tr + extra_tr),
                                      download=True)
        X = np.ones(len(train_dataset))
        Y = train_dataset.targets
        train_indices = []
        val_indices = []
    
        # split = 0
        # sss = StratifiedShuffleSplit(n_splits=Network_parameters['Splits'][Dataset], test_size=0.1,
        #                       random_state=Network_parameters['random_state'])
        skf = StratifiedKFold(n_splits=Network_parameters['Splits'][Dataset],shuffle=True,
                   random_state=Network_parameters['random_state'])
        
        for train_index, val_index in skf.split(X, Y):
             train_indices.append(train_index)
             val_indices.append(val_index)
        
        validation_dataset = Subset_Wrapper(train_dataset, val_indices[split])
        train_dataset = Subset_Wrapper(train_dataset, train_indices[split])
        test_dataset = datasets.CIFAR10(data_dir, train=False,
                                        transform=transforms.Compose(cifar10_tr),
                                        download=True)
        test_dataset = Subset_Wrapper(test_dataset, np.random.permutation(len(test_dataset)))

        dataset_info['Train_Transform'] = cifar10_tr + extra_tr
        dataset_info['Test-Val_Transform'] = cifar10_tr
        dataset_info['Dataset_Sizes'] = [len(train_dataset),
                                         len(validation_dataset),
                                         len(test_dataset)]
    elif Dataset == 'cifar100':
        cifar100_tr = [
            transforms.ToTensor(),
            transforms.Normalize([0.5071, 0.4865, 0.4409],
                                 [0.2673, 0.2564, 0.2762]),
        ]
        extra_tr = [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15)
        ]
        train_dataset = datasets.CIFAR100(data_dir, train=True,
                                       transform=transforms.Compose(cifar100_tr + extra_tr),
                                       download=True)
        X = np.ones(len(train_dataset))
        Y = train_dataset.targets
        train_indices = []
        val_indices = []
    
        sss = StratifiedShuffleSplit(n_splits=Network_parameters['Splits'][Dataset], test_size=0.1,
                              random_state=Network_parameters['random_state'])
        
        for train_index, val_index in sss.split(X, Y):
             train_indices.append(train_index)
             val_indices.append(val_index)

        validation_dataset = Subset_Wrapper(train_dataset, val_indices[0])
        train_dataset = Subset_Wrapper(train_dataset, train_indices[0])
        test_dataset = datasets.CIFAR100(data_dir, train=False,
                                         transform=transforms.Compose(cifar100_tr),
                                         download=True)
        test_dataset = Subset_Wrapper(test_dataset, np.random.permutation(len(test_dataset)))

        dataset_info['Train_Transform'] = cifar100_tr + extra_tr
        dataset_info['Test-Val_Transform'] = cifar100_tr
        dataset_info['Dataset_Sizes'] = [len(train_dataset),
                                         len(validation_dataset),
                                         len(test_dataset)]

    #Do train/val/test split or train/test split only (validating on test data)    
    if Network_parameters['val_split']:
        image_datasets = {'train': train_dataset, 'val': validation_dataset,
                          'test': test_dataset}
    else:
        image_datasets = {'train': train_dataset, 'val': test_dataset,
                          'test': test_dataset}

    # Create training and test dataloaders
    dataloaders_dict = {x: torch.utils.data.DataLoader(image_datasets[x], 
                                                       batch_size=Network_parameters['batch_size'][x], 
                                                       shuffle=True, 
                                                       num_workers=Network_parameters['num_workers'],
                                                       pin_memory=Network_parameters['pin_memory']) for x in ['train', 'val','test']}
    if comet_exp is not None:
        comet_exp.log_others(dataset_info)

    return dataloaders_dict
