# -*- coding: utf-8 -*-
"""
Created on Mon July 01 15:20:43 2019
Basic Dataset Wrapper to ensure compatibility with Josh Peeples' Histogram
Layer code. This is mainly to enable usage of standard TorchVision datasets
like MNIST, FashionMNIST, CIFAR10, CIFAR100, etc.
@author: changspencer
"""
from torch.utils.data import Subset

class Subset_Wrapper(Subset):

    def __init__(self, dataset, indices):
        super().__init__(dataset, indices)

    def __getitem__(self, index):
        img, label = self.dataset[self.indices[index]]
        # label = self.dataset.targets[self.indices[index]]
        return img, label, index
