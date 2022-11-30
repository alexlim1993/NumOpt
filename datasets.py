# -*- coding: utf-8 -*-
"""
Created on Tue Aug  9 14:20:15 2022

@author: uqalim8
"""

import torch
import sklearn.datasets as skdatasets
import torchvision.datasets as datasets

def prepareData(folder_path, dataset):
    if dataset == "MNISTb":
        return MNISTb(folder_path)
    
    if dataset == "CIFAR10b":
        return CIFAR10b(folder_path)
    
    if dataset == "MNISTsb":
        return MNISTsb(folder_path)

def MNISTsb(folder_path):
    trainX, trainY = skdatasets.load_digits(return_X_y = True)
    trainX = torch.tensor(trainX, dtype = torch.float64) / 16
    trainY = torch.tensor(trainY) % 2
    return trainX, trainY, None, None
    

def MNISTb(folder_path):
    """
    binary MNIST
    """
    train_set = datasets.MNIST(folder_path, train = True, download = True)
    test_set = datasets.MNIST(folder_path, train = False, download = True)

    train_set.targets = train_set.targets % 2
    test_set.targets = test_set.targets % 2
    train_set.data = train_set.data.reshape(train_set.data.shape[0], -1)
    test_set.data = test_set.data.reshape(test_set.data.shape[0], -1)
    train_set.data = train_set.data / 255
    test_set.data = test_set.data / 255
        
    return train_set.data.to(torch.float64), train_set.targets, \
        test_set.data.to(torch.float64), test_set.targets
        
def CIFAR10b(folder_path):
    train_set = datasets.CIFAR10(folder_path, train = True, download = True)
    test_set = datasets.CIFAR10(folder_path, train = False, download = True)
    
    train_set.data = torch.tensor(train_set.data, dtype = torch.float64) / 255
    test_set.data = torch.tensor(test_set.data, dtype = torch.float64) / 255
    train_set.data = train_set.data.reshape(train_set.data.shape[0], -1)
    test_set.data = test_set.data.reshape(test_set.data.shape[0], -1)
    train_set.targets = torch.tensor(train_set.targets) % 2
    test_set.targets = torch.tensor(test_set.targets) % 2

    return train_set.data, train_set.targets, \
        test_set.data, test_set.targets