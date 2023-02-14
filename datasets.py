# -*- coding: utf-8 -*-
"""
Created on Tue Aug  9 14:20:15 2022

@author: uqalim8
"""

import torch
import sklearn.datasets as skdatasets
import torchvision.datasets as datasets

TEXT = "{:<20} : {:>20}"

def prepareData(folder_path, func, dataset):
    
    one_hot = True if "nn" in func else False
    
    if dataset == "MNISTb":
        print(TEXT.format("Dataset", dataset))
        return MNIST(folder_path, one_hot, 2)
    
    if dataset == "MNIST":
        print(TEXT.format("Dataset", dataset))
        return MNIST(folder_path, one_hot, 10)
    
    if dataset == "CIFAR10b":
        print(TEXT.format("Dataset", dataset))
        return CIFAR10b(folder_path, one_hot)
    
    if dataset == "MNISTs":
        print(TEXT.format("Dataset", dataset))
        return MNISTs(folder_path, one_hot, 10)
    
    if dataset == "MNISTsb":
        print(TEXT.format("Dataset", dataset))
        return MNISTs(folder_path, one_hot, 2)
        

def MNISTs(folder_path, one_hot, classes):
    """
    MNIST small size (8 by 8 pixels)

    """
    trainX, trainY = skdatasets.load_digits(return_X_y = True)
    trainX = torch.tensor(trainX, dtype = torch.float64) / 16
    trainY = torch.tensor(trainY) % classes
    
    if one_hot:
        trainY = torch.nn.functional.one_hot(trainY.long(), classes).double()
        
    return trainX, trainY, None, None
    
def MNIST(folder_path, one_hot, classes):
    
    train_set = datasets.MNIST(folder_path, train = True, download = True)
    test_set = datasets.MNIST(folder_path, train = False, download = True)

    train_set.data = train_set.data.reshape(train_set.data.shape[0], -1)
    test_set.data = test_set.data.reshape(test_set.data.shape[0], -1)
    train_set.data = train_set.data / 255
    test_set.data = test_set.data / 255

    train_set.targets = train_set.targets % classes
    test_set.targets = test_set.targets % classes

    if one_hot:
        train_set.targets = torch.nn.functional.one_hot(train_set.targets.long(), classes)
        test_set.targets = torch.nn.functional.one_hot(test_set.targets.long(), classes)
            
    return train_set.data.to(torch.float64), train_set.targets.double(), \
        test_set.data.to(torch.float64), test_set.targets.double()
        
        
def CIFAR10b(folder_path):
    train_set = datasets.CIFAR10(folder_path, train = True, download = True)
    test_set = datasets.CIFAR10(folder_path, train = False, download = True)
    
    train_set.targets = train_set.targets % 2
    test_set.targets = test_set.targets % 2
    train_set.data = train_set.data.reshape(train_set.data.shape[0], -1)
    test_set.data = test_set.data.reshape(test_set.data.shape[0], -1)
    train_set.data = train_set.data / 255
    test_set.data = test_set.data / 255

    return train_set.data.to(torch.float64), train_set.targets, \
        test_set.data.to(torch.float64), test_set.targets