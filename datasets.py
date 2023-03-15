# -*- coding: utf-8 -*-
"""
Created on Tue Aug  9 14:20:15 2022

@author: uqalim8
"""

import torch
import sklearn.datasets as skdatasets
import torchvision.datasets as datasets
import pandas as pd

TEXT = "{:<20} : {:>20}"

def prepareData(folder_path, func, dataset, device):
    
    one_hot = True if "nn" in func else False
    
    if dataset == "MNISTb":
        print(TEXT.format("Dataset", dataset))
        return MNIST(folder_path, one_hot, 2, device)
    
    if dataset == "MNIST":
        print(TEXT.format("Dataset", dataset))
        return MNIST(folder_path, one_hot, 10, device)
    
    if dataset == "CIFAR10b":
        print(TEXT.format("Dataset", dataset))
        return CIFAR10(folder_path, one_hot, 2, device)
    
    if dataset == "CIFAR10":
        print(TEXT.format("Dataset", dataset))
        return CIFAR10(folder_path, one_hot, 10, device)
    
    if dataset == "MNISTs":
        print(TEXT.format("Dataset", dataset))
        return MNISTs(folder_path, one_hot, 10, device)
    
    if dataset == "MNISTsb":
        print(TEXT.format("Dataset", dataset))
        return MNISTs(folder_path, one_hot, 2, device)
    
    if dataset == "DelhiClimate":
        print(TEXT.format("Dataset", dataset))
        return DelhiClimate(None, None, None, device)
        
def MNISTs(folder_path, one_hot, classes, device):
    """
    MNIST small size (8 by 8 pixels)

    """
    trainX, trainY = skdatasets.load_digits(return_X_y = True)
    trainX = torch.tensor(trainX, dtype = torch.float64) / 16
    trainY = torch.tensor(trainY) % classes
    
    if one_hot:
        trainY = torch.nn.functional.one_hot(trainY.long(), classes).double()
        
    return trainX.to(device), trainY.to(device), None, None
    
def MNIST(folder_path, one_hot, classes, device):
    
    train_set = datasets.MNIST(folder_path, train = True, download = True)
    test_set = datasets.MNIST(folder_path, train = False, download = True)

    train_set.data = train_set.data.reshape(train_set.data.shape[0], -1).to(torch.float64)
    test_set.data = test_set.data.reshape(test_set.data.shape[0], -1).to(torch.float64)
    train_set.data = train_set.data / 255
    test_set.data = test_set.data / 255

    train_set.targets = train_set.targets % classes
    test_set.targets =test_set.targets % classes

    if one_hot:
        train_set.targets = torch.nn.functional.one_hot(train_set.targets.long(), classes)
        test_set.targets = torch.nn.functional.one_hot(test_set.targets.long(), classes)
            
    return train_set.data.to(device), train_set.targets.double().to(device), \
        test_set.data.to(device), test_set.targets.double().to(device)
        
        
def CIFAR10(folder_path, one_hot, classes, device):

    train_set = datasets.CIFAR10(folder_path, train = True, download = True)
    test_set = datasets.CIFAR10(folder_path, train = False, download = True)
    
    train_set.data = torch.tensor(train_set.data.reshape(train_set.data.shape[0], -1), dtype = torch.float64)
    test_set.data = torch.tensor(test_set.data.reshape(test_set.data.shape[0], -1), dtype = torch.float64)
    train_set.data = train_set.data / 255
    test_set.data = test_set.data / 255

    train_set.targets = torch.tensor(train_set.targets, dtype = torch.float64) % classes
    test_set.targets = torch.tensor(test_set.targets, dtype = torch.float64) % classes
    
    if one_hot:
        train_set.targets = torch.nn.functional.one_hot(train_set.targets.long(), classes)
        test_set.targets = torch.nn.functional.one_hot(test_set.targets.long(), classes)
        
    return train_set.data.to(torch.float64).to(device), train_set.targets.to(device), \
        test_set.data.to(torch.float64).to(device), test_set.targets.to(device)
        
def DelhiClimate(folder_path, one_hot, classes, device, window = 7):
    
    train = pd.read_csv("./custom_data/DailyDelhiClimateTrain.csv")
    test = pd.read_csv("./custom_data/DailyDelhiClimateTest.csv")
    train = torch.tensor(train.drop("date", axis = 1).to_numpy(), dtype = torch.float64, device = device)
    test = torch.tensor(test.drop("date", axis = 1).to_numpy(), dtype = torch.float64, device = device)

    n, d = train.shape
    m, d = test.shape
    trainX = torch.zeros((n - window, window, d), dtype = torch.float64, device = device)
    trainY = torch.zeros((n - window, d), dtype = torch.float64, device = device)
    
    testX = torch.zeros((m - window, window, d), dtype = torch.float64, device = device)
    testY = torch.zeros((m - window, d), dtype = torch.float64, device = device)
    for i in range(window, n):
        trainX[i - window] = train[i - window : i]
        trainY[i - window] = train[i]
        
    for i in range(window, m):
        testX[i - window] = test[i - window : i]
        testY[i - window] = test[i]
        
    return trainX, trainY, testX, testY