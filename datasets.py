# -*- coding: utf-8 -*-
"""
Created on Tue Aug  9 14:20:15 2022

@author: uqalim8
"""

import torch, numpy, pandas, sklearn, math
import sklearn.datasets as skdatasets
import torchvision.datasets as datasets
from hyperparameters import cTYPE

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
        return CIFAR10(folder_path, one_hot, 2)
    
    if dataset == "CIFAR10":
        print(TEXT.format("Dataset", dataset))
        return CIFAR10(folder_path, one_hot, 10)
    
    if dataset == "MNISTs":
        print(TEXT.format("Dataset", dataset))
        return MNISTs(folder_path, one_hot, 10)
    
    if dataset == "MNISTsb":
        print(TEXT.format("Dataset", dataset))
        return MNISTs(folder_path, one_hot, 2)
    
    if dataset == "DelhiClimate":
        print(TEXT.format("Dataset", dataset))
        return DelhiClimate()
    
    if dataset == "Ethylene":
        print(TEXT.format("Dataset", dataset))
        return Ethylene()
    
    if dataset == "Covtype":
        print(TEXT.format("Dataset", dataset))
        return Covtype()
        
def MNISTs(folder_path, one_hot, classes):
    """
    MNIST small size (8 by 8 pixels)

    """
    trainX, trainY = skdatasets.load_digits(return_X_y = True)
    trainX = torch.tensor(trainX, dtype = cTYPE) / 16
    trainY = torch.tensor(trainY) % classes
    
    if one_hot:
        trainY = torch.nn.functional.one_hot(trainY.long(), classes).to(cTYPE)
        
    return trainX, trainY, None, None
    
def MNIST(folder_path, one_hot, classes):
    
    train_set = datasets.MNIST(folder_path, train = True, download = True)
    test_set = datasets.MNIST(folder_path, train = False, download = True)

    train_set.data = train_set.data.reshape(train_set.data.shape[0], -1).to(cTYPE)
    test_set.data = test_set.data.reshape(test_set.data.shape[0], -1).to(cTYPE)
    train_set.data = train_set.data / 255
    test_set.data = test_set.data / 255

    train_set.targets = train_set.targets % classes
    test_set.targets =test_set.targets % classes

    if one_hot:
        train_set.targets = torch.nn.functional.one_hot(train_set.targets.long(), classes)
        test_set.targets = torch.nn.functional.one_hot(test_set.targets.long(), classes)
            
    return train_set.data, train_set.targets.to(cTYPE), \
        test_set.data, test_set.targets.to(cTYPE)
        
        
def CIFAR10(folder_path, one_hot, classes):

    train_set = datasets.CIFAR10("./", train = True, download = True)
    test_set = datasets.CIFAR10("./", train = False, download = True)
    
    trainX = torch.tensor(train_set.data.reshape(train_set.data.shape[0], -1), dtype = cTYPE) / 255
    trainY = torch.tensor(train_set.targets) % classes
    testX = torch.tensor(test_set.data.reshape(test_set.data.shape[0], -1), dtype = cTYPE) / 255
    testY = torch.tensor(test_set.targets) % classes
    del train_set, test_set
    
    if one_hot:
        trainY = torch.nn.functional.one_hot(trainY.long(), classes)
        testY = torch.nn.functional.one_hot(testY.long(), classes)
    
    print(TEXT.format("Training Samples", str(tuple(trainX.shape))))
    print(TEXT.format("Test Samples", str(tuple(testX.shape))))
    return trainX, trainY.to(cTYPE), testX, testY.to(cTYPE)
        
def DelhiClimate(window = 7):
    
    train = pandas.read_csv("./custom_data/DailyDelhiClimateTrain.csv")
    train = torch.tensor(train.drop("date", axis = 1).to_numpy(), dtype = cTYPE)
    
    #Standardise
    #std, mean = torch.std_mean(train, dim = 0)
    #train = (train - mean) / std
    
    n, d = train.shape
    trainX = torch.zeros((n - window, window, d), dtype = cTYPE)
    trainY = torch.zeros((n - window, d), dtype = cTYPE)
    
    for i in range(window, n):
        trainX[i - window] = train[i - window : i]
        trainY[i - window] = train[i]
    
    print(TEXT.format("Data size", str(tuple(trainX.shape))))
    return trainX, trainY, None, None
        
def Ethylene(window = 3, stride = 3):
    classA = torch.tensor(numpy.loadtxt("./custom_data/mean_ethylene_CO.txt", delimiter = ",")) / 100
    classB = torch.tensor(numpy.loadtxt("./custom_data/mean_ethylene_methane.txt", delimiter = ",")) / 100

    #prepX = torch.concat([classA, classB], dim = 0)
    
    #std, mean = torch.std_mean(prepX, dim = 0)
    #prepX = (prepX - mean) / std
    
    n, d = classA.shape
    m, _ = classB.shape
    #del classA, classB
    
    trainX = torch.zeros((len(range(window, n, stride)) +
                          len(range(window, m, stride)), window, d - 2), dtype = cTYPE)
        
    trainY = torch.zeros((len(range(window, n, stride)) + 
                          len(range(window, m, stride)), 1), dtype = cTYPE)
    j = 0
    for i in range(window, n, stride):
        trainX[j, :, 1:] = classA[i - window : i, 2:-1]
        trainX[j, :, 0] = 1.
        #trainX[j, :] = classA[i - window : i, 2:-1]
        trainY[j] = classA[i - 1, -1]
        j += 1

    for i in range(window, m, stride):
        trainX[j, :, 1:] = classB[i - window : i, 2:-1]
        #trainX[j, :] = classB[i - window : i, 2:-1]
        trainY[j] = classB[i - 1, -1]
        j += 1
    
    #shuffle
    n = torch.randperm(trainX.shape[0])
    trainX, trainY = trainX[n], trainY[n]
    #trainY = torch.nn.functional.one_hot(trainY.long(), 2)
    testX, testY = trainX[:44000], trainY[:44000]
    trainX, trainY = trainX[44000:], trainY[44000:]
    print(TEXT.format("Training size", str(tuple(trainX.shape))))
    print(TEXT.format("Test size", str(tuple(testX.shape))))
    return trainX, trainY, testX, testY

def Covtype():
    X, Y = sklearn.datasets.fetch_covtype(return_X_y = True)
    X = torch.tensor(X, dtype = cTYPE)
    Y = torch.tensor(Y, dtype = torch.long) - 1
    
    test_X = X[:2000]
    X = X[2000:302000]
    
    test_Y = Y[:2000]
    Y = Y[2000:302000]
    
    #std, mean = torch.std_mean(X, dim = 0)
    #X = (X - mean) / std
    
    Y = torch.nn.functional.one_hot(Y, 7).to(cTYPE)
    test_Y = torch.nn.functional.one_hot(test_Y, 7).to(cTYPE)
    print(TEXT.format("Data size", str(tuple(X.shape)))) 
    return X, Y, test_X, test_Y
