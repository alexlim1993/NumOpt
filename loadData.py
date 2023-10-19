# -*- coding: utf-8 -*-
"""
Created on Mon Sep 28 11:10:09 2020

@author: Yang Liu
"""

import numpy as np
import math
import torch

def unpickle_csv(file):
    with open(file, 'rb') as fo:
        data = np.loadtxt(fo, delimiter=",", skiprows=0)           
    return data

def loadData():
    raw_data = unpickle_csv("../data/spam/spambase.data")
    train_raw = raw_data[:, 0:-1]
    labels_raw = raw_data[:, -1].reshape(-1)
    return torch.tensor(train_raw, dtype = torch.float64), torch.tensor(labels_raw), None, None


def main():
    A_train, b_train, A_test, b_test = loadData()
    print(A_train.shape,A_train.dtype)
    print(b_train.shape,b_train.dtype) 
    print(A_test.shape,A_test.dtype) 
    print(b_test.shape,b_test.dtype) 
    
if __name__ == '__main__':
    main()
    