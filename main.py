# -*- coding: utf-8 -*-
"""
Created on Thu Aug 11 10:12:13 2022

@author: uqalim8
"""

import utils, torch
import hyperparameters as hyper 
import sys

SEED = 3001 # integers
FOLDER_PATH = "./test/" #f"./{sys.argv[1]}"
DATASET = "CIFAR10" #MNIST, MNISTb, MNISTsb, MNISTs, CIFAR10b, CIFAR10, DelhiClimate, Ethylene, Covtype
FUNC = "ffnnMSE" #"GAN", "logloss", "nls", "ffnnMSE", "ffnnCELoss", "rnnMSE", "rnnCELoss", "rnnCovtypeMSE"
INITX0 = "torch" #zeros, ones, uniform, normal, torch
REG = "None" # None, 2-norm, Non-convex
REG_LAMBDA = 10e-2
VERBOSE = True
HSUB = 0.1 #[1, 0.1, 0.05, 0.01]
MINI = 1

ALG = [
       #("Linesearch_GD", hyper.cGD),
       #("NewtonCG", hyper.cCG),
       ("NewtonMR_NC", hyper.cMR), 
       #("NewtonCG_NC", hyper.cCG_NC),
       #("NewtonCG_NC_FW", hyper.cCG_NC),
       #("NewtonCG_TR_Steihaug", hyper.cTR_STEI)
       #("L-BFGS", hyper.cL_BFGS)
       ]

    
def run(folder_path, dataset, alg, func, x0, mini, Hsub, reg, lamb, verbose):
    
    assert type(alg) == list
        
    print("\n***Running on", str(hyper.cCUDA) + "***\n")
    
    for j, c in alg:
        
        print("\n" + 45 * ".")
        c.Hsub = Hsub
        algo, x0 = utils.execute(folder_path, dataset, j, func, x0, mini, Hsub, reg, lamb, c, verbose)
        utils.saveRecords(folder_path, dataset, j, func, Hsub, algo.record)

if __name__ == "__main__":
    
    if type(SEED) == int:
        torch.manual_seed(SEED)
    
    run(FOLDER_PATH, DATASET, ALG, FUNC, INITX0, MINI, HSUB, REG, REG_LAMBDA, VERBOSE)