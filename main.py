# -*- coding: utf-8 -*-
"""
Created on Thu Aug 11 10:12:13 2022

@author: uqalim8
"""

import utils, torch
import hyperparameters as hyper 
import sys

SEED = 1234 # integers
FOLDER_PATH = f"./{sys.argv[1]}"
DATASET = "Ethylene" #MNIST, MNISTb, MNISTsb, MNISTs, CIFAR10b, CIFAR10, DelhiClimate, Ethylene
FUNC = "rnnMSE" #"logloss", "nls", "ffnnMSE", "ffnnCELoss", "rnnMSE", "rnnCELoss"
INITX0 = "normal" #zeros, ones, uniform, normal, torch
REG = "Non-convex" # None, 2-norm, Non-convex
REG_LAMBDA = 0.1
VERBOSE = True
HSUB = float(sys.argv[2]) #[1, 0.1, 0.05, 0.01]
MINI = 1

if "CG" in sys.argv[1]:
    ALG = [("NewtonCG_NC", hyper.cCG_NC)]
elif "MRnoLS" == sys.argv[1]:
    ALG = [("NewtonMR_NC_no_LS", hyper.cMR)]
elif "MR" in sys.argv[1]:
    ALG = [("NewtonMR_NC", hyper.cMR)]
elif "TR" in sys.argv[1]:
    ALG = [("NewtonCG_TR_Steihaug", hyper.cTR_STEI)]
elif "LBFGS" in sys.argv[1]:
    ALG = [("L-BFGS", hyper.cL_BFGS)]
elif "Adam" in sys.argv[1]:
    ALG = [("Adam", hyper.cADAM)]
elif "SGD" in sys.argv[1]:
    ALG = [("MiniBatchSGD", hyper.cSGD)]
    
#ALG = [
       #("Linesearch_GD", hyper.cGD),
       #("NewtonCG", hyper.cCG),
       #("NewtonMR_NC", hyper.cMR), 
       #("NewtonCG_NC", hyper.cCG_NC),
       #("NewtonCG_NC_FW", hyper.cCG_NC),
       #("NewtonCG_TR_Steihaug", hyper.cTR_STEI)
       #("L-BFGS", hyper.cL_BFGS)
       #("Adam"
       #]

    
def run(folder_path, dataset, alg, func, x0, mini, Hsub, reg, lamb, verbose):
    
    assert type(alg) == list
        
    print("***Running on", str(hyper.cCUDA) + "***")
    
    for j, c in alg:
        print("\n" + 45 * ".")
        c.Hsub = Hsub
        algo, x0 = utils.execute(folder_path, dataset, j, func, x0, mini, Hsub, reg, lamb, c, verbose)
        utils.saveRecords(folder_path, dataset, j, func, Hsub, algo.record)

if __name__ == "__main__":
    
    if type(SEED) == int:
        torch.manual_seed(SEED)
    
    run(FOLDER_PATH, DATASET, ALG, FUNC, INITX0, MINI, HSUB, REG, REG_LAMBDA, VERBOSE)
    
    # if FUNC == "GAN":
    #     runGAN(FOLDER_PATH, DATASET, ALGG, ALGD, CONSTG, CONSTD, HSUB[0], REG, REG_LAMBDA, CUDA)
    # else:
    #     run(FOLDER_PATH, DATASET, ALG, FUNC, INITX0, HSUB, REG, REG_LAMBDA, VERBOSE, CUDA)
