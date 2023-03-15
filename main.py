# -*- coding: utf-8 -*-
"""
Created on Thu Aug 11 10:12:13 2022

@author: uqalim8
"""

import utils, torch
import hyperparameters as hyper 

SEED = None # integers
FOLDER_PATH = "./test/"
DATASET = "DelhiClimate" #MNIST, MNISTb, MNISTsb, MNISTs, CIFAR10b, CIFAR10, DelhiClimate
FUNC = "rnnMSE" #"GAN", "logloss", "nls", "ffnnMSE", "ffnnCELoss", "rnnMSE", "rnnCELoss"
INITX0 = "torch" #zeros, ones, uniform, normal, torch
REG = "None" # None, 2-norm, Non-convex
REG_LAMBDA = 10e-2
VERBOSE = True
CUDA = False
HSUB = [1] #[1, 0.1, 0.05, 0.01]

ALG = [
       #("Linesearch_GD", hyper.cGD),
       #("NewtonCG", hyper.cCG),
       ("NewtonMR_NC", hyper.cMR), 
       #("NewtonCG_NC", hyper.cCG_NC),
       #("NewtonCG_NC_FW", hyper.cCG_NC),
       #("NewtonCG_TR_Steihaug", hyper.cTR_STEI)
       ]

# if FUNC == "GAN":
#     ALGG = "NewtonMR_NC"
#     CONSTG = hyper.cMR
#     CONSTG.Hsub = HSUB[0]
    
#     ALGD = "NewtonMR_NC"
#     CONSTD = hyper.cMR
#     CONSTD.Hsub = HSUB[0]
    
def run(folder_path, dataset, alg, func, x0, Hsub, reg, lamb, verbose, gpu):
    
    assert type(Hsub) == list
    assert type(alg) == list
    
    if gpu:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")
        
    print("***Running on", str(device) + "***")
    
    for j, c in alg:
        for i in Hsub:
            print("\n" + 45 * ".")
            c.Hsub = i
            algo, x0 = utils.execute(folder_path, dataset, j, func, x0, i, reg, lamb, c, verbose, device)
            utils.saveRecords(folder_path, dataset, j, func, i, algo.record)

def runGAN(folder_path, dataset, algG, algD, constG, constD, Hsub, reg, lamb, gpu):
        
    if gpu:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")
        
    print("***Running on", str(device) + "***")

    utils.executeGAN(folder_path, dataset, algG, algD, Hsub, reg, lamb, constG, constD, device)
            
if __name__ == "__main__":
    
    if type(SEED) == int:
        torch.manual_seed(SEED)
    
    run(FOLDER_PATH, DATASET, ALG, FUNC, INITX0, HSUB, REG, REG_LAMBDA, VERBOSE, CUDA)
    
    # if FUNC == "GAN":
    #     runGAN(FOLDER_PATH, DATASET, ALGG, ALGD, CONSTG, CONSTD, HSUB[0], REG, REG_LAMBDA, CUDA)
    # else:
    #     run(FOLDER_PATH, DATASET, ALG, FUNC, INITX0, HSUB, REG, REG_LAMBDA, VERBOSE, CUDA)