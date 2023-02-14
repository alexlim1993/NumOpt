# -*- coding: utf-8 -*-
"""
Created on Thu Aug 11 10:12:13 2022

@author: uqalim8
"""

import datasets, utils
from hyperparameters import cGD, cMR, cCG_NC
import torch

torch.manual_seed(1234)
FOLDER_PATH = "./test/"
DATASET = "MNIST"
FUNC = "ffnnMSE" #"nls", "ffnnMSE", "ffnnCELoss", "AE_MNIST"
INITX0 = "torch" #zeros, ones, uniform, normal, torch
REG = "None" # None, 2-norm, Non-convex
REG_LAMBDA = 10e-3
VERBOSE = True
HSUB = [1, 0.1, 0.05, 0.01]

ALG = [
       #("Linesearch_GD", cGD),
       #("NewtonCG", cCG),
       ("NewtonMR_NC", cMR), 
       #("NewtonCG_NC", cCG_NC),
       ("NewtonCG_NC_FW", cCG_NC),
       ]

def run(folder_path, dataset, algo, func, x0, Hsub, reg, lamb, const, verbose):
    utils.makeFolder(folder_path)
    trainX, trainY, testX, testY = datasets.prepareData(folder_path, func, dataset)
    reg = utils.initReg(reg, lamb)
    x0, pred, func = utils.initFunc_x0(func, x0, trainX, trainY, Hsub, reg)
    algo = utils.initAlg(func, x0.clone(), algo, const)
    algo.optimize(verbose, pred)
    return algo, x0
        
def runMultiples(folder_path, dataset, alg, func, x0, Hsub, reg, lamb, verbose):
    
    assert type(Hsub) == list
    assert type(alg) == list
    
    for j, c in alg:
        for i in Hsub:
            print("\n" + 45 * ".")
            c.Hsub = i
            algo, x0 = run(folder_path, dataset, j, func, x0, i, reg, lamb, c, verbose)
            utils.saveRecords(folder_path, dataset, j, func, i, algo.record)
            #records.append((j + f"ss_{i}", algo.record))
    
   # utils.drawPlots(records, (("orcs", "f"), ("orcs", "g_norm"), 
   #                           ("time", "f"), ("time", "g_norm"),
   #                           ("ite", "f"), ("ite", "g_norm")), folder_path)
    
if __name__ == "__main__":
    runMultiples(FOLDER_PATH, DATASET, ALG, FUNC, INITX0, HSUB, REG, REG_LAMBDA, VERBOSE)
