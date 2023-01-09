# -*- coding: utf-8 -*-
"""
Created on Thu Aug 11 10:12:13 2022

@author: uqalim8
"""

import datasets, utils

FOLDER_PATH = "./results/"
DATASET = "MNISTb"
FUNC = "nls"
ALG = "NewtonMR_NC"
INITX0 = "zeros" #zeros, ones, rand, randn
REG = "None"
HSUB = [0.01]
REG_LAMBDA = 0
VERBOSE = True
MULTIPLES = "HSub"

c = utils.const()
c.alpha0 = 1
c.gradtol = 1e-6
c.maxite = 1000
c.restol = 0.000001
c.inmaxite = 1000
c.maxorcs = 1000
c.lineMaxite = 100
c.lineBetaB = 1e-4
c.lineRho = 0.9
c.lineBetaFB = 0.25
c.Hsub = HSUB

cCG = utils.const()
cCG.alpha0 = 1
cCG.gradtol = 1e-6
cCG.maxite = 1000
cCG.restol = 0.01#1e-4
cCG.inmaxite = 1000
cCG.maxorcs = 1000
cCG.lineMaxite = 100
cCG.lineBeta = 1e-4#0.01
cCG.lineRho = 0.9
cCG.epsilon = 1e-4
cCG.Hsub = HSUB

C = c

def run(folder_path, dataset, alg, func, x0, Hsub, reg, lamb, const, verbose):
    utils.makeFolder(folder_path)
    trainX, trainY, testX, testY = datasets.prepareData(folder_path, dataset)
    x0 = utils.initx0(x0, trainX.shape[-1])
    reg = utils.initReg(reg, lamb)
    func = utils.initFunc(func, trainX, trainY, Hsub, reg)
    algo = utils.initAlg(func, x0, alg, const)
    algo.optimize(verbose)
    return algo
        
def runMultiples(multiple, folder_path, dataset, alg, func, x0, Hsub, reg, lamb, const, verbose):
    
    records = []
    
    if multiple == "HSub":
        assert type(Hsub) == list
        for i in HSUB:
            print("\n" + 15*"." + f"{alg} {multiple} {i}" + 15*"." + "\n")
            const.Hsub = i
            algo = run(folder_path, dataset, alg, func, x0, i, reg, lamb, const, verbose)
            utils.saveRecords(folder_path, dataset, alg, func, i, algo.record)
            records.append((alg + f" {multiple}_{i}", algo.record))
    
    utils.drawPlots(records, (("orcs", "f"), ("orcs", "g_norm"), 
                              ("time", "f"), ("time", "g_norm"),
                              ("ite", "f"), ("ite", "g_norm")), folder_path)
    
if __name__ == "__main__":
    runMultiples(MULTIPLES, FOLDER_PATH, DATASET, ALG, FUNC, INITX0, HSUB, REG, REG_LAMBDA, C, VERBOSE)
