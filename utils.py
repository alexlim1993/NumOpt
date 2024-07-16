# -*- coding: utf-8 -*-
"""
Created on Thu Aug 11 10:12:13 2022

@author: uqalim8
"""

import os, torch, optAlgs, regularizers, funcs, json, datasets
import neuralNetwork as nn
from hyperparameters import cTYPE, cCUDA

TEXT = "{:<20} : {:>20}"

def makeFolder(folder_path):
    if not os.path.isdir(folder_path):
        os.makedirs(folder_path)

def saveRecords(folder_path, dataset, alg, func, hsub, file):
    if folder_path[-1] != "/":
        folder_path += "/"
    folder_path += f"{alg}_{hsub}.json"
    with open(folder_path, "w") as f:
        json.dump(file, f)
        
def openRecords(folder_path, dataset, func):
    if folder_path[-1] != "/":
        folder_path += "/"
    files = os.listdir(f"{folder_path}{dataset}_{func}/")
    records = []
    for i in files:
        with open(folder_path + i, "r") as f:
            records.append((i, json.load(f)))
    return records
                
def initx0(x0_type, size):
    
    if not type(x0_type) == str:
        print(TEXT.format("x0", "initialised"))
        return x0_type.to(cCUDA)
    
    if x0_type == "ones":
        print(TEXT.format("x0", x0_type))
        return torch.ones(size, dtype = cTYPE, device = cCUDA)
    
    if x0_type == "zeros":
        print(TEXT.format("x0", x0_type))
        return torch.zeros(size, dtype = cTYPE, device = cCUDA)
    
    if x0_type == "normal":
        print(TEXT.format("x0", x0_type))
        return torch.randn(size, dtype = cTYPE, device = cCUDA) * 0.1
    
    if x0_type == "uniform":
        print(TEXT.format("x0", x0_type))
        return torch.rand(size, dtype = cTYPE, device = cCUDA)
    
def initAlg(fun, x0, algo, c, mini):
    
    if algo == "NewtonCG":
        print(TEXT.format("Algorithm", algo))
        return optAlgs.NewtonCG(fun, x0, c.alpha0, c.gradtol, c.maxite, 
                                c.maxorcs, c.restol, c.inmaxite)
        
    if algo == "NewtonMR_NC":
        print(TEXT.format("Algorithm", algo))
        return optAlgs.NewtonMR_NC(fun, x0, c.alpha0, c.gradtol, c.maxite, 
                                   c.maxorcs, c.restol, c.inmaxite, c.sigma, c.lineMaxite,
                                   c.lineBetaB, c.lineRho, c.lineBetaFB, c.Hsub)

    if algo == "NewtonMR_NC_no_LS":
        print(TEXT.format("Algorithm", algo))
        return optAlgs.NewtonMR_NC_no_LS(fun, x0, c.alpha0, c.gradtol, c.maxite,
                                         c.maxorcs, c.restol, c.inmaxite, None,
                                         None, None, None, c.Hsub)                      
    
    if algo == "NewtonCG_NC":
        print(TEXT.format("Algorithm", algo))
        return optAlgs.NewtonCG_NC(fun, x0, c.alpha0, c.gradtol, c.maxite, c.maxorcs, 
                                   c.restol, c.inmaxite, c.lineMaxite, c.lineBeta, 
                                   c.lineRho, c.epsilon, c.Hsub)
    
    if algo == "Linesearch_GD":
        print(TEXT.format("Algorithm", algo))
        return optAlgs.linesearchGD(fun, x0, c.alpha0, c.gradtol, c.maxite, 
                                    c.maxorcs, c.lineMaxite, c.lineBetaB, c.lineRho)
    
    if algo == "NewtonCG_NC_FW":
        print(TEXT.format("Algorithm", algo))
        return optAlgs.NewtonCG_NC_FW(fun, x0, c.alpha0, c.gradtol, c.maxite, c.maxorcs, 
                                   c.restol, c.inmaxite, c.lineMaxite, c.lineBeta, 
                                   c.lineRho, c.epsilon, c.Hsub)
    
    if algo == "NewtonCG_TR_Steihaug":
        print(TEXT.format("Algorithm", algo))
        return optAlgs.NewtonCG_TR_Steihaug(fun, x0, c.gradtol, c.maxite, c.maxorcs, 
                                            c.restol, c.inmaxite, c.deltaMax, c.delta0, 
                                            c.eta, c.eta1, c.eta2, c.gamma1, c.gamma2, c.Hsub)
    
    if algo == "L-BFGS":
        print(TEXT.format("Algorithm", algo))
        return optAlgs.L_BFGS(fun, x0, c.alpha0, c.gradtol, c.m, 
                              c.maxite, c.maxorcs, c.lineMaxite)
    
    if algo == "Adam":
        print(TEXT.format("Algorithm", algo))
        return optAlgs.Adam(fun, x0, c.gradtol, c.maxite, c.maxorcs, mini, 
                            c.alpha0, c.beta1, c.beta2, c.epsilon)
    
    if algo == "MiniBatchSGD":
        print(TEXT.format("Algorithm", algo))
        return optAlgs.MiniBatchSGD(fun, x0, c.gradtol, c.maxite, 
                                    c.maxorcs, mini, c.alpha0)

def initReg(reg, lamb):
    
    if reg == "None":
        print(TEXT.format("Regulariser", f"{reg}"))
        return 
    
    if reg == "Non-convex":
        print(TEXT.format("Regulariser", f"{reg} , {lamb}"))
        return lambda x : regularizers.non_convex(x, lamb)
    
    if reg == "2-norm":
        print(TEXT.format("Regulariser", f"{reg} , {lamb}"))
        return lambda x : regularizers.two_norm(x, lamb)
    
def initFunc_x0(func, x0, trainX, trainY, testX, testY, mini, Hsub, reg):
    
    print(TEXT.format("Hsub", Hsub))
    if "MSE" in func:
        loss = torch.nn.MSELoss()
        
    elif "CELoss" in func:
        loss = torch.nn.CrossEntropyLoss()
        
    if func in ["logloss", "nls"]:
        print(TEXT.format("Objective Function", func))
        x0 = initx0(x0, trainX.shape[-1])
        
        if not max(trainY) == 1 or not min(trainY) == 0:
            raise Exception("Only 0-and-1 binary Classification")
        
        def pred(w):
            pred_trainY = torch.round(funcs.logisticModel(trainX, w))
            pred_testY = torch.round(funcs.logisticModel(testX, w))
            return 100*(1 - float(torch.sum(pred_trainY == trainY) / len(trainY))),\
            100*(1 - float(torch.sum(pred_testY == testY) / len(testY)))
        if func == "logloss":
            return x0, pred, lambda w, v : funcs.funcWrapper(funcs.logloss, trainX, trainY, w, Hsub, reg, v)
        return x0, pred, lambda w, v : funcs.funcWrapper(funcs.nls, trainX, trainY, w, Hsub, reg, v)

    #if func == "nls":
    #    print(TEXT.format("Objective Function", func))
    #    x0 = initx0(x0, trainX.shape[-1])
    #    
    #    if not max(trainY) == 1 or not min(trainY) == 0:
    #        raise Exception("Only 0-and-1 binary Classification")
    #        
    #    def pred(w):
    #        pred_trainY = torch.round(funcs.logisticModel(trainX, w))
    #        pred_testY = torch.round(funcs.logisticModel(testX, w))
    #        return 100*(1 - float(torch.sum(pred_trainY == trainY) / len(trainY))),\
    #        100*(1 - float(torch.sum(pred_testY == testY) / len(testY)))
    #
    #    return x0, pred, lambda w, v : funcs.funcWrapper(funcs.nls, trainX, trainY, w, Hsub, reg, v)
    
    if "ffnn" in func:
        print(TEXT.format("Objective Function", func))
        
        dim, cat = trainX.shape[-1], trainY.shape[-1]
        ffn = nn.FFN(dim, cat)
        ffn.to(cCUDA).to(cTYPE)
        
        w = torch.nn.utils.parameters_to_vector(ffn.parameters())
        if x0 == "torch":
            print(TEXT.format("x0", x0))
            x0 = w
        else:
            x0 = initx0(x0, len(w)).requires_grad_(True)
            
        def pred(w):
            with torch.no_grad():
                torch.nn.utils.vector_to_parameters(w, ffn.parameters())
                pred_trainY = torch.argmax(ffn(trainX), dim = -1)
                pred_testY = torch.argmax(ffn(testX), dim = -1)
                return float(torch.sum(pred_trainY == torch.argmax(trainY, dim = -1)) / len(pred_trainY)) * 100,\
                    float(torch.sum(pred_testY == torch.argmax(testY, dim = -1)) / len(pred_testY)) * 100
            
        print(TEXT.format("dimension", x0.shape[0]))

        return x0, pred, nn.ObjFunc(ffn, loss, trainX, trainY, reg, mini, Hsub)

    if "rnn" in func:
        print(TEXT.format("Objective Function", func))
        n, seq, dim = trainX.shape
        _, out = trainY.shape
        rnn = nn.RNNet(dim, 32, 32, [32, 16], out).to(cCUDA).to(cTYPE)
        
        w = torch.nn.utils.parameters_to_vector(rnn.parameters())       
        if x0 == "torch":
            print(TEXT.format("x0", x0))
            x0 = w #torch.nn.utils.parameters_to_vector(rnn.parameters())
        else:
            x0 = initx0(x0, len(w)).requires_grad_(True)
        
        print(TEXT.format("dimension", x0.shape[0]))
        
        def pred(w):
            with torch.no_grad():
                torch.nn.utils.vector_to_parameters(w, rnn.parameters())
                return float(torch.mean(torch.abs(trainY - rnn(trainX)))), float(torch.mean(torch.abs(testY - rnn(testX))))

        return x0, pred, nn.ObjFunc(rnn, loss, trainX, trainY, reg, mini, Hsub)
    
def execute(folder_path, dataset, algo, func, x0, mini, Hsub, reg, lamb, const, verbose):
    makeFolder(folder_path)
    trainX, trainY, testX, testY = datasets.prepareData(folder_path, func, dataset)
    testX, testY = testX.to(cCUDA), testY.to(cCUDA)
    trainX, trainY = trainX.to(cCUDA), trainY.to(cCUDA)
    reg = initReg(reg, lamb)
    x0, pred, func = initFunc_x0(func, x0, trainX, trainY, testX, testY, mini, Hsub, reg)
    algo = initAlg(func, x0.clone(), algo, const, mini)
    algo.optimize(verbose, pred)
    return algo, x0
