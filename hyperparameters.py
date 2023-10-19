# -*- coding: utf-8 -*-
"""
Created on Tue Feb 14 11:23:10 2023

@author: uqalim8
"""
import torch

cTYPE = torch.float64
cCUDA = True
cSPLIT = 1000000

if cCUDA:
    cCUDA = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
else:
    cCUDA = torch.device("cpu")

class const():
    pass

cMR = const()
cMR.alpha0 = 1
cMR.gradtol = 1e-9
cMR.maxite = 1e6
cMR.restol = 10 #1e-8 
cMR.inmaxite = 100
cMR.maxorcs = 1e6
cMR.lineMaxite = 100
cMR.lineBetaB = 1e-4
cMR.lineRho = 0.5
cMR.lineBetaFB = 1e-4

cCG_NC = const()
cCG_NC.alpha0 = 1
cCG_NC.gradtol = 1e-9
cCG_NC.maxite = 1e6
cCG_NC.restol = 0.99
cCG_NC.inmaxite = 100
cCG_NC.maxorcs = 1e6
cCG_NC.lineMaxite = 100
cCG_NC.lineBeta = 1e-4
cCG_NC.lineRho = 0.5
cCG_NC.epsilon = 0.99

cGD = const()
cGD.alpha0 = 1
cGD.gradtol = 1e-5
cGD.maxite = 1e5
cGD.maxorcs = 1e5
cGD.lineMaxite = 1000
cGD.lineBetaB = 1e-4
cGD.lineRho = 0.9

cTR_STEI = const()
cTR_STEI.gradtol = 1e-9
cTR_STEI.maxite = 1e6
cTR_STEI.inmaxite = 1000
cTR_STEI.maxorcs = 1e6
cTR_STEI.restol = 0.01                          
cTR_STEI.deltaMax = 1e10
cTR_STEI.delta0 = 1e5
cTR_STEI.eta = 0.01
cTR_STEI.eta1 = 1/4
cTR_STEI.eta2 = 3/4
cTR_STEI.gamma1 = 1/4
cTR_STEI.gamma2 = 2

cL_BFGS = const()
cL_BFGS.alpha0 = 1
cL_BFGS.gradtol = 1e-9
cL_BFGS.m = 20
cL_BFGS.maxite = 1e6
cL_BFGS.maxorcs = 1e6
cL_BFGS.lineMaxite = 100

cADAM = const()
cADAM.alpha0 = 0.00001
cADAM.beta1 = 0.9 #0.9
cADAM.beta2 = 0.999
cADAM.epsilon = 1e-8
cADAM.gradtol = 1e-9
cADAM.maxite = 1e8
cADAM.maxorcs = 1e6

cSGD = const()
cSGD.alpha0 = 0.5
cSGD.gradtol = 1e-5
cSGD.maxite = 1e8
cSGD.maxorcs = 1e5
