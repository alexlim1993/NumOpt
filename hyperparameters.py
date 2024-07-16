# -*- coding: utf-8 -*-
"""
Created on Tue Feb 14 11:23:10 2023

@author: uqalim8
"""
import torch

cTYPE = torch.float64
cCUDA = True

if cCUDA:
    cCUDA = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
else:
    cCUDA = torch.device("cpu")

class const():
    pass
    
MAXITE = 1e6
MAXORCS = 1e5
GRADTOL = 1e-6

INMAXITE = 1000
SIGMA = 1e-32

ALPHA0 = 1
LINEMAXITE = 100
LINEBETA_B = 1e-4
LINEBETA_FB = 1e-4
LINERHO = 0.5

######################################################
######################################################
#######################         ######################
#######################  MR_NC  ######################
#######################         ######################
######################################################
######################################################
cMR = const()

cMR.gradtol = GRADTOL
cMR.maxite = MAXITE
cMR.maxorcs = MAXORCS

cMR.alpha0 = ALPHA0
cMR.restol = 100
cMR.inmaxite = INMAXITE
cMR.sigma = SIGMA

cMR.lineMaxite = LINEMAXITE
cMR.lineBetaB = LINEBETA_B
cMR.lineRho = LINERHO
cMR.lineBetaFB = LINEBETA_FB

######################################################
######################################################
#######################         ######################
#######################  CG_NC  ######################
#######################         ######################
######################################################
######################################################
cCG_NC = const()

cCG_NC.alpha0 = ALPHA0
cCG_NC.gradtol = GRADTOL
cCG_NC.maxite = MAXITE
cCG_NC.restol = 0.1
cCG_NC.inmaxite = INMAXITE
cCG_NC.maxorcs = MAXORCS
cCG_NC.lineMaxite = LINEMAXITE
cCG_NC.lineBeta = LINEBETA_B
cCG_NC.lineRho = LINERHO
cCG_NC.epsilon = 0.999

######################################################
######################################################
#######################         ######################
#######################  LBFGS  ######################
#######################         ######################
######################################################
######################################################
cL_BFGS = const()
cL_BFGS.alpha0 = 1
cL_BFGS.gradtol = GRADTOL
cL_BFGS.m = 20
cL_BFGS.maxite = MAXITE
cL_BFGS.maxorcs = MAXORCS
cL_BFGS.lineMaxite = LINEMAXITE

cGD = const()
cGD.alpha0 = 1
cGD.gradtol = 1e-5
cGD.maxite = 1e5
cGD.maxorcs = 1e5
cGD.lineMaxite = 1000
cGD.lineBetaB = 1e-4
cGD.lineRho = 0.9

cTR_STEI = const()
cTR_STEI.gradtol = GRADTOL
cTR_STEI.maxite = MAXITE
cTR_STEI.inmaxite = INMAXITE
cTR_STEI.maxorcs = MAXORCS
cTR_STEI.restol = 0.1                          
cTR_STEI.deltaMax = 1e10
cTR_STEI.delta0 = 1e5
cTR_STEI.eta = 0.01
cTR_STEI.eta1 = 1/4
cTR_STEI.eta2 = 3/4
cTR_STEI.gamma1 = 1/4
cTR_STEI.gamma2 = 2

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
