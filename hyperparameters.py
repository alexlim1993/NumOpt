# -*- coding: utf-8 -*-
"""
Created on Tue Feb 14 11:23:10 2023

@author: uqalim8
"""
import utils

cMR = utils.const()
cMR.alpha0 = 1
cMR.gradtol = 1e-5
cMR.maxite = 1e5
cMR.restol = 0.01
cMR.inmaxite = 1000
cMR.maxorcs = 1e5
cMR.lineMaxite = 1000
cMR.lineBetaB = 1e-4
cMR.lineRho = 0.8
cMR.lineBetaFB = 0.25

cCG_NC = utils.const()
cCG_NC.alpha0 = 1
cCG_NC.gradtol = 1e-5
cCG_NC.maxite = 100000
cCG_NC.restol = 0.01#1e-4
cCG_NC.inmaxite = 100
cCG_NC.maxorcs = 20000
cCG_NC.lineMaxite = 100
cCG_NC.lineBeta = 1e-4#0.01
cCG_NC.lineRho = 0.9
cCG_NC.epsilon = 1e-4

cGD = utils.const()
cGD.alpha0 = 1
cGD.gradtol = 1e-5
cGD.maxite = 1e5
cGD.maxorcs = 1e5
cGD.lineMaxite = 1000
cGD.lineBetaB = 1e-4
cGD.lineRho = 0.9

cTR_STEI = utils.const()
cTR_STEI.gradtol = 1e-5
cTR_STEI.maxite = 1000
cTR_STEI.inmaxite = 100
cTR_STEI.maxorcs = 5000
cTR_STEI.restol = 0.01 # should this be the same as gradtol                            
cTR_STEI.deltaMax = 1e10
cTR_STEI.delta0 = 1e5
cTR_STEI.eta = 0.01
cTR_STEI.eta1 = 1/4
cTR_STEI.eta2 = 3/4
cTR_STEI.gamma1 = 1/4
cTR_STEI.gamma2 = 2