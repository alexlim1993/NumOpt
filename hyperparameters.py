# -*- coding: utf-8 -*-
"""
Created on Tue Feb 14 11:23:10 2023

@author: uqalim8
"""
import utils

cMR = utils.const()
cMR.alpha0 = 1
cMR.gradtol = 1e-6
cMR.maxite = 1000
cMR.restol = 0.01
cMR.inmaxite = 100
cMR.maxorcs = 5000
cMR.lineMaxite = 100
cMR.lineBetaB = 1e-4
cMR.lineRho = 0.9
cMR.lineBetaFB = 0.25

cCG_NC = utils.const()
cCG_NC.alpha0 = 1
cCG_NC.gradtol = 1e-6
cCG_NC.maxite = 5000
cCG_NC.restol = 0.01#1e-4
cCG_NC.inmaxite = 1000
cCG_NC.maxorcs = 5000
cCG_NC.lineMaxite = 100
cCG_NC.lineBeta = 1e-4#0.01
cCG_NC.lineRho = 0.9
cCG_NC.epsilon = 1e-4

cGD = utils.const()
cGD.alpha0 = 1
cGD.gradtol = 1e-6
cGD.maxite = 5000
cGD.maxorcs = 1000
cGD.lineMaxite = 100
cGD.lineBetaB = 1e-4
cGD.lineRho = 0.9
