# -*- coding: utf-8 -*-
"""
Created on Mon Oct 24 18:31:13 2022

@author: uqalim8
"""
from linesearch import backwardArmijo, backForwardArmijo, dampedNewtonCGLinesearch
from loss_funcs import logisticFun
from loadData import loadData
import torch
from optimizer import *
from CG import CG, CappedCG
from MINRES import myMINRES

class linesearchGD(Optimizer):
    
    def __init__(self, fun, x0, alpha0, gradtol, maxite, maxorcs, lineMaxite, lineBetaB, lineRho):
        self.info = GD_STATS
        self.lineMaxite = lineMaxite
        self.lineBetaB = lineBetaB
        self.lineRho = lineRho
        super().__init__(fun, x0, alpha0, gradtol, maxite, maxorcs)
        
    def step(self):
        pk = -self.gk
        self.alphak, self.lineite = backwardArmijo(lambda x : self.fun(x, "0"), self.xk, self.fk, self.gk, self.alpha0, pk,
                                     self.lineBetaB, self.lineRho, self.lineMaxite)
        self.xk += self.alphak * pk
        self.fk, self.gk = self.fun(self.xk, "01")
        self.gknorm = torch.linalg.norm(self.gk, 2)

    def recordStats(self):
        if self.k == 0:
            self.fk, self.gk = self.fun(self.xk, "01")
            self.gknorm = torch.linalg.norm(self.gk, 2)
            self.recording((0, 0, 0, float(self.fk), float(self.gknorm), 0))
        else:
            self.recording((self.k, self.orcs, self.toc, 
                               float(self.fk), float(self.gknorm), self.alphak))
        
    def oracleCalls(self):
        self.orcs += 2 + self.lineite
        
class NewtonCG(Optimizer):
    
    def __init__(self, fun, x0, alpha0, gradtol, maxite, maxorcs, restol, inmaxite, 
                 lineMaxite, lineBetaB, lineRho):
        self.info = NEWTON_STATS
        self.restol = restol
        self.inmaxite = inmaxite
        self.lineMaxite = lineMaxite
        self.lineBetaB = lineBetaB
        self.lineRho = lineRho
        super().__init__(fun, x0, alpha0, gradtol, maxite, maxorcs)
        
    def step(self):
        pk, self.inite = CG(self.hk, -self.gk, tol = self.restol, maxite = self.inmaxite)
        self.alphak, self.lineite = backwardArmijo(lambda x : self.fun(x, "0"), self.xk, self.fk, self.gk, self.alpha0, pk,
                                     self.lineBetaB, self.lineRho, self.lineMaxite)
        self.xk += self.alphak * pk
        self.fk, self.gk, self.hk = self.fun(self.xk, "012")
        self.gknorm = torch.linalg.norm(self.gk, 2)
    
    def recordStats(self):
        if self.k == 0:
            self.fk, self.gk, self.hk = self.fun(self.xk, "012")
            self.inite = 0
            self.gknorm = torch.linalg.norm(self.gk, 2)
            self.recording((0, 0, 0, 0, float(self.fk), float(self.gknorm), 0))
        else:
            self.recording((self.k, self.inite, self.orcs, self.toc, 
                               float(self.fk), float(self.gknorm), self.alphak))
        
    def oracleCalls(self):
        self.orcs += 2 + 2 * self.inite + self.lineite
        
class NewtonCG_NC(Optimizer):
    
    # Without second order optimality 
    # Simplified, i.e. without minimum eigenvalue oracle
    
    def __init__(self, fun, x0, alpha0, gradtol, maxite, maxorcs, restol, inmaxite,
                 lineMaxite, lineBeta, lineRho, epsilon, Hsub):
        self.info = NEWTON_NC_STATS
        self.restol = restol
        self.inmaxite = inmaxite
        self.lineMaxite = lineMaxite
        self.lineBeta = lineBeta
        self.lineRho = lineRho
        self.epsilon = epsilon
        self.Hsub = Hsub
        super().__init__(fun, x0, alpha0, gradtol, maxite, maxorcs)
    
    def step(self):
        pk, self.dtype, self.inite, pHp, _ = CappedCG(self.hk, -self.gk, self.restol, self.epsilon, self.inmaxite)
        normpk = torch.linalg.norm(pk, 2)**3
        if self.dtype == "NC":
            pk = - torch.sign(torch.dot(pk, self.gk)) * abs(pHp) * pk / normpk
        self.alphak, self.lineite = dampedNewtonCGLinesearch(lambda x : self.fun(x, "0"), self.xk, self.fk, self.alpha0, pk, 
                                                             normpk, self.lineBeta, self.lineRho, self.lineMaxite)
        self.xk += self.alphak * pk
        self.fk, self.gk, self.hk = self.fun(self.xk, "012")
        
    def recordStats(self):
        if self.k == 0:
            self.fk, self.gk, self.hk = self.fun(self.xk, "012")
            self.inite = 0
            self.gknorm = torch.linalg.norm(self.gk, 2)
            self.recording((0, 0, "None", 0, 0, float(self.fk), 
                             float(self.gknorm), 0))
        else:
            self.gknorm = torch.linalg.norm(self.gk, 2)
            self.recording((self.k, self.inite, self.dtype, self.orcs, 
                               self.toc, float(self.fk), float(self.gknorm), self.alphak))
            
    def oracleCalls(self):
        self.orcs += 2 + 2 * self.inite * self.Hsub + self.lineite
        
class NewtonMR_NC(Optimizer):
    
    def __init__(self, fun, x0, alpha0, gradtol, maxite, maxorcs, restol, inmaxite, 
                 lineMaxite, lineBetaB, lineRho, lineBetaFB, Hsub):
        self.info = NEWTON_NC_STATS
        self.restol = restol
        self.inmaxite = inmaxite
        self.lineMaxite = lineMaxite
        self.lineBetaB = lineBetaB
        self.lineRho = lineRho
        self.lineBetaFB = lineBetaFB
        self.Hsub = Hsub
        super().__init__(fun, x0, alpha0, gradtol, maxite, maxorcs)
        
    def step(self):
        pk, _, self.inite, r, self.dtype = myMINRES(self.hk, -self.gk, rtol = self.restol, maxit = self.inmaxite)
        if self.dtype == "Sol":
            self.alphak, self.lineite = backwardArmijo(lambda x : self.fun(x, "0"), self.xk, self.fk, self.gk, self.alpha0, pk,
                                         self.lineBetaB, self.lineRho, self.lineMaxite)
        else:
            self.alphak, self.lineite = backForwardArmijo(lambda x : self.fun(x, "0"), self.xk, self.fk, self.gk, self.alpha0, pk,
                                            self.lineBetaFB, self.lineRho, self.lineMaxite)
        self.xk += self.alphak * pk
        self.fk, self.gk, self.hk = self.fun(self.xk, "012")
        
    def recordStats(self):
        if self.k == 0:
            self.fk, self.gk, self.hk = self.fun(self.xk, "012")
            self.inite = 0
            self.gknorm = torch.linalg.norm(self.gk, 2)
            self.recording((0, 0, "None", 0, 0, float(self.fk), 
                             float(self.gknorm), 0))
        else:
            self.gknorm = torch.linalg.norm(self.gk, 2)
            self.recording((self.k, self.inite, self.dtype, self.orcs, 
                               self.toc, float(self.fk), float(self.gknorm), self.alphak))
        
    def oracleCalls(self):
        self.orcs += 2 + 2 * self.inite * self.Hsub + self.lineite
    
if __name__ == "__main__":
    A_train, b_train, *_ = loadData()
    fun = lambda x, v : logisticFun(x, A_train, b_train, 1, v)
    x0 = torch.zeros(A_train.shape[-1], dtype = torch.float64)
    Lg = torch.linalg.matrix_norm(A_train, 2)**2 / 4 + 1
    optGD = linesearchGD(fun, x0.clone(), 10/Lg, 10e-4, 1000, 1000, 100, 10e-4, 0.9)
    optGD.optimize(True)
    optNEWTON = NewtonCG(fun, x0.clone(), 1, 10e-4, 1000, 1000, 10e-2, 100, 100, 10e-4, 0.9)
    optNEWTON.optimize(True)
    optNEWTONMR = NewtonMR_NC(fun, x0.clone(), 1, 10e-4, 1000, 1000, 10e-2, 100, 100, 10e-4, 0.9, 10e-4, 1)
    optNEWTONMR.optimize(True)
    optNEWTONCG = NewtonCG_NC(fun, x0.clone(), 1, 10e-4, 1000, 1000, 10e-2, 100, 100, 10e-4, 0.9, 10e-4, 1)
    optNEWTONCG.optimize(True)
    #xk, record3 = L_BFGS(fun, x0.copy(), 1)
    
    import matplotlib.pyplot as plt
    
    fig = plt.figure()
    plt.loglog(torch.tensor(optGD.record["ite"]) + 1, optGD.record["f"], label = "GD")
    plt.loglog(torch.tensor(optNEWTON.record["ite"]) + 1, optNEWTON.record["f"], label = "NewtonCG")
    plt.loglog(torch.tensor(optNEWTONMR.record["ite"]) + 1, optNEWTONMR.record["f"], label = "NewtonMR")
    plt.loglog(torch.tensor(optNEWTONCG.record["ite"]) + 1, optNEWTONCG.record["f"], label = "NewtonCG_NC")

    plt.legend()
    plt.show()
    
    fig = plt.figure()
    plt.loglog(torch.tensor(optGD.record["ite"]) + 1, optGD.record["g_norm"], label = "GD")
    plt.loglog(torch.tensor(optNEWTON.record["ite"]) + 1, optNEWTON.record["g_norm"], label = "NewtonCG")
    plt.loglog(torch.tensor(optNEWTONMR.record["ite"]) + 1, optNEWTONMR.record["g_norm"], label = "NewtonMR")
    plt.loglog(torch.tensor(optNEWTONCG.record["ite"]) + 1, optNEWTONCG.record["g_norm"], label = "NewtonCG_NC")
    plt.legend()
    plt.show()