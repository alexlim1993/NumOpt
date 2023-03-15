# -*- coding: utf-8 -*-
"""
Created on Mon Oct 24 18:31:13 2022

@author: uqalim8
"""
from linesearch import (backwardArmijo, 
                        backForwardArmijo, 
                        dampedNewtonCGLinesearch, 
                        dampedNewtonCGbackForwardLS, 
                        lineSearchWolfeStrong)
import torch
from optimizer import *
from CG import CG, CappedCG, CGSteihaug
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

    def recordStats(self, acc):
        if self.k == 0:
            self.fk, self.gk = self.fun(self.xk, "01")
            self.gknorm = torch.linalg.norm(self.gk, 2)
            self.recording((0, 0, 0, float(self.fk), float(self.gknorm), 0, acc))
        else:
            self.recording((self.k, self.orcs, self.toc, 
                               float(self.fk), float(self.gknorm), self.alphak, acc))
        
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
    
    def recordStats(self, acc):
        if self.k == 0:
            self.fk, self.gk, self.hk = self.fun(self.xk, "012")
            self.inite = 0
            self.gknorm = torch.linalg.norm(self.gk, 2)
            self.recording((0, 0, 0, 0, float(self.fk), float(self.gknorm), 0, acc))
        else:
            self.recording((self.k, self.inite, self.orcs, self.toc, 
                               float(self.fk), float(self.gknorm), self.alphak, acc))
        
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
        
    def recordStats(self, acc):
        if self.k == 0:
            self.fk, self.gk, self.hk = self.fun(self.xk, "012")
            self.inite = 0
            self.gknorm = torch.linalg.norm(self.gk, 2)
            self.recording((0, 0, "None", 0, 0, float(self.fk), 
                             float(self.gknorm), 0, acc))
        else:
            self.gknorm = torch.linalg.norm(self.gk, 2)
            self.recording((self.k, self.inite, self.dtype, self.orcs, 
                               self.toc, float(self.fk), float(self.gknorm), self.alphak, acc))
            
    def oracleCalls(self):
        self.orcs += 2 + 2 * self.inite * self.Hsub + self.lineite
        
class NewtonCG_NC_FW(NewtonCG_NC):

    def step(self):
        pk, self.dtype, self.inite, pHp, _ = CappedCG(self.hk, -self.gk, self.restol, self.epsilon, self.inmaxite)
        normpkcubed = torch.linalg.norm(pk, 2)**3
        if self.dtype == "NC":
            pk = - torch.sign(torch.dot(pk, self.gk)) * abs(pHp) * pk / normpkcubed
            self.alphak, self.lineite = dampedNewtonCGbackForwardLS(lambda x : self.fun(x, "0"), self.xk, self.fk, self.alpha0, pk, 
                                                                 normpkcubed, self.lineBeta, self.lineRho, self.lineMaxite)
        else:
            self.alphak, self.lineite = dampedNewtonCGLinesearch(lambda x : self.fun(x, "0"), self.xk, self.fk, self.alpha0, pk, 
                                                             normpkcubed, self.lineBeta, self.lineRho, self.lineMaxite)
        
        self.xk += self.alphak * pk
        self.fk, self.gk, self.hk = self.fun(self.xk, "012")
         
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
        if self.dtype == "Sol" or self.dtype == "MAX":
            self.alphak, self.lineite = backwardArmijo(lambda x : self.fun(x, "0"), 
                                                       self.xk, self.fk, self.gk, self.alpha0, pk, 
                                                       self.lineBetaB, self.lineRho, self.lineMaxite)
        else:
            self.alphak, self.lineite = backForwardArmijo(lambda x : self.fun(x, "0"), 
                                                          self.xk, self.fk, self.gk, self.alpha0, r, 
                                                          self.lineBetaFB, self.lineRho, self.lineMaxite)
            pk = r
            
        assert torch.dot(pk, self.gk) < 0
        self.xk += self.alphak * pk
        self.fk, self.gk, self.hk = self.fun(self.xk, "012")
        
    def recordStats(self, acc):
        if self.k == 0:
            self.fk, self.gk, self.hk = self.fun(self.xk, "012")
            self.inite = 0
            self.gknorm = torch.linalg.norm(self.gk, 2)
            self.recording((0, 0, "None", 0, 0, float(self.fk), 
                             float(self.gknorm), 0, acc))
        else:
            self.gknorm = torch.linalg.norm(self.gk, 2)
            self.recording((self.k, self.inite, self.dtype, self.orcs, 
                               self.toc, float(self.fk), float(self.gknorm), self.alphak, acc))
        
    def oracleCalls(self):
        self.orcs += 2 + 2 * self.inite * self.Hsub + self.lineite
        
class NewtonCG_TR_Steihaug(Optimizer):
    
    def __init__(self, fun, x0, alpha0, gradtol, maxite, maxorcs, restol, inmaxite, 
                 deltaMax, delta0, eta, eta1, eta2, gamma1, gamma2, Hsub):
        
        if not (0 < eta1 and eta1 <= eta2 and eta2 < 1 and eta < eta1):
            raise Exception("etas 0 < eta < eta1 <= eta2 < 1")
        
        if not ((0 < gamma1 and gamma1 < 1) and (gamma2 > 1)):
            raise Exception("0 < gamma1 < 1 and gamma2 > 1")
        
        self.info = NEWTON_TR_STATS
        self.restol = restol
        self.inmaxite = inmaxite
        self.delta = delta0
        self.deltaMax = deltaMax
        self.eta = eta
        self.eta1 = eta1
        self.eta2 = eta2
        self.gamma1 = gamma1
        self.gamma2 = gamma2
        self.Hsub = Hsub
        super().__init__(fun, x0, 0, gradtol, maxite, maxorcs)
        
    def step(self):
        pk, self.dtype, m, self.inite = CGSteihaug(self.hk, self.gk, self.delta, self.restol, self.inmaxite)
        self.rho = (self.fk - self.fun(self.xk + pk, "0")) / m
        
        if self.rho < self.eta1:
            self.delta *= self.gamma1
        
        else:
            if self.rho > self.eta2 and self.dtype == "SOL,=":
                self.delta = min(self.delta * self.gamma2, self.deltaMax)
        
        if self.rho > self.eta:
            self.xk = self.xk + pk
            self.fk, self.gk, self.hk = self.fun(self.xk, "012")
            
    def recordStats(self, acc):
        if self.k == 0:
            self.fk, self.gk, self.hk = self.fun(self.xk, "012")
            self.inite = 0
            self.gknorm = torch.linalg.norm(self.gk, 2)
            self.recording((0, 0, "None", 0, 0, float(self.fk), 
                             float(self.gknorm), 0, acc))
        else:
            self.gknorm = torch.linalg.norm(self.gk, 2)
            self.recording((self.k, self.inite, self.dtype, self.orcs, 
                               self.toc, float(self.fk), float(self.gknorm), self.delta, acc))  
            
    def oracleCalls(self):
        self.orcs += 2 + 2 * self.inite * self.Hsub + 2

class L_BFGS(Optimizer):

    def __init__(self, fun, x0, alpha0, gradtol, m, maxite, maxorcs, lineMaxite):
        self.info = L_BFGS_STATS
        self.m = m
        self.lineMaxite = lineMaxite
        self.s, self.y = [], []
        super().__init__(fun, x0, alpha0, gradtol, maxite, maxorcs)
    
    def _twoloop(self, w, s, y):
        alpha, rho = [], []
        for si, yi in zip(s[::-1], y[::-1]):
            rho.append(1/(torch.dot(si, yi)))
            alpha.append(rho[-1] * (torch.dot(si, w)))
            w = w - alpha[-1] * yi
            
        w = ((torch.dot(s[-1], y[-1])) / torch.dot(y[-1], y[-1])) * w
        k = len(s) - 1
        for si, yi in zip(s, y):
            beta = rho[k] * (torch.dot(yi, w))
            w = w + (alpha[k] - beta) * si
            k -= 1
        return w
    
    def step(self):
        
        if not self.k:
            pk = -self.gk
        else:
            pk = self._twoloop(-self.gk, self.s, self.y)
        
        self.alpha, self.lineite = lineSearchWolfeStrong(lambda x : self.fun(x, "01"), self.xk, pk, 
                                                         self.alpha0, 1e-4, 0.9, self.lineMaxite)
        xkp1 = self.xk + self.alpha * pk
        self.fk, gkp1 = self.fun(xkp1, "01")
        self.s.append(xkp1 - self.xk)
        self.y.append(gkp1 - self.gk)
        self.gk = gkp1
        self.xk = xkp1
        if len(self.s) > self.m:
            self.s = self.s[1:]
            self.y = self.y[1:]

    def recordStats(self, acc):
        if self.k == 0:
            self.fk, self.gk = self.fun(self.xk, "01")
            self.inite = 0
            self.gknorm = torch.linalg.norm(self.gk, 2)
            self.recording((0, 0, 0, float(self.fk), 
                             float(self.gknorm), 0, acc))
        else:
            self.gknorm = torch.linalg.norm(self.gk, 2)
            self.recording((self.k, self.orcs, self.toc, float(self.fk), 
                            float(self.gknorm), self.alpha, acc))  
            
    def oracleCalls(self):
        self.orcs += 2 + 2 * self.lineite

if __name__ == "__main__":
    from loss_funcs import logisticFun, logisticModel
    from loadData import loadData
    
    A_train, b_train, *_ = loadData()
    fun = lambda x, v : logisticFun(x, A_train, b_train, 1, v)
    x0 = torch.zeros(A_train.shape[-1], dtype = torch.float64)
    Lg = torch.linalg.matrix_norm(A_train, 2)**2 / 4 + 1
    
    def pred(x):
        b_pred = torch.round(logisticModel(A_train, x))
        return torch.sum(b_train == b_pred) / len(b_train) * 100
    
    print("=================== LineSearch GD ========================")
    optGD = linesearchGD(fun, x0.clone(), 10/Lg, 10e-4, 1000, 1000, 100, 10e-4, 0.9)
    optGD.optimize(True, pred)
    print("=================== Newton CG ========================")
    optNEWTON = NewtonCG(fun, x0.clone(), 1, 10e-4, 1000, 1000, 10e-2, 100, 100, 10e-4, 0.9)
    optNEWTON.optimize(True, pred)
    
    print("=================== Newton MR NC ========================")
    optNEWTONMR = NewtonMR_NC(fun, x0.clone(), 1, 10e-4, 1000, 1000, 10e-2, 100, 100, 10e-4, 0.9, 10e-4, 1)
    optNEWTONMR.optimize(True, pred)
    
    print("=================== Newton CG NC ========================")
    optNEWTONCG = NewtonCG_NC(fun, x0.clone(), 1, 10e-4, 1000, 1000, 10e-2, 100, 100, 10e-4, 0.9, 10e-4, 1)
    optNEWTONCG.optimize(True, pred)
    
    print("=================== L-BFGS ========================")
    optL_BFGS = L_BFGS(fun, x0.clone(), 1, 10e-4, 20, 1000, 1000, 1000)
    optL_BFGS.optimize(True, pred)
    
    print("=================== Newton TR ========================")
    optNewtonTR = NewtonCG_TR_Steihaug(fun, x0.clone(), None, 10e-4, 1000, 1000, 
                                       10e-2, 100, 1e10, 1e5, 1/8, 0.25, 0.75, 0.25, 2, 1)
    optNewtonTR.optimize(True, pred)
    
    import matplotlib.pyplot as plt
    
    fig = plt.figure()
    plt.loglog(torch.tensor(optGD.record["orcs"]) + 1, optGD.record["f"], label = "GD")
    plt.loglog(torch.tensor(optNEWTON.record["orcs"]) + 1, optNEWTON.record["f"], label = "NewtonCG")
    plt.loglog(torch.tensor(optNEWTONMR.record["orcs"]) + 1, optNEWTONMR.record["f"], label = "NewtonMR_NC")
    plt.loglog(torch.tensor(optNEWTONCG.record["orcs"]) + 1, optNEWTONCG.record["f"], label = "NewtonCG_NC")
    plt.loglog(torch.tensor(optNewtonTR.record["orcs"]) + 1, optNewtonTR.record["f"], label = "NewtonTR")
    plt.loglog(torch.tensor(optL_BFGS.record["orcs"]) + 1, optL_BFGS.record["f"], label = "L_BFGS")

    plt.legend()
    plt.show()
    
    fig = plt.figure()
    plt.loglog(torch.tensor(optGD.record["orcs"]) + 1, optGD.record["g_norm"], label = "GD")
    plt.loglog(torch.tensor(optNEWTON.record["orcs"]) + 1, optNEWTON.record["g_norm"], label = "NewtonCG")
    plt.loglog(torch.tensor(optNEWTONMR.record["orcs"]) + 1, optNEWTONMR.record["g_norm"], label = "NewtonMR_NC")
    plt.loglog(torch.tensor(optNEWTONCG.record["orcs"]) + 1, optNEWTONCG.record["g_norm"], label = "NewtonCG_NC")
    plt.loglog(torch.tensor(optNewtonTR.record["orcs"]) + 1, optNewtonTR.record["g_norm"], label = "NewtonTR")
    plt.loglog(torch.tensor(optL_BFGS.record["orcs"]) + 1, optL_BFGS.record["g_norm"], label = "L_BFGS")

    plt.legend()
    plt.show()