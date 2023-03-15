# -*- coding: utf-8 -*-
"""
Created on Thu Nov 24 12:35:21 2022

@author: uqalim8
"""

import torch, GAN
import math
from derivativeTest import derivativeTest

def funcWrapper(func, A, b, w, Hsub, reg, order):
    
    if Hsub == 1 or not "2" in order:
        if not reg is None:
            return fgHv(lambda v : func(A, b, v) + reg(v), w, order)
        else:
            return fgHv(lambda v : func(A, b, v), w, order)
        
    if Hsub != 1:
        
        n, d = A.shape
        m = math.ceil(n * Hsub)
        perm = torch.randperm(n)
        reg_f, reg_g, reg_H = 0, 0, lambda v : 0
        
        if "2" == order:
            if not reg is None:
                reg_H = fgHv(reg, w, "2")
                
            H = fgHv(lambda v : func(A[perm, :][:m, :], b[perm][:m], v),
                      w, "2")
            return lambda v : H(v) + reg_H(v)

        if "012" == order:
            if not reg is None:
                reg_f, reg_g, reg_H = fgHv(reg, w, "012")
                
            f1, g1, H = fgHv(lambda v : func(A[perm, :][:m, :], b[perm][:m], v),
                             w, "012")
        
            f2, g2 = fgHv(lambda v : func(A[perm, :][m:, :], b[perm][m:], v),
                          w, "01")
            
            # do we always need to take mean?
            # will there be a function that it doesn't take mean?
            # should mean be taken here or within the function itself?
            f = m * f1 / n + (n - m) * f2 / n + reg_f
            g = m * g1 / n + (n - m) * g2 / n + reg_g
            Hv = lambda v : H(v) + reg_H(v)
            
            # ran = torch.rand(d, dtype = torch.float64)
            # Hv = lambda v : ran * torch.dot(ran, v)
            return f, g, Hv
        
def fgHv(func, w, order = "012"):
    
    with torch.no_grad():
        x = w.clone().requires_grad_(True)
    f = func(x)
    
    if "0" == order:
        return f.detach()
    
    if "01" == order:
        g = torch.autograd.grad(f, x, create_graph = False, retain_graph = True)[0]
        return f.detach(), g.detach()
    
    if "2" == order:
        g = torch.autograd.grad(f, x, create_graph = True, retain_graph = True)[0]
        Hv = lambda v : torch.autograd.grad((g,), x, v, create_graph = False, retain_graph = True)[0].detach()
        return Hv
    
    if "012" == order:
        g = torch.autograd.grad(f, x, create_graph = True, retain_graph = True)[0]
        Hv = lambda v : torch.autograd.grad((g,), x, v, create_graph = False, retain_graph = True)[0].detach()
        return f.detach(), g.detach(), Hv

def nls(A, b, w):
    """
    Non-linear Least Square (NLS)
    
    !! This has not been averaged !!
    
    binary logistic regression with sum of square error loss (mean or not).
    (non-convex function)
    """
    Aw = - torch.mv(A, w)
    c = torch.maximum(Aw, torch.zeros_like(Aw, dtype = torch.float64))
    expc = torch.exp(-c)
    de = expc + torch.exp(Aw - c)
    total = torch.sum((b - expc / de)**2)
    return total / A.shape[0]

def logloss(A, b, w):
    """
    Binary Logistic Loss function
    
    binary logloss (convex function)
    """
    Aw = - torch.mv(A, w)
    c = torch.maximum(Aw, torch.zeros_like(Aw, dtype = torch.float64))
    expc = torch.exp(-c)
    return torch.sum(c + torch.log(expc + torch.exp(Aw - c)) - b * Aw) / A.shape[0]

def logisticModel(A, w):
    expo = - torch.mv(A, w)
    c = torch.maximum(expo, torch.zeros_like(expo, dtype = torch.float64))
    expc = torch.exp(-c)
    de = expc + torch.exp(- c + expo)
    return expc / de

if __name__ == "__main__":
    n, d = 1000, 50
    A = torch.randn((n, d), dtype = torch.float64)
    b = torch.randint(0, 2, (n,))
    fun = lambda x: fgHv(lambda v : logloss(A, b, v),
                         x.clone().detach().requires_grad_(True), "012")
    derivativeTest(fun, torch.ones(d, dtype = torch.float64))