# -*- coding: utf-8 -*-
"""
Created on Thu Nov 24 12:35:21 2022

@author: uqalim8
"""

import torch
import math
from derivativeTest import derivativeTest

def funcWrapper(func, A, b, w, Hsub, reg, order):

    if "012" == order and Hsub != 1:

        if not reg is None: 
            reg_f, reg_g, reg_H = fgHv(reg, w)
        else:
            reg_f, reg_g, reg_H = 0, 0, lambda v : 0
            
        n, _ = A.shape
        m = math.ceil(n * Hsub)
        perm = torch.randperm(n)
        
        f, g = fgHv(lambda v : func(A, b, v), w, "01")
        H = fgHv(lambda v : func(A[perm, :][:m, :], b[perm][:m], v), w, "2")
        # f1, g1, H = fgHv(lambda v : func(A[perm, :][:m, :], b[perm][:m], v),
        #                  w, "012")
        
        # f2, g2 = fgHv(lambda v : func(A[perm, :][m:, :], b[perm][m:], v),
        #               w, "01")
        
        # f = m * f1 / n + (n - m) * f2 / n + reg_f
        # g = m * g1 / n + (n - m) * g2 / n + reg_g
        # Hv = lambda v : H(v) + reg_H(v)
        f = f + reg_f
        g = g + reg_g
        Hv = lambda v : H(v) + reg_H(v)
        return f, g, Hv
    
    if not reg is None:
        return fgHv(lambda v : func(A, b, v) + reg(v), w, order)
    
    return fgHv(lambda v : func(A, b, v), w, order)
        
        
def fgHv(func, w, order = "012"):
    
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
    
    binary logistic regression with mean square error loss.
    (non-convex function)
    """
    n, _ = A.shape
    Aw = - torch.mv(A, w)
    c = torch.maximum(Aw, torch.zeros_like(Aw, dtype = torch.float64))
    expc = torch.exp(-c)
    de = expc + torch.exp(Aw - c)
    total = torch.sum((b - expc / de)**2)
    return total / n

def logloss(A, b, w):
    """
    Binary Logistic Loss function
    
    binary logloss (convex function)
    """
    Aw = - torch.mv(A, w)
    c = torch.maximum(Aw, torch.zeros_like(Aw, dtype = torch.float64))
    expc = torch.exp(-c)
    return torch.sum(c + torch.log(expc + torch.exp(Aw - c)) - b * Aw)

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