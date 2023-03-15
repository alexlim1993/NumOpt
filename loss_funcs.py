# -*- coding: utf-8 -*-
"""
Created on Fri Jul 22 15:54:53 2022

@author: uqalim8
"""
import torch

def nls(X, y, w, order = "012"):      
    n, d = X.shape
    f, g, H = None, None, None
    model_f, model_g, model_H = logisticRegression(X, w, order)
    if "0" in order:
       f = torch.sum((model_f - y) ** 2) / n
    
    if "1" in order:    
        g = 2 * torch.mv(X.T, model_g * (model_f - y)) / n
    
    if "2" in order:
        model_H = 2 * ((model_f - y) * model_H + model_g ** 2) / n
        H = lambda v : hess_vec_product(X, model_H, v)
    
    return f, g, H

def hess_vec_product(X, H, v):
    Xv = torch.mv(X, v)
    return torch.mv(X.T, H * Xv)

def logisticRegression(X, w, order):
    n, d = X.shape
    t = torch.mv(X, w)
    M = torch.max(torch.zeros_like(t),t)
    a = torch.exp(-M) + torch.exp(t-M)
    s = torch.exp(t-M)/a
    g, H = None, None
    
    if "1" in order:    
        g = s*(1-s)
    
    if "2" in order:    
        H = s*(1-s)*(1-2*s)

    return s, g, H

def logisticFun(x, A, b, reg, order_deriv = "012", H_matrix = False):
    Ax = torch.mv(A, x)
    c = torch.maximum(Ax, torch.zeros_like(Ax))
    expc = torch.exp(-c)
    expAx = torch.exp(Ax - c)
    f = torch.sum(c + torch.log(expc + expAx) - b * Ax) + 0.5 * reg * torch.linalg.norm(x)**2
    
    if "0" == order_deriv:
        return f
    
    t = expAx/(expc + expAx)
    if "01" == order_deriv:
        g = torch.sum((t - b).reshape(-1, 1) * A, axis = 0) + reg * x
        return f, g
    
    g = torch.sum((t - b).reshape(-1, 1) * A, axis = 0) + reg * x
    
    if H_matrix:
        H = A.T @ ((t * (1 - t)).reshape(-1, 1) * A) + reg * torch.eye(len(x))
        
    else:
        H = lambda v : A.T @ ((t * (1 - t)).reshape(-1, 1) * A @ v) + reg * v
    
    return f, g, H

def logisticModel(A, w):
    expo = - torch.mv(A, w)
    c = torch.maximum(expo, torch.zeros_like(expo, dtype = torch.float64))
    expc = torch.exp(-c)
    de = expc + torch.exp(- c + expo)
    return expc / de

if __name__ == "__main__":
    import torch
    from derivativeTest import derivativeTest
    
    n, d = 1000, 50
    A = torch.randn((n, d), dtype = torch.float64)
    b = torch.randint(0, 2, (n,))
    fun = lambda x : logisticFun(x, A, b, 1, "012")
    derivativeTest(fun, torch.ones(d, dtype = torch.float64))