# -*- coding: utf-8 -*-
"""
Created on Fri Jul 22 12:24:43 2022

@author: uqalim8
"""
import torch 

def logistic_regression(X, w, order_deriv = "012"):
    
    # Log-Sum-Exp trick
    # 1/(1 + exp(-t)) = e^t / (1 + e^t) = e^(t-c) / d
    # where d = exp(-c) + exp(t - c)
    t = torch.mv(X, w)
    c = torch.max(torch.zeros_like(t), t)
    d = torch.exp(-c) + torch.exp(t - c)
    s = torch.exp(t - c) / d
    g, H = None, None
    
    # gradient, Hessian
    if "1" in order_deriv:
        g = s * (1 - s)
        
    if "2" in order_deriv:
        H = s * (1 - s) * (1 - 2 * s)
        
    return s, g, H

def binary_nonLinearLeastSqaure(A, b, w):
    #n, _ = A.size
    expo = - A @ w
    c = torch.maximum(expo, torch.zeros_like(expo))
    expc = torch.exp(-c)
    de = expc + torch.exp(- c - expo)
    total = torch.sum((b - expc / de)**2)
    return total