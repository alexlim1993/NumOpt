# -*- coding: utf-8 -*-
"""
Created on Tue Oct 25 14:54:22 2022

@author: uqalim8
"""
from linesearch import lineSearchWolfeStrong
import numpy as np
#from util import recordStat, printStat
import time

def twoloop(w, s, y):
    alpha, rho = [], []
    for si, yi in zip(s[::-1], y[::-1]):
        rho.append(1/(si.T @ yi))
        alpha.append(rho[-1] * (si.T @ w))
        w = w - alpha[-1] * yi
        
    w = ((s[-1].T @ y[-1]) / np.linalg.norm(y[-1])**2) * w
    k = len(s) - 1
    for si, yi in zip(s, y):
        beta = rho[k] * (yi.T @ w)
        w = w + (alpha[k] - beta) * si
        k -= 1
    return w

def L_BFGS(fun, x0, alpha0, tol = 1e-4, maxite = 1000):
    record = None
    tic = time.time()
    k = 0
    fk, gk = fun(x0, "01")
    gk_norm = np.linalg.norm(gk)
    xk = x0
    s, y = [], []
    while gk_norm > tol and k < maxite:
        if not k:
            pk = -gk
        else:
            pk = twoloop(-gk, s, y)
        
        alpha, _ = lineSearchWolfeStrong(lambda x : fun(x, "01"), xk, pk)
        xkp1 = xk + alpha * pk
        fkp1, gkp1 = fun(xkp1, "01")
        s.append(xkp1 - xk)
        y.append(gkp1 - gk)
        printStat(iters = k, inite = 0, ti = time.time() - tic, fk = fk, gk = gk_norm, alphak = 0)
        record = recordStat(record, iters = k, inters = 0, t = time.time() - tic, fk = fk, fk_norm = gk_norm, alpha = 0)

        if len(s) > 20:
            s = s[1:]
            y = y[1:]
        gk_norm = np.linalg.norm(gkp1)
        gk = gkp1
        xk = xkp1
        fk = fkp1
        k += 1
    return xk, record