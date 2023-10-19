# -*- coding: utf-8 -*-
"""
Created on Fri Nov 25 14:57:40 2022

@author: uqalim8
"""
import torch

def non_convex(w, lamb):
    sqw = w ** 2
    return lamb * torch.sum(sqw / (1 + sqw))

def two_norm(w, lamb):
    return lamb * torch.dot(w, w)

def LASSO(w, lamb):
    return lamb * torch.linalg.norm(w, 1)

def none_reg(x, order):
    if order == "012":
        return 0, 0, lambda y : 0
    if order == "0" or order == "1":
        return 0
    if order == "01":
        return 0, 0
    if order == "2":
        return lambda y : 0
    if order == "02" or order == "12":
        return 0, lambda y : 0