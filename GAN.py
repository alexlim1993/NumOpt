# -*- coding: utf-8 -*-
"""
Created on Wed Mar  8 15:37:43 2023

@author: uqalim8
"""
import torch
import torch.nn as nn
import math
import neuralNetwork as neural

class Generator(nn.Module):
    
    def __init__(self, input_dim, dim, output_dim):
        super().__init__()
        self._dim = dim
        self._nn = nn.Sequential(nn.Conv2d(1, 4, 3),
                                 nn.Linear(input_dim, 32),
                                 #nn.BatchNorm1d(32, eps = 0, momentum = 0, affine = False, dtype = torch.float64),
                                 nn.Tanh(),
                                 nn.Linear(32, 64),
                                 #nn.BatchNorm1d(64, eps = 0, momentum = 0, affine = False, dtype = torch.float64),
                                 nn.Tanh(),
                                 nn.Linear(64, output_dim),
                                 nn.Tanh())

    def show_pic(self, x):
        x = x.reshape(-1, *self._dim)
        return self._nn(x)
    
    def forward(self, x):
        x = x.reshape(-1, *self._dim)
        x = self._nn(x)
        return self._nn(x)
    
class Discriminator(nn.Module):
    
    def __init__(self, input_dim):
        super().__init__()
        self._nn = nn.Sequential(nn.Linear(input_dim, 64),
                                 nn.Tanh(),
                                 nn.Linear(64, 1),
                                 nn.Sigmoid())

        
    def forward(self, x):
        return self._nn(x)

def GENLoss(fake, dummy, dis):
    """
    Generative model loss 
    see https://arxiv.org/pdf/1406.2661.pdf
    """
    
    pred = dis(fake)
    return -torch.sum(torch.log(pred)) / pred.shape[0]

def DISLoss(fake_real, dummy):
    """
    Discriminative model 'loss' (we want to maximize this)
    see https://arxiv.org/pdf/1406.2661.pdf
    """
    n = fake_real.shape[0] // 2
    fake, real = fake_real[:n], fake_real[n:]
    return torch.sum(torch.log(real) + torch.log(1 - fake)) / n

def GANWrapper(func, loss, fake, trainX, w, Hsub, reg, order):
    
    if not "2" in order or Hsub == 1:
        
        if not trainX is None:
            trainX = torch.concat([fake, trainX], dim = 0)
        else:
            trainX = fake
            
        return neural.nnWrapper(func, loss, trainX, None, w, 1, reg, order)
    
    print("!!!!!!!!!!!!!!!never!!!!!!!!!!!!!")
    
    n, _ = fake.shape
    m = math.ceil(n * Hsub)
    if not trainX is None:
        perm = torch.randperm(n)
        X = torch.concat([fake[:m], trainX[perm][:m]], dim = 0)
        trainX = torch.concat([fake, trainX], dim = 0)
    else:
        X = fake[:m]
        trainX = fake
    
    Hv = neural.nnWrapper(func, loss, X, None, w, 1, reg, "2")
    
    if "0" in order or "1" in order:
        return neural.nnWrapper(func, loss, trainX, None, w, 1, reg, order), Hv
    
    return Hv
    
def trainGAN(gen, dis, wG, wD, trainX, Hsub, reg, algG, algD, device):
    
    trainX = trainX * 2 - 1
    latent = torch.randn(trainX.shape, dtype = torch.float64, device = device)
            
    algD.fun = lambda w, order : GANWrapper(dis, lambda x, y : -DISLoss(x, None), gen(latent), trainX, w, Hsub, reg, order)
    
    #fun = lambda w : GAN.GANWrapper(dis, lambda x, y : -GAN.DISLoss(x, None), gen(trainX), trainX, w, 1, None, "012")
    
    algG.fun = lambda w, order : GANWrapper(gen, lambda x, y: GENLoss(x, None, dis), 
                                                 latent, None, w, Hsub, reg, order)
        
    algG.recordStats(lambda x : None)
    algD.recordStats(lambda x : None)
    j = 0
    while True:
        
        latent = torch.randn(trainX.shape, dtype = torch.float64, device = device)
            
        for _ in range(0):
            algD.fun = lambda w, order : GANWrapper(dis, lambda x, y : -DISLoss(x, y), gen(latent), trainX, 
                                                     w, Hsub, reg, order)
            algD.step()
            algD.progress(False, lambda x : None)
            
            print("..... Discriminator .....")
            print("fk:", float(algD.fk))
            print("gk:", float(torch.norm(algD.gk)))
            print("alphak:", float(algD.alphak))
            
            wD = algD.xk.clone()
            
            # we may be able to skip the following step (check!)
            nn.utils.vector_to_parameters(wD, dis.parameters())
            
        for _ in range(1):
            algG.fun = lambda w, order : GANWrapper(gen, lambda x, y: GENLoss(x, y, dis), 
                                                     latent, None, w, Hsub, reg, order)
            algG.step()
            algG.progress(False, lambda x : None)
            wG = algG.xk.clone()
            
            # we may be able to skip the following step (check!)
            nn.utils.vector_to_parameters(wG, gen.parameters())
            
    
            print("..... Generator .....")
            print("fk:", float(algG.fk))
            print("gk:", float(torch.norm(algG.gk)))
            print("alphak:", float(algG.alphak))
        
        j += 1
        print(j)
        if j % 25 == 0:
            print("here")