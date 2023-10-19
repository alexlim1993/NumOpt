# -*- coding: utf-8 -*-
"""
Created on Mon Jan  9 12:33:43 2023

@author: uqalim8
"""

import torch.nn as nn
import torch, math, tests
from functorch import make_functional
from hyperparameters import cCUDA, cSPLIT, cTYPE
from regularizers import none_reg

class auto_Encoder_MNIST(nn.Module):
    def __init__(self):
        super(auto_Encoder_MNIST, self).__init__()
        self.encoder = nn.Sequential(nn.Linear(28*28, 512),
                                      nn.Tanh(),
                                      nn.Linear(512, 256),
                                      nn.Tanh(),
                                      nn.Linear(256, 128),
                                      nn.Tanh(),
                                      nn.Linear(128, 64),
                                      nn.Tanh(),
                                      nn.Linear(64, 32),
                                      nn.Tanh(),
                                      nn.Linear(32, 16),
                                      )
        self.decoder = nn.Sequential(nn.Linear(16, 32),
                                      nn.Tanh(),
                                      nn.Linear(32, 64),
                                      nn.Tanh(),
                                      nn.Linear(64, 128),
                                      nn.Tanh(),
                                      nn.Linear(128, 256),
                                      nn.Tanh(),
                                      nn.Linear(256, 512),
                                      nn.Tanh(),
                                      nn.Linear(512, 28*28),
                                      )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
    
class FFN(nn.Module):
    
    """
    Do not initialise the weights at zeros
    """
    
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self._nn = nn.Sequential(nn.Linear(input_dim, 512),
                                 nn.Tanh(),
                                 #nn.Linear(1024, 512),
                                 #nn.Tanh(),
                                 nn.Linear(512, 256),
                                 nn.Tanh(),
                                 nn.Linear(256, 128),
                                 #nn.Tanh(),
                                 #nn.Linear(128, 64),
                                 #nn.Tanh(),
                                 #nn.Linear(64, 32),
                                 #nn.Tanh(),
                                 #nn.Linear(32, 16),
                                 nn.Tanh(),
                                 nn.Linear(128, output_dim),
                                 nn.Softmax(dim = 1))

    def forward(self, x):
        return self._nn(x)
        
class RNNet(nn.Module):
    
    def __init__(self, input_dim, hidden_size, layers, depth, output):
        super().__init__()
        self._hidden_size = hidden_size
        self._layers = layers
        self._input_dim = input_dim
        self._nn = nn.Sequential(nn.Linear(layers * hidden_size, depth[0]),
                                 nn.Tanh(),
                                 nn.Linear(depth[0], depth[1]),
                                 nn.Tanh(),
                                 nn.Linear(depth[1], output))
        self._rnn = nn.RNN(input_dim, hidden_size, layers, batch_first = True)
        
    def forward(self, x):
        with torch.backends.cudnn.flags(enabled=False):
            x = self._rnn(x)[1]
        x = x.movedim(0, 1).reshape(-1, self._layers * self._hidden_size)
        return self._nn(x)
    
class Wrapper:
    
    def __init__(self):
        self._funcs = {"0" : self.f, "1" : self.g, "2" : self.Hv,
                       "01" : self.fg, "02" : self.fHv, "12" : self.gHv,
                       "012": self.fgHv}
        
    def f(self, w):
        raise NotImplementedError
    
    def g(self, w):
        raise NotImplementedError
    
    def Hv(self, w):
        raise NotImplementedError
    
    def fg(self, w):
        raise NotImplementedError
    
    def fgHv(self, w):
        raise NotImplementedError
    
    def gHv(self, w):
        raise NotImplementedError
        
    def fHv(self, w):
        raise NotImplementedError

    def __call__(self, x, order):
        return self._funcs[order](x)
        
class funcWrapper(Wrapper):

    def __init__(self, func):
        super().__init__()
        self.func = func
    
    def _gradIt(self, w):
        if w.requires_grad:
            return w.detach().requires_grad_(True)
        else:
            return w.requires_grad_(True)
        
    def f(self, w):
        with torch.no_grad():
            return self.func(w)
            
    def g(self, w):
        w = self._gradIt(w)
        f = self.func(w)
        g = torch.autograd.grad(f, w)[0]
        f.detach()
        return g.detach()
    
    def fg(self, w):
        w = self._gradIt(w)
        f = self.func(w)
        g = torch.autograd.grad(f, w)[0]
        return f.detach(), g.detach()
    
    def fgHv(self, w):
        w = self._gradIt(w)
        f = self.func(w)
        g = torch.autograd.grad(f, w, create_graph = True)[0]
        Hv = lambda v : torch.autograd.grad((g,), w, v, create_graph = False, 
                                            retain_graph = True)[0].detach()
        return f.detach(), g.detach(), Hv

class nnWrapper(Wrapper):
    
    def __init__(self, func, loss):
        super().__init__()
        self.func, self.loss = func, loss
    
    def _toModule_toFunctional(self, w):
        if w.requires_grad:
            w = w.detach().requires_grad_(True)
        else:
            w = w.requires_grad_(True)
        
        nn.utils.vector_to_parameters(w, self.func.parameters())
        return make_functional(self.func, disable_autograd_tracking = False)
    
    def f(self, x, X, Y):
        if (X.shape[0] - 1) // cSPLIT >= 1:
            return self._accf(x, X, Y)
        
        device = x.device
        functional, w = self._toModule_toFunctional(x)
        with torch.no_grad():
            return self.loss(functional(w, X.to(device)), Y.to(device))
            
    def g(self, x, X, Y):
        if (X.shape[0] - 1) // cSPLIT >= 1:
            return self._accg(x, X, Y)
        
        device = x.device
        functional, w = self._toModule_toFunctional(x)
        val = self.loss(functional(w, X.to(device)), Y.to(device))
        g = torch.autograd.grad(val, w)
        g = nn.utils.parameters_to_vector(g)
        return g.detach()
    
    def fg(self, x, X, Y):
        if (X.shape[0] - 1) // cSPLIT >= 1:
            return self._accfg(x, X, Y)
        
        device = x.device
        functional, w = self._toModule_toFunctional(x)
        val = self.loss(functional(w, X.to(device)), Y.to(device))
        g = torch.autograd.grad(val, w)
        g = nn.utils.parameters_to_vector(g)
        return val.detach(), g.detach()
    
    def fgHv(self, x, X, Y):
        if (X.shape[0] - 1) // cSPLIT >= 1:
            return self._accfgHv(x, X, Y)
        
        device = x.device
        functional, x = self._toModule_toFunctional(x)
        val = self.loss(functional(x, X.to(device)), Y.to(device))
        g = torch.autograd.grad(val, x, create_graph = True)
        g = nn.utils.parameters_to_vector(g)
        Hv = lambda v : nn.utils.parameters_to_vector(
            torch.autograd.grad(g, x, grad_outputs = v, create_graph = False, retain_graph = True)
            ).detach()
        return val.detach(), g.detach(), Hv
    
    def Hv(self, x, X, Y):
        if (X.shape[0] - 1) // cSPLIT >= 1:
            return self._accHv(x, X, Y)
        
        device = x.device
        functional, x = self._toModule_toFunctional(x)
        val = self.loss(functional(x, X.to(device)), Y.to(device))
        g = torch.autograd.grad(val, x, create_graph = True)
        g = nn.utils.parameters_to_vector(g)
        return lambda v : nn.utils.parameters_to_vector(torch.autograd.grad(g, x, v, create_graph = False, retain_graph = True)
                                                        ).detach()
    
    def _HvSingle(self, v, x, X, Y):
        device = x.device
        functional, x = self._toModule_toFunctional(x)
        val = self.loss(functional(x, X.to(device)), Y.to(device))
        g = torch.autograd.grad(val, x, create_graph = True)
        g = nn.utils.parameters_to_vector(g)
        return nn.utils.parameters_to_vector(torch.autograd.grad(g, x, v)).detach()
    
    def _accf(self, x, X, Y):
        n = X.shape[0]
        acc_f = torch.tensor(0, dtype = cTYPE, device = cCUDA)
        for i in range(cSPLIT, n + 1, cSPLIT):
            acc_f += self.f(x, X[i - cSPLIT : i], Y[i - cSPLIT : i])
        
        acc_f *= (cSPLIT / n)
        if n % cSPLIT != 0:
            acc_f += self.f(x, X[i:], Y[i:]) * (n % cSPLIT / n)
        return acc_f
        
    def _accg(self, x, X, Y):
        n = X.shape[0]
        acc_g = torch.zeros_like(x, dtype = cTYPE, device = cCUDA)
        for i in range(cSPLIT, n + 1, cSPLIT):
            acc_g += self.g(x, X[i - cSPLIT : i], Y[i - cSPLIT : i])
              
        acc_g *= (cSPLIT / n)
        if n % cSPLIT != 0:
            g = self.g(x, X[i:], Y[i:])
            acc_g += g * (n % cSPLIT / n)
        return acc_g
           
    def _accfg(self, x, X, Y):
        n = X.shape[0]
        acc_f = torch.tensor(0, dtype = cTYPE, device = cCUDA)
        acc_g = torch.zeros_like(x, dtype = cTYPE, device = cCUDA)
        for i in range(cSPLIT, n + 1, cSPLIT):
            f, g = self.fg(x, X[i - cSPLIT : i], Y[i - cSPLIT : i])
            acc_f += f
            acc_g += g
            del f, g
        
        acc_f *= (cSPLIT / n)
        acc_g *= (cSPLIT / n)
        if n % cSPLIT != 0:
            f, g = self.fg(x, X[i:], Y[i:])
            acc_f += f * (n % cSPLIT / n)
            acc_g += g * (n % cSPLIT / n)
        return acc_f, acc_g
    
    def _accfgHv(self, x, X, Y):
        return *self._accfg(x, X, Y), lambda v : self._accHv(v, x, X, Y)
    
    def _accHv(self, v, x, X, Y):
        n = X.shape[0]
        Hvec = torch.zeros_like(x, dtype = cTYPE, device = cCUDA)
        for i in range(cSPLIT, n + 1, cSPLIT):
            Hvec += self._HvSingle(v, x, X[i - cSPLIT : i], Y[i - cSPLIT : i])

        Hvec *= (cSPLIT / n)
        if n % cSPLIT != 0:
            Hvec += self._HvSingle(v, x, X[i:], Y[i:]) * (n % cSPLIT / n)
        return Hvec
    
    def __call__(self, x, order, X, Y):
        return self._funcs[order](x, X, Y)
    
class ObjFunc:
    
    def __init__(self, func, loss, X, Y, reg, mini, Hsub):
        
        if X.shape[0] // cSPLIT >= 1:
            self.X, self.Y = X, Y
        else:
            self.X, self.Y = X.to(cCUDA), Y.to(cCUDA)
            
        self.fun = nnWrapper(func, loss)
        self.Hsub = Hsub
        self.mini = mini
        if reg is None:
            self.reg = none_reg
        else:
            self.reg = funcWrapper(reg)
            
    def minibatch(self, x, order):
        
        n = self.X.shape[0]
        m = math.ceil(n * self.mini)
        perm = torch.randperm(n)
        
        if order == "0" or order == "1":
            return self.fun(x, order, self.X[perm[:m]], self.Y[perm[:m]]) + self.reg(x, order)
        
        if order == "01":
            f, g = self.fun(x, "01", self.X[perm[:m]], self.Y[perm[:m]])
            f_reg, g_reg = self.reg(x, order)
            return f + f_reg, g + g_reg
            
    def subHessian(self, x, order):
        
        n = self.X.shape[0]
        m = math.ceil(n * self.Hsub)
        perm = torch.randperm(n)

        if order == "012":
            f1, g1, Hv = self.fun(x, order, self.X[perm[:m]], self.Y[perm[:m]])
            f2, g2 = self.fun(x, "01", self.X[perm[m:]], self.Y[perm[m:]])
            
            f_reg, g_reg, Hv_reg = self.reg(x, order)
            
            f = m * f1 / n + (n - m) * f2 / n
            g = m * g1 / n + (n - m) * g2 / n
            return f + f_reg, g + g_reg, lambda v : Hv(v) + Hv_reg(v)
        
        if order == "2":
            return lambda v : self.fun.Hv(x, self.X[perm[:m]], self.Y[perm[:m]])(v) + self.reg(x, order)(v)
        
        if order == "02" or order == "12":
            raise NotImplementedError
        
            
    def __call__(self, x, order):
    
        if order == "f":
            return self.fun(x, "0", self.X, self.Y) + self.reg(x, "0")
            
        if "2" in order and self.Hsub != 1:
            return self.subHessian(x, order)
        
        if self.mini != 1:
            return self.minibatch(x, order)
        
        if order == "0" or order == "1":
            return self.fun(x, order, self.X, self.Y) + self.reg(x, order)
        
        if order == "2":
            return lambda w : self.fun(x, order, self.X, self.Y)(w) + self.reg(x, order)(w)
        
        if order == "01":
            f, g = self.fun(x, order, self.X, self.Y)
            reg_f, reg_g = self.reg(x, order)
            return f + reg_f, g + reg_g
        
        if order == "012":
            f, g, Hv = self.fun(x, order, self.X, self.Y)
            reg_f, reg_g, reg_Hv = self.reg(x, order)
            return f + reg_f, g + reg_g, lambda x : Hv(x) + reg_Hv(x)
        
        if order == "12" or order == "02":
            gof, Hv = self.fun(x, order, self.X, self.Y)
            reg_gof, reg_Hv = self.reg(x, order)
            return gof + reg_gof, lambda x : Hv(x) + reg_Hv(x)
    
if __name__ == "__main__":
    from derivativeTest import derivativeTest
    from funcs import logisticFun, logisticModel
    #from regularizers import non_convex
    
    trainX = torch.rand((2000, 28 * 28)).to(cTYPE)
    trainY = torch.rand((2000, 10)).to(cTYPE)
    ffn = FFN(28 * 28, 10).to(cTYPE)
    
    loss = nn.MSELoss()
    s = nn.utils.parameters_to_vector(ffn.parameters()).detach()
    #fun = ObjFunc(ffn, loss, trainX, trainY, None, 1)
    
    # Test to see if the accumulative derivatives are the same as 
    # non-accumulative derivatives and if the derivatives are correct
    # cSPLIT = 20000
    # f1, g1, Hv1 = derivativeTest(lambda x : fun(x, "012"), s)
    # cSPLIT = 27
    # f2, g2, Hv2 = derivativeTest(lambda x : fun(x, "012"), s)
    # assert torch.all(torch.isclose(f1, f2))
    # assert torch.all(torch.isclose(g1, g2))
    # assert torch.all(torch.isclose(Hv1, Hv2))
    
    # Check others 
    # for i in range(100):
    #     x = torch.rand(s.shape[0], dtype = cTYPE)
    #     cSPLIT = 10000
    #     f1, g1 = fun(x, "01")
    #     cSPLIT = 479
    #     f2, g2 = fun(x, "01")
    #     assert torch.isclose(torch.mean(g1), torch.mean(g2))
    #     assert torch.isclose(f1, f2)
    
    # print("Passed!")
    
    #tests.subHessian_test(lambda x, y, v : ObjFunc(ffn, loss, x, y, None, 1, v), s.shape[0], trainX, trainY)
    #print("Passed!")
    #tests.minibatch_test(lambda x, y, v : ObjFunc(ffn, loss, x, y, None, v, 1), s.shape[0], trainX, trainY)
    #print("Passed!")
