# -*- coding: utf-8 -*-
"""
Created on Mon Jan  9 12:33:43 2023

@author: uqalim8
"""
import torch.nn as nn
import torch, math, GAN

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
        self._nn = nn.Sequential(nn.Linear(input_dim, 128),
                                 nn.Sigmoid(),
                                 nn.Linear(128, 64),
                                 nn.Sigmoid(),
                                 nn.Linear(64, 32),
                                 nn.Sigmoid(),
                                 nn.Linear(32, output_dim),
                                 nn.Softmax(dim = 1))

    def forward(self, x):
        return self._nn(x)
    
class RNNet(nn.Module):
    
    def __init__(self, input_dim, hidden_size, layers, output):
        super().__init__()
        self._hidden_size = hidden_size
        self._layers = layers
        self._input_dim = input_dim
        self._nn = nn.Linear(layers * hidden_size, output)
        self._rnn = nn.RNN(input_dim, hidden_size, layers, batch_first = True)
        
    def forward(self, x):
        x = self._rnn(x)[1].movedim(0, 1).reshape(-1, self._layers * self._hidden_size)
        return self._nn(x)
    
def nnWrapper(func, loss, trainX, trainY, w, Hsub, reg, order):
    
    nn.utils.vector_to_parameters(w, func.parameters())
    
    if Hsub == 1 or not "2" in order:
        if reg is None:
            return nnfgHv(func, loss, trainX, trainY, order)
        
        if order == "0":
            with torch.no_grad():
                reg_f = fgHv(reg, w, order)
                f = nnfgHv(func, loss, trainX, trainY, order)
            return f + reg_f
        
        if order == "01":
            reg_f, reg_g = fgHv(reg, w, order)
            f, g = nnfgHv(func, loss, trainX, trainY, order)
            return f + reg_f, g + reg_g
        
        if order == "012":
            reg_f, reg_g , reg_Hv = fgHv(reg, w, order)
            f, g, Hv = nnfgHv(func, loss, trainX, trainY, order)
            return f + reg_f, g + reg_g, lambda v : Hv(v) + reg_Hv(v)
        
        if order == "2":
            reg_Hv = fgHv(reg, w, order)
            Hv = nnfgHv(func, loss, trainX, trainY, order)
            return lambda v : Hv(v) + reg_Hv(v)
            
    n = trainX.shape[0]
    m = math.ceil(n * Hsub)
    perm = torch.randperm(n)
    reg_f, reg_g, reg_H = 0, 0, lambda v : 0
    
    if order == "2":
        if not reg is None:
            reg_H = fgHv(reg, w, order)
            
        H = nnfgHv(func, loss, trainX[perm][:m], trainY[perm][:m], order)
        return lambda v : H(v) + reg_H(v)

    if order == "012":
        if not reg is None:
            reg_f, reg_g, reg_H = fgHv(reg, w, order)
            
        f1, g1, H = nnfgHv(func, loss, trainX[perm][:m], trainY[perm][:m], order)
        f2, g2 = nnfgHv(func, loss, trainX[perm][m:], trainY[perm][m:], "01")
        
        # do we always need to take mean?
        # will there be a function that it doesn't take mean?
        # should mean be taken here or within the function itself?
        f = m * f1 / n + (n - m) * f2 / n + reg_f
        g = m * g1 / n + (n - m) * g2 / n + reg_g
        return f, g, lambda v : H(v) + reg_H(v)

def nnfgHv(func, loss, trainX, trainY, order):
    
    f = func(trainX)
    val = loss(f, trainY)
    func.zero_grad()

    if order == "0":
        return val.detach()
    
    if order == "01":
        g = torch.autograd.grad(val, func.parameters(), create_graph = False, retain_graph = True)
        return val.detach(), nn.utils.parameters_to_vector(g).detach()
    
    if order == "012":
        g = torch.autograd.grad(val, func.parameters(), create_graph = True, retain_graph = True)
        g = nn.utils.parameters_to_vector(g)
        Hv = lambda v : nn.utils.parameters_to_vector(
            torch.autograd.grad(g, func.parameters(), grad_outputs = v, create_graph = False, retain_graph = True)
            ).detach()
        return val.detach(), g.detach(), Hv
    
    if "2" == order:
        g = torch.autograd.grad(val, func.parameters(), create_graph = True, retain_graph = True)
        g = nn.utils.parameters_to_vector(g)
        Hv = lambda v : nn.utils.parameters_to_vector(
            torch.autograd.grad(g, func.parameters(), v, create_graph = False, retain_graph = True)
            ).detach()
        return Hv
    
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
    
if __name__ == "__main__":
    from derivativeTest import derivativeTest

    trainX = torch.rand((1000, 28, 28)).double()
    trainY = torch.rand((1000, 10)).double()
    ffn = RNNet(28, 20, 10).double()
    
    loss = nn.MSELoss()
    fun = lambda w : nnWrapper(ffn, loss, trainX, trainY, w, 1, None, "012")
    derivativeTest(fun, nn.utils.parameters_to_vector(ffn.parameters()).detach())
    

    