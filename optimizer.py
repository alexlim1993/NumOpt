# -*- coding: utf-8 -*-
"""
Created on Thu Nov 24 15:59:37 2022

@author: uqalim8
"""
import time

GD_STATS = {"ite":"g", "orcs":"g", "time":".2f", "f":".2e", 
            "g_norm":".2e", "alpha":".2e"}

NEWTON_STATS = {"ite":"g", "inite":"g", "orcs":"g", "time":".2f", 
                "f":".2e", "g_norm":".2e", "alpha":".2e"}

NEWTON_NC_STATS = {"ite":"g", "inite":"g", "dtype":"", "orcs":"g",
                   "time":".2f", "f":".2e", "g_norm":".2e", "alpha":".2e"}

class Optimizer:
    
    def __init__(self, fun, x0, alpha0, gradtol, maxite, maxorcs):
        self.fun = fun
        self.xk = x0
        self.alpha0 = alpha0
        self.maxorcs = maxorcs
        self.k, self.orcs, self.toc, self.lineite = 0, 0, 0, 0
        self.gknorm, self.record = None, None
        self.maxite = maxite
        self.gradtol = gradtol
        self.record = dict(((i, []) for i in self.info.keys()))
        
    def recording(self, stats):
        for n, i in enumerate(self.record.keys()):
            self.record[i].append(stats[n])
            
    def printStats(self):
        if not (self.k - 1) % 10:
            print(".." + 5 * len(self.info) * ".." + "..")
            form = ["{:^10}"] * len(self.info)
            print("|".join(form).format(*self.info.keys()))
            print(".." + 5 * len(self.info) * ".." + "..")
        form = ["{:^10" + i + "}" for i in self.info.values()]
        print("|".join(form).format(*(self.record[i][-1] for i in self.info.keys())))        
    
    def progress(self, verbose):
        self.k += 1
        self.oracleCalls()
        self.recordStats()
        if verbose:
            self.printStats()

    def termination(self):
        return self.k >= self.maxite or self.gknorm <= self.gradtol or self.orcs >= self.maxorcs
    
    def optimize(self, verbose):
        self.recordStats()
        while not self.termination():
            tic = time.time()
            self.step()
            self.toc += time.time() - tic
            self.progress(verbose)

            
    def recordStats(self):
        raise NotImplementedError
        
    def step(self):
        raise NotImplementedError
    
    def oracleCalls(self):
        raise NotImplementedError