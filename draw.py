# -*- coding: utf-8 -*-
"""
Created on Thu Feb 16 09:56:23 2023

@author: uqalim8
"""

import matplotlib.pyplot as plt
import matplotlib
import os, json, torch
import seaborn as sns

FOLDER_PATH = "./RNN/CG/"
SPLIT = ".json"

def keys(x):
    # n = x.split("%")[0]
    # if x == "Single Sample.json":
    #     return 0
    # else:
    #    return float(n)
    return float(x.split("reg-")[1][:-5])
    
def drawPlots(records, stats, folder_path):
    STATS = {"ite":"Iterations", "inite":"Inner Iteration", "orcs":"Oracle Calls",
             "time":"Time(second)", "f":"Function Value", "g_norm":"Norm of Gradient",
             "alpha":"Step Size", "acc_train":"Training Error", "acc_val":"Validation Error"}
    COLORS = ["tab:olive", "tab:gray", "tab:cyan", "tab:pink", "r", "g", "b", "k"]
    LINESTYLE = ["--", "-", "-.", ":"]
    
    for x, y in stats:
        n = 0
        font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 12}

        matplotlib.rc('font', **font)
        plt.figure(figsize = (10,6))
        for name, record in records:
            if "zL-BFGS" == name:
                name = "L-BFGS"
            #record[y][0] = 10.487999767065048 #cifar training
            #record[y][0] = 10.740000009536743 #cifar validation
            #record[y] = torch.tensor(record[y])
            record[y] = torch.tensor(record[y]) - torch.tensor([*record[y][1:], record[y][-1]])
            plt.semilogx(torch.tensor(record[x]) + 1, record[y], color = COLORS[n], 
                       linestyle = LINESTYLE[n // 4 % 4], label = name)
            n += 1
        #plt.title("5% Sub-sampled Hessian")
        plt.xlabel(STATS[x], fontsize = 20)
        plt.ylabel(STATS[y], fontsize = 20)
        plt.legend()
        plt.savefig(folder_path + f"{x}_{y}.png")
        plt.close()
        
def kde_density(records, folder_path):
    plt.figure(figsize = (10,6))
    for name, record in records:
        sns.kdeplot(torch.tensor(record["relHr"][1:]), log_scale = True, fill = True, common_norm=False,
                    alpha=.25, linewidth=.5, label = name)
    plt.xlabel("Relative Normal Residual", fontsize = 12)
    plt.legend()
    plt.savefig(FOLDER_PATH + f"{name}_relHr.png")
    plt.close()
    
    plt.figure(figsize = (10,6))
    for name, record in records:
        sns.kdeplot(torch.tensor(record["relr"][1:]), log_scale = True, fill = True, common_norm=False,
                    alpha=.25, linewidth=.5, label = name)
    plt.xlabel("Relative Residual", fontsize = 12)
    plt.legend()
    plt.savefig(FOLDER_PATH + f"{name}_relr.png")
    plt.close()
    
    
if __name__ == "__main__":
    files = os.listdir(FOLDER_PATH)
    files = list(filter(lambda x : ".json" in x, files))
    #files.sort(key = keys)
    records = []
    for i in files:
        name, _ = i.split(SPLIT)
        with open(FOLDER_PATH + i, "r") as f:
            records.append((name, json.load(f)))
    #("orcs", "f"), ("ite", "f"), ("orcs", "acc_train"), ("ite", "acc_train"), ("orcs", "acc_val"), ("ite", "acc_val")
    drawPlots(records, (("orcs", "acc_train"), ("ite", "acc_train")), FOLDER_PATH)
    #kde_density(records, FOLDER_PATH)   