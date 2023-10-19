# -*- coding: utf-8 -*-
"""
Created on Thu Feb 16 09:56:23 2023

@author: uqalim8
"""

import matplotlib.pyplot as plt
import os, json, torch, optAlgs

FOLDER_PATH = "./try_again/Ethylene/MR_CG/"
FILE = "NewtonMR.json"
STAT = optAlgs.NEWTON_NC_STATS
DRAW, WRITE = True, False

def drawPlots(records, stats, name):
    
    lines = ["-.", "-", ":", "--"]
    color = ["b", "g", "r", "k"]#, "y", "c"]
    markers = ["^", "o", "1", "*", "x", "d"]
    labels = {"f" : "$f(x)$", "g_norm" : "$\|\|g\|\|$", 
              "orcs" : "Oracle Calls", "ite" : "Iterations", 
              "time" : "Time", "acc" : "Accuracies"}
     
    for x, y in stats:
        plt.figure(figsize=(10,6))
        c = 0
        for j, i in records:
            truth = torch.tensor(i["g_norm"]) > 1e-8
            if j[0] == "Z":
                j = j[1:]
            plt.loglog(torch.tensor(i[x])[truth] + 1, torch.tensor(i[y])[truth],
                       linestyle = lines[(c // len(color)) % len(lines)],
                       #marker = markers[c % len(lines)],
                       color = color[c % len(color)],
                       label = j)
            c += 1
        plt.xlabel(labels.get(x, x), fontsize=24)
        plt.ylabel(labels.get(y, y), fontsize=24)
        plt.legend()
        plt.savefig(name + x + "_" + y)
        plt.close()

def writeRecords(folder_path, file, stat):
    labels = {"f" : "f(x)", "g_norm" : "||g||", "alpha" : "alpha",
              "orcs" : "Oracle Calls", "ite" : "Iterations", "inite" : "Inner Ite-s",
              "time" : "Time", "acc" : "Accuracies", "dtype" : "D-Type", "delta" : "Delta",
              "iteLS": "LiSe Ite"}
    
    with open(folder_path + file, "r") as f:
        record = json.load(f)
        
    name, _ = file.split(".json")
    text = 8 * len(record.keys()) * ".." + "\n"
    form = ["{:^15}"] * len(record.keys())
    text += "|".join(form).format(*[labels[i] for i in record.keys()]) + "\n"
    text += 8 * len(record.keys()) * ".." + "\n"
    
    form = ["{:^15" + i + "}" for i in stat.values()]
    with open(folder_path + name + ".txt", "w", newline = "") as f:
        f.write(text)
        count = 1
        for j in zip(*record.values()):
            f.write("|".join(form).format(*j) + "\n")
            if not count % 50:
                f.write(text)
            count += 1          
        
def keys(x):
    n = x.split("%")[0]
    if x == "Single Sample.json":
        return 0
    else:
        return float(n)

if __name__ == "__main__":
    if DRAW:
        files = os.listdir(FOLDER_PATH)
        files = filter(lambda x : ".json" in x, files)
        #files.sort(key = keys)
        records = []
        for i in files:
            name, _ = i.split(".json")
            with open(FOLDER_PATH + i, "r") as f:
                records.append((name, json.load(f)))
                
        drawPlots(records, (("orcs", "f"), ("ite", "f")), FOLDER_PATH)
        
    if WRITE:
        writeRecords(FOLDER_PATH, FILE, STAT)