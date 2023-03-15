# -*- coding: utf-8 -*-
"""
Created on Thu Feb 16 09:56:23 2023

@author: uqalim8
"""

import matplotlib.pyplot as plt
import utils, os, json

FOLDER_PATH = "./test/DelhiClimate_rnnMSE/"
SPLIT = ".json"

files = os.listdir(FOLDER_PATH)
records = []
for i in files:
    name, _ = i.split(SPLIT)
    with open(FOLDER_PATH + i, "r") as f:
        records.append((name, json.load(f)))
        
utils.drawPlots(records, (("orcs", "f"), ("ite", "f"), ("orcs", "g_norm"), ("ite", "g_norm")), FOLDER_PATH)
        