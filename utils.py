import os, torch, optAlgs, regularizers, funcs, json
import matplotlib.pyplot as plt

class const():
    pass

def makeFolder(folder_path):
    if not os.path.isdir(folder_path):
        os.makedirs(folder_path)

def saveRecords(folder_path, dataset, alg, func, hsub, file):
    if folder_path[-1] != "/":
        folder_path += "/"
    folder_path += f"{dataset}_{func}/"
    makeFolder(folder_path)
    folder_path += f"{alg} hsub-{hsub}.json"
    with open(folder_path, "w") as f:
        json.dump(file, f)
        
def openRecords(folder_path, dataset, func):
    if folder_path[-1] != "/":
        folder_path += "/"
    files = os.listdir(f"{folder_path}{dataset}_{func}/")
    records = []
    for i in files:
        with open(folder_path + i, "r") as f:
            records.append((i, json.load(f)))
    return records
            
def initx0(x0_type, size):
    
    if not type(x0_type) == str:
        return x0_type
    
    if x0_type == "ones":
        return torch.ones(size, dtype = torch.float64)
    
    if x0_type == "zeros":
        return torch.zeros(size, dtype = torch.float64)
    
    if x0_type == "randn":
        return torch.randn(size, dtype = torch.float64)
    
    if x0_type == "rand":
        return torch.rand(size, dtype = torch.float64)
    
def initAlg(func, x0, alg, c):
    
    if alg == "NewtonCG":
        return optAlgs.NewtonCG(func, x0, c.alpha0, c.gradtol, c.maxite, 
                                c.maxorcs, c.restol, c.inmaxite)
        
    if alg == "NewtonMR_NC":
        return optAlgs.NewtonMR_NC(func, x0, c.alpha0, c.gradtol, c.maxite, 
                                   c.maxorcs, c.restol, c.inmaxite, c.lineMaxite, 
                                   c.lineBetaB, c.lineRho, c.lineBetaFB, c.Hsub)
    
    if alg == "NewtonCG_NC":
        return optAlgs.NewtonCG_NC(func, x0, c.alpha0, c.gradtol, c.maxite, c.maxorcs, 
                                   c.restol, c.inmaxite, c.lineMaxite, c.lineBeta, 
                                   c.lineRho, c.epsilon, c.Hsub)

def initReg(reg, lamb):
    
    if reg == "None":
        return 
    
    if reg == "Non-convex":
        return lambda x : regularizers.non_convex(x, lamb)
    
    if reg == "2-norm":
        return lambda x : regularizers.two_norm(x, lamb)
        
def initFunc(func, trainX, trainY, Hsub, reg):
    
    if func == "nls":
        return lambda x, v : funcs.funcWrapper(funcs.nls, trainX, trainY, x, Hsub, reg, v)
    
def drawPlots(records, stats, name):
    
    lines = ["-", "-.", ":", "--"]
    color = ["b", "g", "r", "k"]
    markers = ["^", "o", "1", "*", "x", "d"]
    labels = {"f" : "$f(x_k)$", "g_norm" : "$||g_k||$", 
              "orcs" : "Oracle Calls", "ite" : "Iterations", "time" : "Time"}
    
    for x, y in stats:
        plt.figure(figsize = (10, 10))
        c = 0
        for j, i in records:
            plt.loglog(torch.tensor(i[x]) + 1, i[y],
                       linestyle = lines[c % len(lines)],
                       marker = markers[c % len(lines)],
                       color = color[c % len(color)],
                       label = j)
            c += 1
        plt.xlabel(labels.get(x, x))
        plt.ylabel(labels.get(y, y))
        plt.legend()
        plt.savefig(name + x + "_" + y)
        plt.close()
    
    
    
    
    
    
    
    
    