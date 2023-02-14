import os, torch, optAlgs, regularizers, funcs, json
import neuralNetwork as nn
import matplotlib.pyplot as plt

TEXT = "{:<20} : {:>20}"

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
    folder_path += f"{alg}_hsub-{hsub}.json"
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
        print(TEXT.format("x0", "initialised"))
        return x0_type.double()
    
    if x0_type == "ones":
        print(TEXT.format("x0", x0_type))
        return torch.ones(size, dtype = torch.float64)
    
    if x0_type == "zeros":
        print(TEXT.format("x0", x0_type))
        return torch.zeros(size, dtype = torch.float64)
    
    if x0_type == "normal":
        print(TEXT.format("x0", x0_type))
        return torch.randn(size, dtype = torch.float64)
    
    if x0_type == "uniform":
        print(TEXT.format("x0", x0_type))
        return torch.rand(size, dtype = torch.float64)
    
def initAlg(func, x0, algo, c):
    
    
    if algo == "NewtonCG":
        print(TEXT.format("Algorithm", algo))
        return optAlgs.NewtonCG(func, x0, c.alpha0, c.gradtol, c.maxite, 
                                c.maxorcs, c.restol, c.inmaxite)
        
    if algo == "NewtonMR_NC":
        print(TEXT.format("Algorithm", algo))
        return optAlgs.NewtonMR_NC(func, x0, c.alpha0, c.gradtol, c.maxite, 
                                   c.maxorcs, c.restol, c.inmaxite, c.lineMaxite, 
                                   c.lineBetaB, c.lineRho, c.lineBetaFB, c.Hsub)
    
    if algo == "NewtonCG_NC":
        print(TEXT.format("Algorithm", algo))
        return optAlgs.NewtonCG_NC(func, x0, c.alpha0, c.gradtol, c.maxite, c.maxorcs, 
                                   c.restol, c.inmaxite, c.lineMaxite, c.lineBeta, 
                                   c.lineRho, c.epsilon, c.Hsub)
    
    if algo == "Linesearch_GD":
        print(TEXT.format("Algorithm", algo))
        return optAlgs.linesearchGD(func, x0, c.alpha0, c.gradtol, c.maxite, 
                                    c.maxorcs, c.lineMaxite, c.lineBetaB, c.lineRho)
    
    if algo == "NewtonCG_NC_FW":
        print(TEXT.format("Algorithm", algo))
        return optAlgs.NewtonCG_NC_FW(func, x0, c.alpha0, c.gradtol, c.maxite, c.maxorcs, 
                                   c.restol, c.inmaxite, c.lineMaxite, c.lineBeta, 
                                   c.lineRho, c.epsilon, c.Hsub)

def initReg(reg, lamb):
    
    if reg == "None":
        print(TEXT.format("Regulariser", f"{reg} , {lamb}"))
        return 
    
    if reg == "Non-convex":
        print(TEXT.format("Regulariser", f"{reg} , {lamb}"))
        return lambda x : regularizers.non_convex(x, lamb)
    
    if reg == "2-norm":
        print(TEXT.format("Regulariser", f"{reg} , {lamb}"))
        return lambda x : regularizers.two_norm(x, lamb)
        
def initFunc_x0(func, x0, trainX, trainY, Hsub, reg):
    
    print(TEXT.format("Hsub", Hsub))
     
    if func == "nls":
        print(TEXT.format("Objective Function", func))
        x0 = initx0(x0, trainX.shape[-1])
        
        def pred(w):
            Y = torch.round(funcs.logisticModel(trainX, w))
            return torch.sum(Y == trainY) / len(Y)
    
        return x0, pred, lambda w, v : funcs.funcWrapper(funcs.nls, trainX, trainY, w, Hsub, reg, v)
    
    if "ffnn" in func:
        
        if "MSE" in func:
            print(TEXT.format("Objective Function", func))
            loss = torch.nn.MSELoss()
            
        elif "CELoss" in func:
            print(TEXT.format("Objective Function", func))
            loss = torch.nn.CrossEntropyLoss()
        
        dim, cat = trainX.shape[-1], trainY.shape[-1]
        ffn = nn.FFN(dim, cat)
        
        if x0 == "torch":
            print(TEXT.format("x0", x0))
            x0 = torch.nn.utils.parameters_to_vector(ffn.parameters()).double()
        else:
            x0 = initx0(x0, torch.nn.utils.parameters_to_vector(ffn.parameters()).shape)
        
        def pred(w):
            with torch.no_grad():
                torch.nn.utils.vector_to_parameters(w, ffn.parameters())
                Y = torch.argmax(ffn(trainX), dim = -1)
                return float(torch.sum(Y == torch.argmax(trainY, dim = -1)) / len(Y)) * 100
        
        return x0, pred, lambda w, v : nn.nnWrapper(ffn, loss, trainX, trainY, 
                                                    w, Hsub, reg, v)
    
    if "AE_MNIST" in func:
        #dim, cat = trainX.shape[-1], trainY.shape[-1]
        fnn = nn.auto_Encoder_MNIST()
        x0 = initx0(x0, torch.nn.utils.parameters_to_vector(fnn.parameters()).shape)
        loss = torch.nn.MSELoss()
        return x0, lambda w, v : nn.nnWrapper(fnn, loss, trainX, trainX, 
                                              w, Hsub, reg, v)
    
def drawPlots(records, stats, name):
    
    lines = ["-", "-.", ":", "--"]
    color = ["b", "g", "r", "k"]
    markers = ["^", "o", "1", "*", "x", "d"]
    labels = {"f" : "$f(x_k)$", "g_norm" : "$||g_k||$", 
              "orcs" : "Oracle Calls", "ite" : "Iterations", "time" : "Time"}
    
    for x, y in stats:
        plt.figure(figsize=(10,6))
        c = 0
        for j, i in records:
            plt.loglog(torch.tensor(i[x]) + 1, i[y],
                       linestyle = lines[c % len(lines)],
                       marker = markers[c % len(lines)],
                       color = color[c % len(color)],
                       label = j)
            c += 1
        plt.xlabel(labels.get(x, x), fontsize=24)
        plt.ylabel(labels.get(y, y), fontsize=24)
        plt.legend()
        plt.savefig(name + x + "_" + y)
        plt.close()