import os, torch, optAlgs, regularizers, funcs, json, datasets, GAN
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
                
def initx0(x0_type, size, device):
    
    if not type(x0_type) == str:
        print(TEXT.format("x0", "initialised"))
        return x0_type.to(device)
    
    if x0_type == "ones":
        print(TEXT.format("x0", x0_type))
        return torch.ones(size, dtype = torch.float64, device = device)
    
    if x0_type == "zeros":
        print(TEXT.format("x0", x0_type))
        return torch.zeros(size, dtype = torch.float64, device = device)
    
    if x0_type == "normal":
        print(TEXT.format("x0", x0_type))
        return torch.randn(size, dtype = torch.float64, device = device)
    
    if x0_type == "uniform":
        print(TEXT.format("x0", x0_type))
        return torch.rand(size, dtype = torch.float64, device = device)
    
def initAlg(fun, x0, algo, c):
    
    
    if algo == "NewtonCG":
        print(TEXT.format("Algorithm", algo))
        return optAlgs.NewtonCG(fun, x0, c.alpha0, c.gradtol, c.maxite, 
                                c.maxorcs, c.restol, c.inmaxite)
        
    if algo == "NewtonMR_NC":
        print(TEXT.format("Algorithm", algo))
        return optAlgs.NewtonMR_NC(fun, x0, c.alpha0, c.gradtol, c.maxite, 
                                   c.maxorcs, c.restol, c.inmaxite, c.lineMaxite, 
                                   c.lineBetaB, c.lineRho, c.lineBetaFB, c.Hsub)
    
    if algo == "NewtonCG_NC":
        print(TEXT.format("Algorithm", algo))
        return optAlgs.NewtonCG_NC(fun, x0, c.alpha0, c.gradtol, c.maxite, c.maxorcs, 
                                   c.restol, c.inmaxite, c.lineMaxite, c.lineBeta, 
                                   c.lineRho, c.epsilon, c.Hsub)
    
    if algo == "Linesearch_GD":
        print(TEXT.format("Algorithm", algo))
        return optAlgs.linesearchGD(fun, x0, c.alpha0, c.gradtol, c.maxite, 
                                    c.maxorcs, c.lineMaxite, c.lineBetaB, c.lineRho)
    
    if algo == "NewtonCG_NC_FW":
        print(TEXT.format("Algorithm", algo))
        return optAlgs.NewtonCG_NC_FW(fun, x0, c.alpha0, c.gradtol, c.maxite, c.maxorcs, 
                                   c.restol, c.inmaxite, c.lineMaxite, c.lineBeta, 
                                   c.lineRho, c.epsilon, c.Hsub)
    
    if algo == "NewtonCG_TR_Steihaug":
        print(TEXT.format("Algorithm", algo))
        return optAlgs.NewtonCG_TR_Steihaug(fun, x0, None, c.gradtol, c.maxite, c.maxorcs, 
                                            c.restol, c.inmaxite, c.deltaMax, c.delta0, 
                                            c.eta, c.eta1, c.eta2, c.gamma1, c.gamma2, c.Hsub)

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
        
def initFunc_x0(func, x0, trainX, trainY, Hsub, reg, device):
    
    print(TEXT.format("Hsub", Hsub))
    
    if "MSE" in func:
        loss = torch.nn.MSELoss()
        
    elif "CELoss" in func:
        loss = torch.nn.CrossEntropyLoss()
        
    if func == "logloss":
        print(TEXT.format("Objective Function", func))
        x0 = initx0(x0, trainX.shape[-1], device)
        
        if not max(trainY) == 1 or not min(trainY) == 0:
            raise Exception("Only 0-and-1 binary Classification")
        
        def pred(w):
            Y = torch.round(funcs.logisticModel(trainX, w))
            return float(torch.sum(Y == trainY) / len(Y))
            
        return x0, pred, lambda w, v : funcs.funcWrapper(funcs.logloss, trainX, trainY, w, Hsub, reg, v)

    if func == "nls":
        print(TEXT.format("Objective Function", func))
        x0 = initx0(x0, trainX.shape[-1], device)
        
        if not max(trainY) == 1 or not min(trainY) == 0:
            raise Exception("Only 0-and-1 binary Classification")
            
        def pred(w):
            Y = torch.round(funcs.logisticModel(trainX, w))
            return float(torch.sum(Y == trainY) / len(Y))
    
        return x0, pred, lambda w, v : funcs.funcWrapper(funcs.nls, trainX, trainY, w, Hsub, reg, v)
    
    if "ffnn" in func:
        print(TEXT.format("Objective Function", func))
        
        dim, cat = trainX.shape[-1], trainY.shape[-1]
        ffn = nn.FFN(dim, cat)
        ffn.to(device)
        
        if x0 == "torch":
            print(TEXT.format("x0", x0))
            x0 = torch.nn.utils.parameters_to_vector(ffn.parameters()).double().to(device)
        else:
            x0 = initx0(x0, None, device)
        
        print(TEXT.format("dimension", x0.shape[0]))
        
        def pred(w):
            with torch.no_grad():
                torch.nn.utils.vector_to_parameters(w, ffn.parameters())
                Y = torch.argmax(ffn(trainX), dim = -1)
                return float(torch.sum(Y == torch.argmax(trainY, dim = -1)) / len(Y)) * 100
        
        return x0, pred, lambda w, v : nn.nnWrapper(ffn, loss, trainX, trainY, 
                                                    w, Hsub, reg, v)
    
    if "rnn" in func:
        n, seq, dim = trainX.shape
        rnn = nn.RNNet(dim, 16, 8, dim)
        rnn.to(device)
        
        if x0 == "torch":
            print(TEXT.format("x0", x0))
            x0 = torch.nn.utils.parameters_to_vector(rnn.parameters()).double().to(device)
        else:
            x0 = initx0(x0, None, device)
        
        print(TEXT.format("dimension", x0.shape[0]))

        return x0, lambda x : 0, lambda w, v : nn.nnWrapper(rnn, loss, trainX, trainY, 
                                                    w, Hsub, reg, v)
    
    if "AE_MNIST" in func:
        #dim, cat = trainX.shape[-1], trainY.shape[-1]
        fnn = nn.auto_Encoder_MNIST()
        x0 = initx0(x0, torch.nn.utils.parameters_to_vector(fnn.parameters()).shape)
        loss = torch.nn.MSELoss()
        return x0, lambda w : None, lambda w, v : nn.nnWrapper(fnn, loss, trainX, trainX, 
                                              w, Hsub, reg, v)
    
def execute(folder_path, dataset, algo, func, x0, Hsub, reg, lamb, const, verbose, device):
    makeFolder(folder_path)
    trainX, trainY, testX, testY = datasets.prepareData(folder_path, func, dataset, device)
    reg = initReg(reg, lamb)
    x0, pred, func = initFunc_x0(func, x0, trainX, trainY, Hsub, reg, device)
    algo = initAlg(func, x0.clone(), algo, const)
    algo.optimize(verbose, pred)
    return algo, x0

def executeGAN(folder_path, dataset, algG, algD, Hsub, reg, lamb, constG, constD, device):
    makeFolder(folder_path)
    trainX, _, _, _ = datasets.prepareData(folder_path, "nn", dataset, device)
    reg = initReg(reg, lamb)
    
    d = trainX.shape[1]
    gen = GAN.Generator(d, d)
    gen.to(device)
    
    dis = GAN.Discriminator(d)
    dis.to(device)
    
    wG = torch.nn.utils.parameters_to_vector(gen.parameters()).double().to(device)
    torch.nn.utils.vector_to_parameters(wG, gen.parameters())
    wD = torch.nn.utils.parameters_to_vector(dis.parameters()).double().to(device)
    torch.nn.utils.vector_to_parameters(wD, dis.parameters())
    
    algG = initAlg(None, wG.clone(), algG, constG)
    algD = initAlg(None, wD.clone(), algD, constD)
    
    GAN.trainGAN(gen, dis, wG, wD, trainX, Hsub, reg, algG, algD, device)
    

def drawPlots(records, stats, name):
    
    lines = ["-", "-.", ":", "--"]
    color = ["b", "g", "r", "k", "y", "c"]
    markers = ["^", "o", "1", "*", "x", "d"]
    labels = {"f" : "$f(x)$", "g_norm" : "$\|\|g\|\|$", 
              "orcs" : "Oracle Calls", "ite" : "Iterations", 
              "time" : "Time", "acc" : "Accuracies"}
    
    for x, y in stats:
        plt.figure(figsize=(10,6))
        c = 0
        for j, i in records:
            plt.loglog(torch.tensor(i[x]) + 1, i[y],
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
        
