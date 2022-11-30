import torch

def linesearch_NewtonMR(obj, fk, gTp, x, p, maxiter=200, c1=1e-4):
    """    
    INPUTS:
        obj: a function handle of f (+ gradient, Hessian-vector-product)
        g: starting gradient gk
        x: starting x
        p: direction p
        maxiter: maximum iteration of Armijo line-search
        alpha: initial step-size
        c1: parameter of Armijo line-search
        c2: parameter of strong Wolfe condition
    OUTPUTS:
        x: results
        alpha: proper step-size
        T: iterations
    """
    T = 0
#    alpha = 2*(1 - 2*c1)
    alpha = 1
#    if alpha == 1:
    zeta = 2
#    else:
#        zeta = 2*(1 - 2*c1)
    # fa = _fa(obj, x, alpha, p) 
    # fa = obj(x + alpha * p, 'f')
    fa = obj(x + alpha * p, 'f')
    # fk = obj(x, 'f')
    # print(gTp, fa, fk)
    while fa > fk+alpha*c1*gTp and T < maxiter or torch.isnan(fa):
        alpha = alpha/zeta
        # fa = _fa(obj, x, alpha, p) 
        fa = obj(x + alpha * p, 'f')
        T = T + 1    
        if alpha < 1E-18: 
            alpha = 0 # Kill small alpha
            break
    x = x + alpha*p    
    return x, alpha, T

def linesearch_NewtonMR_NC(obj, fk, gTp, x, p, maxiter=200, c1=1e-4, alpha_max=1e8):
    """    
    INPUTS:
        obj: a function handle of f (+ gradient, Hessian-vector-product)
        g: starting gradient gk
        x: starting x
        p: direction p
        maxiter: maximum iteration of Armijo line-search
        alpha: initial step-size
        c1: parameter of Armijo line-search
        c2: parameter of strong Wolfe condition
    OUTPUTS:
        x: results
        alpha: proper step-size
        T: iterations
    """
    T = 0
#    alpha = 2*(1 - 2*c1)
    alpha = 1
    zeta = 2
    # fa = _fa(obj, x, alpha, p)
    fa = obj(x + alpha * p, 'f')   
    # fk = obj(x, 'f')
    # print(fa)
    if torch.isnan(fa):
        print('FisNan')
    if fa <= fk+alpha*c1*gTp:
#        print('NC Linesearch passes')
#        fal = fk
        # while fa <= fk+alpha*c1*gTp and T < maxiter and alpha < alpha_max:
        while fa <= fk+alpha*c1*gTp and T < maxiter and alpha <= alpha_max:
#            print('fa', fa, alpha)
            alpha = alpha*zeta
#            fal = fa
            # fa = _fa(obj, x, alpha, p) 
            fa = obj(x + alpha * p, 'f')   
            T = T + 1    
        x = x + alpha/zeta*p    
        return x, alpha, T
    else:
        while fa > fk+alpha*c1*gTp and T < maxiter or torch.isnan(fa):
#            print('fa', fa, alpha)
            alpha = alpha/zeta
            # print(alpha, zeta)
            # fa = _fa(obj, x, alpha, p)
            fa = obj(x + alpha * p, 'f')   
            T = T + 1    
            if alpha < 1E-18: 
                alpha = 0 # Kill small alpha
                break
        x = x + alpha*p    
        return x, alpha, T