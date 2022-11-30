from MINRES import MINRES
from util import ZERO, NUM_PRINT
from util import myPrint, termination
from line_search import linesearch_NewtonMR, linesearch_NewtonMR_NC
from time import time
import torch

def Newton_MR(obj, x0, HProp, mainLoopMaxItrs, funcEvalMax, innerSolverTol,
                 innerSolverMaxItrs=200, lineSearchMaxItrs=50, gradTol=1e-10,
                 beta=0.25, show=True, 
                 record_txt=None):
    iters = 0
    orcl = 0
    # x = copy.deepcopy(x0)  
    x = x0.clone() 
    fk, gk, Hk = obj(x)
    gk_norm = gk.norm()
    alphak = 1    
    tmk = 0
    rel_res = 1
    iterSolver = 0
    iterLS = 0
    dType = 'None'
    
    # Initialize f0, g0, oracle_call0, time0, alpha0, NC
    record = torch.tensor([fk, gk_norm, 0, 0, 1, 0], device=x.device).reshape(1,-1)
    while True:
        if (show and iters % NUM_PRINT == 0) or orcl >= funcEvalMax or gk_norm < gradTol:
            myPrint(fk, gk_norm, orcl, iters, tmk, alphak, iterLS, iterSolver, 
                    rel_res, dType)
        if termination(fk, gk_norm, gradTol, iters, mainLoopMaxItrs, orcl, funcEvalMax):
            break

        t0 = time()
            
        p, rel_res, iterSolver, rk, dType =  MINRES(Hk, -gk, innerSolverTol, 
                                           innerSolverMaxItrs, reOrth=True)
        if torch.isnan(p).any():
            break
        if dType == 'Sol':
             # when c1 = 0.25, linesearch_NewtonMR is Armijo backward Linesearch
            x, alphak, iterLS = linesearch_NewtonMR(
                 obj, fk, torch.dot(gk, p), x, p, lineSearchMaxItrs, c1=beta)
        else:
            p = rk
             # linesearch_NewtonMR_NC is two-direction linesearch
            x, alphak, iterLS = linesearch_NewtonMR_NC(
                 obj, fk, torch.dot(gk, p), x, p, lineSearchMaxItrs, c1=beta)
        
        #orcl += orc_call(iterSolver, HProp, iterLS)
            
        if alphak < ZERO:
            orcl = funcEvalMax
        else:
            fk, gk, Hk = obj(x)
            gk_norm = gk.norm()
            
        iters += 1  
        tmk += time()-t0

        #record = recording(record, fk, gk_norm, orcl, tmk, alphak, dType)
    
    return x, record