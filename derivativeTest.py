import torch
import matplotlib.pyplot as plt

def derivativeTest(fun, x0):
    """
    INPUTS:
        fun: a function handle that gives f, g, Hv
        x0: starting point
    OUTPUTS:
        derivative test plots
    """
    #x0 = x0.resize(len(x0),1)
    fun0 = fun(x0)
    dx = torch.randn(len(x0), dtype = torch.float64)
    M = 20;
    dxs = torch.zeros((M,1), dtype = torch.float64)
    firsterror = torch.zeros((M,1), dtype = torch.float64)
    seconderror = torch.zeros((M,1), dtype = torch.float64)
    
    for i in range(M):
        x = x0 + dx
        fun1 = fun(x)
        H0 = Ax(fun0[2],dx)
        firsterror[i] = abs(fun1[0] - (fun0[0] +
                dx.T @ fun0[1]))/abs(fun0[0])
        seconderror[i] = abs(fun1[0] - (fun0[0] +
                dx.T @ fun0[1] + 0.5 * dx.T @ H0))/abs(fun0[0])
        print('First Order Error is %8.2e;   Second Order Error is %8.2e'% (
                firsterror[i], seconderror[i]))
        dxs[i] = torch.norm(dx)
        dx = dx/2
    
    step = [2**(-i-1) for i in range(M)]
    plt.figure(figsize=(12,8))
    plt.subplot(211)
    plt.loglog(step, abs(firsterror.clone().detach()),'r', label = '1st Order Error')
    plt.loglog(step, dxs**2,'b', label = 'Theoretical Order')
    plt.gca().invert_xaxis()
    plt.legend()
    
    plt.subplot(212)
    plt.loglog(step, abs(seconderror.clone().detach()),'r', label = '2nd Order Error')
    plt.loglog(step, dxs**3,'b', label = 'Theoretical Order')
    plt.gca().invert_xaxis()
    plt.legend()
    
            
    return plt.show()


def Ax(A, x):
    if callable(A):
        Ax = A(x)
    else:
        Ax =A.dot(x)
    return Ax

    