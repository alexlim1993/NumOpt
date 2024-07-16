# -*- coding: utf-8 -*-
"""
Created on Tue Oct 25 12:54:55 2022

@author: uqalim8
"""
import torch
ZERO = 1e-8

def CG(A, b, tol = 1e-2, maxite = 100):
    x0 = torch.zeros(b.shape[0], dtype = torch.float64)
    rkm1 = b - Ax(A, x0)
    pk = rkm1.clone()
    xkm1 = x0
    k = 0
    while torch.norm(rkm1) > tol and k < maxite:
        Apk = Ax(A, pk)
        rk_1sq = torch.norm(rkm1)**2
        alpha = rk_1sq / torch.dot(pk, Apk)
        xk = xkm1 + alpha * pk
        rk = rkm1 - alpha * Apk
        beta = torch.norm(rk)**2 / rk_1sq
        pk = rk + beta * pk
        rkm1 = rk
        xkm1 = xk
        k += 1
    return xkm1, k

def CGSteihaug(H, g, delta, tol, maxite):
    
    z = torch.zeros_like(g)
    # if torch.norm(g) < tol:
    #     return z, "||g||<tol", 1, 0
    
    j = 0
    d, r = -g.clone(), g.clone()
    norm_g = torch.norm(g)
    norm_r = norm_g
    Bd = Ax(H, d)
    norm_Hg = torch.norm(Bd)
    norm_Hr = norm_Hg
    while j <= maxite:
        dBd = torch.dot(d, Bd)
        j += 1
        if dBd <= 0:
            dz = torch.dot(d, z)
            norm_d, norm_z = torch.norm(d), torch.norm(z)
            numerator = - dz + torch.sqrt(dz**2  - norm_d**2 * (norm_z**2 - delta**2))
            tau = numerator / norm_d**2
            p = z + tau * d
            m0_mk = - torch.dot(g, p) - torch.dot(p, Ax(H, p)) / 2
            return p, norm_r / norm_g, norm_Hr / norm_Hg, "NC", m0_mk, j
        
        norm_r = torch.dot(r, r)
        alpha = norm_r / dBd
        zp1 = z + alpha * d
        if torch.norm(zp1) >= delta:
            dz = torch.dot(d, z)
            norm_d, norm_z = torch.norm(d), torch.norm(z)
            numerator = - dz + torch.sqrt(dz**2  - norm_d**2 * (norm_z**2 - delta**2))
            tau = numerator / norm_d**2
            p = z + tau * d
            m0_mk = - torch.dot(g, p) - torch.dot(p, Ax(H, p)) / 2
            return p, norm_r / norm_g, norm_Hr / norm_Hg, "SOL,=", m0_mk, j

        z = zp1
        r = r + alpha * Bd
        if torch.norm(r) < tol:
            p = z
            m0_mk = - torch.dot(g, p) - torch.dot(p, Ax(H, p)) / 2
            return p, norm_r / norm_g, norm_Hr / norm_Hg, "SOL,<", m0_mk, j
        
        norm_rp1 = torch.dot(r, r)
        beta = norm_rp1 / norm_r
        d = -r + beta * d
        norm_r = norm_rp1
        Bdkp1 = Ax(H, d)
        norm_Hr = torch.norm(Bdkp1 - beta * Bd)
        Bd = Bdkp1
     
    p = z
    m0_mk = - torch.dot(g, p) - torch.dot(p, Ax(H, p)) / 2

    return p, norm_r / norm_g, norm_Hr / norm_Hg, "MAX,<", m0_mk, j
        

def CappedCG(H, b, zeta, epsilon, maxiter, M=0):
    g = -b
    y =  torch.zeros_like(g)
    kappa, tzeta, tau, T = para(M, epsilon, zeta)
    tHy = y.clone()
    tHY = y.reshape(-1, 1)
    Y = y.reshape(-1, 1)
    r = g
    p = -g
    tHp = Ax(H, p) + 2*epsilon*p
    j = 1
    ptHp = torch.dot(p, tHp)
    norm_g = torch.norm(g)
    norm_p = norm_g
    rr = torch.dot(r, r)
    dType = 'Sol'
    relres = 1
    if ptHp < epsilon*norm_p**2:
        d = p
        dType = 'NC'
        return d, dType, j, ptHp, 1, 1
    norm_Hp = torch.norm(tHp - 2*epsilon*p)
    norm_Hb = norm_Hp
    if norm_Hp > M*norm_p:
        M = norm_Hp/norm_p
        kappa, tzeta, tau, T = para(M, epsilon, zeta)    
    while j < maxiter:
        alpha = rr/ptHp
        y = y + alpha*p
        #Y = torch.cat((Y, y.reshape(-1, 1)), 1) #record y
        norm_y = torch.norm(y)
        tHy = tHy + alpha*tHp
        #tHY = torch.cat((tHY, tHy.reshape(-1, 1)), 1) # record tHy
        norm_Hy = torch.norm(tHy - 2*epsilon*y)
        r = r + alpha*tHp
        rr_new = torch.dot(r, r) 
        beta = rr_new/rr
        rr = rr_new
        p = -r + beta*p #calculate Hr
        norm_p = torch.norm(p)    
        tHp_new = Ax(H, p) + 2*epsilon*p #the only Hessian-vector product
        j = j + 1
        tHr = beta*tHp - tHp_new #calculate Hr
        tHp = tHp_new
        norm_Hp = torch.norm(tHp - 2*epsilon*p)
        ptHp = torch.dot(p, tHp)  
        if  norm_Hp> M*norm_p:
            M = norm_Hp/norm_p
            kappa, tzeta, tau, T = para(M, epsilon, zeta)
        if norm_Hy > M*norm_y:
            M = norm_Hy/norm_y
            kappa, tzeta, tau, T = para(M, epsilon, zeta)
        norm_r = torch.norm(r)
        relres = norm_r/norm_g
#        print(norm_r/norm_g, tzeta)
        norm_Hr = torch.norm(tHr - 2*epsilon*r)
#        print(norm_r, torch.norm(H(y) + g))
        if  norm_Hr> M*norm_r:
            M = norm_Hr/norm_r         
            kappa, tzeta, tau, T = para(M, epsilon, zeta)
        if torch.dot(y, tHy) < epsilon*norm_y**2:
            d = y
            dType = 'NC'
            # print('y')
            return d, dType, j, torch.dot(y, tHy), relres, norm_Hr / norm_Hb
        elif norm_r < tzeta*norm_g:
            # print('relres', relres)
            d = y
            return d, dType, j, 0, relres, norm_Hr / norm_Hb
        elif torch.dot(p, tHp) < epsilon*norm_p**2:
            d = p
            dType = 'NC'
            # print('p')
            return d, dType, j, torch.dot(p, tHp), relres, norm_Hr / norm_Hb
        elif norm_r > torch.sqrt(T*tau**j)*norm_g:
            print('Uncomment tensors Y, tHY')
            alpha_new = rr/ptHp
            y_new = y + alpha_new*p            
            tHy_new = tHy + alpha_new*tHp
            for i in range(j):
                dy = y_new - Y[:, i]
                dtHy = tHy_new - tHY[:, i]
                if torch.dot(dy, dtHy) < epsilon*torch.norm(dy)**2:
                    d = dy
                    dType = 'NC'
                    print('dy')
                    return d, dType, j, torch.dot(dy, dtHy), relres, norm_Hr / norm_Hb
    print('Maximum iteration exceeded!')
    return y, dType, j, 0, relres, norm_Hr / norm_Hb

def para(M, epsilon, zeta):
    # if torch.tensor(M):
    #     M = M.item()
    kappa = (M + 2*epsilon)/epsilon
    tzeta = zeta/3/kappa
    # print('kappa', kappa)
    sqk = torch.sqrt(torch.tensor(float(kappa)))
    tau = sqk/(sqk + 1)
    T = 4*kappa**4/(1 + torch.sqrt(tau))**2
    return kappa, tzeta, tau, T    

def Ax(A, x):
    if callable(A):
        Ax = A(x)
    else:
        Ax = A @ x
    return Ax

# if __name__ == "__main__":
#     A = np.random.rand(100, 100)
#     A = A.T @ A
#     b = A @ np.ones((100, 1))
#     x, k, r = CG(A, b)
#     print(np.linalg.norm(A @ x - b), k, r)
