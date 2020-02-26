def HLL_solve(WLin, WRin, vf, GAMMA, FB):
    # transform lab frame to face frame
    WL = np.copy(WLin)
    WR = np.copy(WRin)
    WL[:,1] = WLin[:,1] - vf       # subtract face velocity from gas velocity
    WR[:,1] = WRin[:,1] - vf
    WL[:,4] = WLin[:,4] - vf
    WR[:,4] = WRin[:,4] - vf
        
    UL = prim2cons(WL, GAMMA, FB)
    UR = prim2cons(WR, GAMMA, FB)
    
    fL = prim2flux(WL, GAMMA, FB)
    fR = prim2flux(WR, GAMMA, FB)
    
    csl = np.sqrt(GAMMA*WL[:,2]/WL[:,0])
    csr = np.sqrt(GAMMA*WR[:,2]/WR[:,0])
    
    Sm = (WL[:,1] - csl).reshape(-1,1)
    Sp = (WR[:,1] + csr).reshape(-1,1)
    
    # HLL central state in face frame
    fHLL = np.zeros_like(fL)
    fHLL[:,:3] = (Sp*fL[:,:3] - Sm*fR[:,:3] + Sp*Sm*(UR [:,:3]- UL[:,:3])) / (Sp - Sm)
    
    indexL = Sm.reshape(-1) >= 0
    indexR = Sp.reshape(-1) <= 0
    fHLL[indexL,:3] = fL[indexL,:3]
    fHLL[indexR,:3] = fR[indexR,:3]
    
    # ### ### ### DUST ### ### ###
    #    Calculate signal speed for dust
    ld = (np.sqrt(WL[:,3])*WL[:,4] + np.sqrt(WR[:,3])*WR[:,4]) / (np.sqrt(WL[:,3]) + np.sqrt(WR[:,3]))
    
    #   Calculate DUST flux in frame of face (note if vL < 0 < vR, then fHLL = 0.)
    indexL = (ld > 1e-15) & np.logical_not(((WL[:,4] < 0) & (WR[:,4] > 0)))
    indexC = (np.abs(ld) < 1e-15 ) & np.logical_not(((WL[:,4] < 0) & (WR[:,4] > 0)))
    indexR = (ld < -1e-15) & np.logical_not(((WL[:,4] < 0) & (WR[:,4] > 0)))
    fHLL[indexL,3:] = fL[indexL,3:]
    fHLL[indexC,3:] = (fL[indexC,3:] + fR[indexC,3:])/2.
    fHLL[indexR,3:] = fR[indexR,3:]
    
    w_f = ld.reshape(-1,1)
    f_dust = w_f*np.where(w_f > 0, UL[:,3:], UR[:,3:]) 
    
    fHLL[:, 3:] = f_dust
    
    # Correct to lab frame
    fHLL_lab = np.copy(fHLL)
    fHLL_lab[:,1] += fHLL[:,0]*vf
    fHLL_lab[:,2] += 0.5*(fHLL[:,0] + FB*fHLL[:,3])*vf**2 + (fHLL[:,1]+FB*fHLL[:,4])*vf
    fHLL_lab[:,4] += fHLL[:,3]*vf
    
    return fHLL_lab