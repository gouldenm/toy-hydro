def reconstruction(Q, xc):
    """Reconstruct the left/right states"""
    #### self._xc = xc = compute_centroids(xe, m)
    dx = xc[2:] - xc[:-2]
    xe = 0.5*(xc[1:] + xc[:-1])
    
    Qm = Q[:-2]
    Q0 = Q[1:-1]
    Qp = Q[2:]

    Qmax = np.maximum(np.maximum(Qp, Qm), Q0)
    Qmin = np.minimum(np.minimum(Qp, Qm), Q0)
        
    #Not the least squares estimate, but what is used in AREPO code release
    grad = (Qp - Qm) / dx

    dQ = grad*(xe[1:] - xc[1:-1])
    Qp = Q0 + dQ

    pos = Qp > Qmax ; neg = Qp < Qmin
    phir = np.where(pos, (Qmax - Q0)/dQ, np.where(neg, (Qmin - Q0)/dQ, 1))
        
    dQ = grad*(xe[0:-1] - xc[1:-1])
    Qm = Q0 + dQ

    pos = Qm > Qmax ; neg = Qm < Qmin
    phil = np.where(pos, (Qmax - Q0)/dQ, np.where(neg, (Qmin - Q0)/dQ, 1))

    alpha = np.maximum(0, np.minimum(1, np.minimum(phir, phil)))
    grad *= alpha
    Qm = Q0 + grad*(xe[0:-1] - xc[1:-1])
    Qp = Q0 + grad*(xe[1:] - xc[1:-1])
    
    return Qm, Qp, grad
