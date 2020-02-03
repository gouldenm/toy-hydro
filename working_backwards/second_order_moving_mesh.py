from __future__ import print_function
import numpy as np

GAMMA = 5/3.
NHYDRO = 5

class Arepo2(object):
    """Second-order reconstruction as in AREPO."""
    STENCIL = 2
    ORDER = 2
    def __init__(self, xc, m):
        if m != 0:
            raise ValueError("Arepo2 assumes m = 0")

        self._xc = xc
        #### self._xc = xc = compute_centroids(xe, m)
        self._dx = xc[2:] - xc[:-2]
        self._xe = 0.5*(xc[1:] + xc[:-1])

    def reconstruct(self, Q, xc):
        dx = xc[2:] - xc[:-2]
        xe = 0.5*(xc[1:] + xc[:-1])
        """Reconstruct the left/right states"""
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


def prim2cons(W):
    U = np.full((len(W), NHYDRO), np.nan) #conserved state vector
    U[:,0] = W[:,0] #gas density
    U[:,1] = W[:,0]*W[:,1] #gas momentum
    U[:,2] = W[:,2]/(GAMMA-1) + (W[:,0]*W[:,1]**2)/2    #gas energy
    U[:,3] = W[:,3]                                        #dust density
    U[:,4] = W[:,3]*W[:,4]                             #dust momentum
    return(U)
    
def cons2prim(U):
    W = np.full((len(U), NHYDRO), np.nan) #primitive state vector
    W[:,0] = U[:,0] #gas density
    W[:,1] = U[:,1]/U[:,0] #gas velocity
    W[:,2] = (GAMMA-1)*(U[:,2] - (U[:,1]**2/U[:,0])/2)  #gas pressure
    W[:,3] = U[:,3]                                     #dust density
    W[:,4] = U[:,4]/U[:,3]                             #dust velocity
    return(W)
    
def prim2flux(W):
    F = np.full((len(W), NHYDRO), np.nan)
    F[:,0] = W[:,0]*W[:,1] #mass flux
    F[:,1] = W[:,0]*W[:,1]**2 + W[:,2] #momentum flux
    F[:,2] = W[:,1] * (W[:,2]/(GAMMA-1) + (W[:,0]*W[:,1]**2)/2 + W[:,2]) #energy flux
    F[:,3] = W[:,3]*W[:,4]                                                  #dust mass flux
    F[:,4] = W[:,3]*W[:,4]**2                                              #dust momentum flux
    return(F)


def HLL_solver(WL, WR, vf):
    #5. Transform lab-frame to face frame.
    WL[:,1] -= vf       # subtract face velocity from gas velocity
    WR[:,1] -= vf 
    WL[:,4] -= vf
    WR[:,4] -= vf
        
    UL = prim2cons(WL)
    UR = prim2cons(WR)
    
    fL = prim2flux(WL)
    fR = prim2flux(WR)
    
    csl = np.sqrt(GAMMA*WL[:,2]/WL[:,0])
    csr = np.sqrt(GAMMA*WR[:,2]/WR[:,0])
    
    Sm = (WL[:,1] - csl).reshape(-1,1)
    Sp = (WR[:,1] + csr).reshape(-1,1)

    
    # HLL central state in face frame
    fHLL = np.zeros_like(fL)
    fHLL[:,:3] = (Sp*fL[:,:3] - Sm*fR[:,:3] + Sp*Sm*(UR [:,:3]- UL[:,:3])) / (Sp - Sm)

    # Left / Right states
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
    fHLL_lab[:,2] += 0.5*fHLL[:,0]*vf**2 + fHLL[:,1]*vf
    fHLL_lab[:,4] += fHLL[:,3]*vf
    
    return fHLL_lab

"""_HLLC = HLLC(gamma=GAMMA)
def HLLC_solver(Wl, Wr):
    Ul, Ur = prim2cons(Wl), prim2cons(Wr)
    flux = _HLLC(Ul.T, Ur.T)

    return flux.T
"""

def max_wave_speed(U):
    W = cons2prim(U)
    return np.max(np.abs(W[:,1]) + np.sqrt(GAMMA*W[:,2]/W[:,0]))





def solve_euler(Npts, IC, reconstruction, tout, Ca = 0.7, fixed_v = 0.0, mesh_type = "fixed", 
                K=0.1, dust_scheme = "explicit", FB=0):
    """Test schemes using an Explicit TVD RK integration"""
    # Setup up the grid
    stencil = reconstruction.STENCIL
    order = reconstruction.ORDER
    
    shape = Npts + 2*stencil
    dx0 = 1. / Npts
    xc = np.linspace(-dx0*stencil + dx0*0.5, 1+ dx0*stencil - dx0*0.5, shape)
    dx = (xc[2:] - xc[:-2])*0.5
    
    # Reconstruction function:
    R = reconstruction(xc, 0)
    
    def boundary(Q):
        Qb = np.empty([shape, NHYDRO])
        Qb[stencil:-stencil] = Q
        Qb[ :stencil] = Qb[Npts:Npts+stencil]
        Qb[-stencil:] = Qb[stencil:2*stencil]

        return Qb

    def update_stage(U, dt, xcin):
        #1. Apply Boundaries
        Ub = boundary(U)

        #2. Compute Primitive variables
        Wb = cons2prim(Ub)

        #3. Reconstruct the edge states
        Wp = np.full([U.shape[0]+2,NHYDRO], np.nan)
        Wm = np.full([U.shape[0]+2,NHYDRO], np.nan)
        gradW = np.full([U.shape[0]+2,NHYDRO], np.nan)
        for i in range(NHYDRO):
            Wm[:,i], Wp[:,i], gradW[:,i] = R.reconstruct(Wb[:,i], xcin)
        
        WL = Wp[:-1]; WR = Wm[1:]
        
        #4. Correct velocities
        if mesh_type == "Lagrangian":
            vl = WL[:,1]
            vr = WR[:,1]
            vf =  0.5 * (vl+vr)
        elif mesh_type == "fixed":
            vf = fixed_v
        
        #5. Transform lab-frame to face frame.
        WL[:,1] -= vf       # subtract face velocity from gas velocity
        WR[:,1] -= vf
        WL[:,4] -= vf
        WR[:,4] -= vf
        #5. Compute fluxes
        flux =              HLL_solver(WL, WR, vf)

        #6. Return flux
        return dt*np.diff(flux, axis=0), gradW
    
    def update_stage_prim(W, dWdtL, dWdtR, dt, xcin):
        #1. Apply Boundaries
        Wb = boundary(W)

        #2. Compute Conserved variables
        U = cons2prim(W)

        #3. Reconstruct the edge states
        Wp = np.full([U.shape[0]+2,NHYDRO], np.nan)
        Wm = np.full([U.shape[0]+2,NHYDRO], np.nan)
        gradW = np.full([U.shape[0]+2,NHYDRO], np.nan)
        for i in range(NHYDRO):
            Wm[:,i], Wp[:,i], gradW[:,i] = R.reconstruct(Wb[:,i], xcin)
        
        WL = Wp[:-1]; WR = Wm[1:]
        
        #4. Correct velocities
        if mesh_type == "Lagrangian":
            vl = WL[:,1]
            vr = WR[:,1]
            vf =  0.5 * (vl+vr)
        elif mesh_type == "fixed":
            vf = fixed_v
        
        
        
        #Include also time correction:
        WL += dWdtL*dt
        WR += dWdtR*dt
        #4. Compute fluxes
        flux =                  HLL_solver(Wp[:-1], Wm[1:], vf)

        #5. Update Q
        return dt*np.diff(flux, axis=0)

    
    def time_diff_W(W, gradW, vf):# ###, FB):
        W[:,1] -= vf
        W[:,4] -= vf
        dWdt = np.zeros_like(W)
        
        rho_g = W[:, 0]
        grad_rho_g = gradW[:, 0]
        
        v_g = W[:, 1]
        grad_v_g = gradW[:, 1]
        
        P = W[:, 2]
        grad_P = gradW[:, 2]
        
        
        rho_d = W[:, 3]
        grad_rho_d = gradW[:, 3]
        
        v_d = W[:, 4]
        grad_v_d = gradW[:, 4]
        
        dWdt[:,0] = v_g*grad_rho_g + rho_g*grad_v_g
        dWdt[:,1] = grad_P/rho_g + v_g*grad_v_g # ###+ FB*( - self.K*rho_d*v_d + self.K*rho_d*v_g)
        dWdt[:,2] = GAMMA*P*grad_v_g + v_g * grad_P
        dWdt[:,3] =  v_d*grad_rho_d + rho_d*grad_v_d
        dWdt[:,4] = v_d*grad_v_d + K*rho_g*v_d - K*rho_g*v_g
        dWdt *= -1
        return(dWdt)
    
    def update_mesh(xc, dt, W):
    #  Modify x coordinate based on velocity of cell centre
        if mesh_type == "Lagrangian":
            Wb = boundary(W)
            xc = xc + Wb[:,1]*dt
        else:
            xc = xc + fixed_v*dt
        xc_ng = xc[stencil:-stencil]
        dx = (xc[2:] - xc[:-2])*0.5
        return(xc, dx)
    
    # Set the initial conditions
    W = IC(xc[stencil:-stencil])
    U = prim2cons(W)
    Q = U * dx[1:-1].reshape(-1,1)

    t = 0
    while t < tout:
        # 0. Calculate new timestep
        dtmax = Ca * min(dx) / max_wave_speed(U)
        dt = min(dtmax, tout-t)
        
        # 1. Apply boundaries
        Qb = boundary(xc, Q)
        
        #2. Compute prim variables
        Ub = Qb / dx[1:-1].reshape(-1,1)
        Wb = cons2prim(Ub)
        
        #3. Compute gradient
        Wp = np.full([U.shape[0]+2,NHYDRO], np.nan)
        Wm = np.full([U.shape[0]+2,NHYDRO], np.nan)
        gradW = np.full([U.shape[0]+2,NHYDRO], np.nan)
        for i in range(NHYDRO):
            Wm[:,i], Wp[:,i], gradW[:,i] = R.reconstruct(Wb[:,i], xcin)
        
        #4. Compute state at cell edges
        WL = Wp[:-1]; WR = Wm[1:]
        
        #5. Set interface velocities TODO: compare this with how velocities are got in broken.py
        if mesh_type == "Lagrangian":
            vl = WL[:,1]
            vr = WR[:,1]
            vf =  0.5 * (vl+vr)
        elif mesh_type == "fixed":
            vf = fixed_v
        
        #6. Compute flux at t
        
        F1, gradW1 =                        update_stage(U , dt, xc)
        
        # 5) Compute predicted dWdt for both sides (requires FACE velocity, so can't do both sides at once)
        W = cons2prim(U)
        Wb = boundary(W)[1:-1]#match gradient extent
        WL = np.copy(Wb[:-1]); WR = np.copy(Wb[1:])
        if mesh_type == "Lagrangian":
            vl = WL[:,1]
            vr = WR[:,1]
            vf =  0.5 * (vl+vr)
        elif mesh_type == "fixed":
            vl = fixed_v
            vr = fixed_v
            vf = fixed_v
            
        dWdtL = time_diff_W(WL, gradW1[:-1], vl)
        dWdtR = time_diff_W(WR, gradW1[1:], vr)
        
        
        # 6.) Update mesh
        xc, dx = update_mesh(xc, dt, W)
        
        # 7) Compute fluxes again
        Fp =                                update_stage_prim(W, dWdtL, dWdtR, dt, xc)
        
        # 8) Time average fluxes (both used dt, so just *0.5)
        Qold = np.copy(Q)
        flux = 0.5*(F1+Fp)
        
        Q[:,:4] = Qold[:,:4] - flux[:,:4]
        
        p_g = Qold[:,2]
        p_d = Qold[:,4]
        rho_g = Q[:,0] / dx[1:-1]
        rho_d = Q[:,3] / dx[1:-1]
        
        if dust_scheme == "explicit":
            Q[:,4] = (p_d - flux[:,4] + K*rho_d*dt*p_g)   /    (1 + K*rho_g*dt)
            
        U = Q/dx[1:-1].reshape(-1,1)
        
    
        t = min(tout, t+dt)

    xc = xc[stencil:-stencil]
    return xc, cons2prim(U)





def _test_convergence(IC, pmin=4, pmax=10, figs_evol=None, fig_err=None):
    N = 2**np.arange(pmin, pmax+1)
    scheme = Arepo2
    errs = []
    c=None
    label=scheme.__name__
    for Ni in N:
        print (scheme.__name__, Ni)
        _, W0 = solve_euler(Ni, IC, scheme, 0, Ca = 0.4, mesh_type = "Lagrangian",  fixed_v = 5.0, FB=0)
        x, W = solve_euler(Ni, IC, scheme, 3.0, Ca = 0.4, mesh_type = "Lagrangian", fixed_v = 5.0, FB=0)
        if figs_evol is not None:
            c = figs_evol[0].plot(x, W[:,0], c=c, 
                                  label=label)[0].get_color()
            figs_evol[1].plot(x, W[:,1], c=c)
            figs_evol[2].plot(x, W[:,2], c=c)
            
            figs_evol[0].set_ylabel('Density')
            figs_evol[1].set_ylabel('Velocity')
            figs_evol[2].set_ylabel('Pressure')
            figs_evol[2].set_xlabel('x')
            
            label=None

        errs.append(np.sqrt(np.mean((W[:,1] - W0[:,1])**2)))
    if fig_err is not None:
        fig_err.loglog(N, errs, c=c, label=scheme.__name__)

    if fig_err is not None:
        fig_err.set_xlabel('N')
        fig_err.set_ylabel('L2 velocity error')
        fig_err.plot(N, 1e-4/N**2, label='1/N^2', c='k')
        fig_err.legend()
    if figs_evol is not None:
        figs_evol[0].legend(loc='best',frameon=False)

def init_wave(xc, cs0=1.0, rho0=1.0, v0=1.0, drho=1e-6):
    kx = 2*np.pi*xc

    W = np.full([len(xc), NHYDRO], np.nan)
    W[:,0] = rho0 + drho*np.sin(kx)
    W[:,1] = v0 + drho*cs0*np.sin(kx)
    W[:,2] = (rho0*cs0**2/GAMMA) * (W[:,0]/rho0)**GAMMA
    return W


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    _test_convergence(init_wave, 
                      figs_evol=plt.subplots(3, 1)[1],
                      fig_err=plt.subplots(1)[1])

    plt.show()
    