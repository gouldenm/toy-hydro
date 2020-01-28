from __future__ import print_function
import numpy as np

GAMMA = 5/3.
NHYDRO = 3

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

    def _mc(self, Q):
        Qm = Q[:-2]
        Q0 = Q[1:-1]
        Qp = Q[2:]

        def mm(l, r):
            return np.where(l*r > 0, np.where(np.abs(l) < np.abs(r), l, r), 0)

        dQr = Qp - Q0
        dQl = Q0 - Qm 

        mc =  mm(0.5*(dQl + dQr), 2*mm(dQl, dQr))
        return mc

    def reconstruct(self, Q):
        """Reconstruct the left/right states"""
        Qm = Q[:-2]
        Q0 = Q[1:-1]
        Qp = Q[2:]

        Qmax = np.maximum(np.maximum(Qp, Qm), Q0)
        Qmin = np.minimum(np.minimum(Qp, Qm), Q0)
        
        #Not the least squares estimate, but what is used in AREPO code release
        grad = (Qp - Qm) / self._dx

        dQ = grad*(self._xe[1:] - self._xc[1:-1])
        Qp = Q0 + dQ

        pos = Qp > Qmax ; neg = Qp < Qmin
        phir = np.where(pos, (Qmax - Q0)/dQ, np.where(neg, (Qmin - Q0)/dQ, 1))
        
        dQ = grad*(self._xe[0:-1] - self._xc[1:-1])
        Qm = Q0 + dQ

        pos = Qm > Qmax ; neg = Qm < Qmin
        phil = np.where(pos, (Qmax - Q0)/dQ, np.where(neg, (Qmin - Q0)/dQ, 1))

        alpha = np.maximum(0, np.minimum(1, np.minimum(phir, phil)))
        grad *= alpha

        Qm = Q0 + grad*(self._xe[0:-1] - self._xc[1:-1])
        Qp = Q0 + grad*(self._xe[1:] - self._xc[1:-1])

        return Qm, Qp, grad

        
        
    def __call__(self, v_edge, Q):
        """Compute the upwinded face value"""

        Qm, Qp, grad = self.reconstruct(Q)
        
        return np.where(v_edge[2:-2] > 0, Qp[:-1], Qm[1:])



def prim2cons(W):
    U = np.full((len(W), NHYDRO), np.nan) #conserved state vector
    U[:,0] = W[:,0] #gas density
    U[:,1] = W[:,0]*W[:,1] #gas momentum
    U[:,2] = W[:,2]/(GAMMA-1) + (W[:,0]*W[:,1]**2)/2	#gas energy
    return(U)
	
def cons2prim(U):
    W = np.full((len(U), NHYDRO), np.nan) #primitive state vector
    W[:,0] = U[:,0] #gas density
    W[:,1] = U[:,1]/U[:,0] #gas velocity
    W[:,2] = (GAMMA-1)*(U[:,2] - (U[:,1]**2/U[:,0])/2)	#gas pressure
    return(W)
	
def prim2flux(W):
    F = np.full((len(W), NHYDRO), np.nan)
    F[:,0] = W[:,0]*W[:,1] #mass flux
    F[:,1] = W[:,0]*W[:,1]**2 + W[:,2] #momentum flux
    F[:,2] = W[:,1] * (W[:,2]/(GAMMA-1) + (W[:,0]*W[:,1]**2)/2 + W[:,2]) #energy flux
    return(F)


def HLL_solver(WL, WR):
    UL = prim2cons(WL)
    UR = prim2cons(WR)
    
    fL = prim2flux(WL)
    fR = prim2flux(WR)
    
    csl = np.sqrt(GAMMA*WL[:,2]/WL[:,0])
    csr = np.sqrt(GAMMA*WR[:,2]/WR[:,0])
    
    Sm = (WL[:,1] - csl).reshape(-1,1)
    Sp = (WR[:,1] + csr).reshape(-1,1)

    
    # HLL central state
    fHLL = (Sp*fL - Sm*fR + Sp*Sm*(UR - UL)) / (Sp - Sm)

    # Left / Right states
    indexL = Sm.reshape(-1) >= 0
    indexR = Sp.reshape(-1) <= 0
    fHLL[indexL] = fL[indexL]
    fHLL[indexR] = fR[indexR]
	      
    return fHLL

"""_HLLC = HLLC(gamma=GAMMA)
def HLLC_solver(Wl, Wr):
    Ul, Ur = prim2cons(Wl), prim2cons(Wr)
    flux = _HLLC(Ul.T, Ur.T)

    return flux.T
"""

def max_wave_speed(U):
    W = cons2prim(U)
    return np.max(np.abs(W[:,1]) + np.sqrt(GAMMA*W[:,2]/W[:,0]))

        
def solve_euler(Npts, IC, reconstruction, tout, Ca = 0.7):
    """Test schemes using an Explicit TVD RK integration"""
    # Setup up the grid
    stencil = reconstruction.STENCIL
    order = reconstruction.ORDER
    
    shape = Npts + 2*stencil
    dx = 1. / Npts
    xc = np.linspace(-dx*stencil + dx*0.5, 1+ dx*stencil - dx*0.5, shape)
    
    # Reconstruction function:
    R = reconstruction(xc, 0)
    
    def boundary(Q):
        Qb = np.empty([shape, NHYDRO])
        Qb[stencil:-stencil] = Q
        Qb[ :stencil] = Qb[Npts:Npts+stencil]
        Qb[-stencil:] = Qb[stencil:2*stencil]

        return Qb

    def update_stage(U, dt):

        #1. Apply Boundaries
        Ub = boundary(U)

        #2. Compute Primitive variables
        Wb = cons2prim(Ub)

        #3. Reconstruct the edge states
        Wp = np.full([U.shape[0]+2,NHYDRO], np.nan)
        Wm = np.full([U.shape[0]+2,NHYDRO], np.nan)
        for i in range(NHYDRO):
            Wm[:,i], Wp[:,i], grad = R.reconstruct(Wb[:,i])
        
        #4. Compute fluxes
        flux = HLL_solver(Wp[:-1], Wm[1:])

        #5. Update Q
        return U - dt*np.diff(flux, axis=0)/dx


    # Set the initial conditions
    W = IC(xc[stencil:-stencil])
    U = prim2cons(W)

    t = 0
    while t < tout:
        dtmax = Ca * dx / max_wave_speed(U)
        dt = min(dtmax, tout-t)
        if order==3:
            Us =          update_stage(U , dt)
            Us = (3*U +   update_stage(Us, dt))/4
            U  = (  U + 2*update_stage(Us, dt))/3
        elif order == 2:
            Us =      update_stage(U , dt)
            U  = (U + update_stage(Us, dt))/2.
        else:
            U  = update_stage(U, dt)
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
        _, W0 = solve_euler(Ni, IC, scheme, 0, Ca = 0.4)
        x, W = solve_euler(Ni, IC, scheme, 3.0, Ca = 0.4)
        
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
    