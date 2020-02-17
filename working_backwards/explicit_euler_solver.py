from __future__ import print_function
import numpy as np
from dustywave_sol2 import *
from dust_settling_sol import *

NHYDRO = 5
FB = 0
K=10.0

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
        grad = (Qp - Qm) / dx.reshape(-1,1)

        dQ = grad*(xe[1:] - xc[1:-1]).reshape(-1,1)
        Qp = Q0 + dQ

        pos = Qp > Qmax ; neg = Qp < Qmin
        phir = np.where(pos, (Qmax - Q0)/dQ, np.where(neg, (Qmin - Q0)/dQ, 1))
        
        dQ = grad*(xe[0:-1] - xc[1:-1]).reshape(-1,1)
        Qm = Q0 + dQ

        pos = Qm > Qmax ; neg = Qm < Qmin
        phil = np.where(pos, (Qmax - Q0)/dQ, np.where(neg, (Qmin - Q0)/dQ, 1))

        alpha = np.maximum(0, np.minimum(1, np.minimum(phir, phil)))
        grad *= alpha
        
        return grad


def prim2cons(W, GAMMA):
    U = np.full((len(W), NHYDRO), np.nan) #conserved state vector
    U[:,0] = W[:,0] #gas density
    U[:,1] = W[:,0]*W[:,1] #gas momentum
    U[:,2] = W[:,2]/(GAMMA-1) + (W[:,0]*W[:,1]**2)/2.    + FB*(W[:,3]*W[:,4]**2)/2.   #gas energy + dust KE
    U[:,3] = W[:,3]                                        #dust density
    U[:,4] = W[:,3]*W[:,4]                             #dust momentum
    return(U)
    
def cons2prim(U, GAMMA):
    W = np.full((len(U), NHYDRO), np.nan) #primitive state vector
    W[:,0] = U[:,0] #gas density
    W[:,1] = U[:,1]/U[:,0] #gas velocity
    W[:,2] = (GAMMA-1)*(U[:,2] - (U[:,1]**2/U[:,0])/2.   - FB*(U[:,4]**2/U[:,3])/2. )  #gas pressure
    W[:,3] = U[:,3]                                     #dust density
    W[:,4] = U[:,4]/U[:,3]                             #dust velocity
    return(W)
    
def prim2flux(W, GAMMA):
    F = np.full((len(W), NHYDRO), np.nan)
    F[:,0] = W[:,0]*W[:,1] #mass flux
    F[:,1] = W[:,0]*W[:,1]**2 + W[:,2] #momentum flux
    F[:,2] = W[:,1] * (W[:,2]/(GAMMA-1) + (W[:,0]*W[:,1]**2)/2 + W[:,2]) \
             +  FB* W[:,4] * (W[:,3]*W[:,4]**2)/2.                 #gas energy flux + dust energy flux
    F[:,3] = W[:,3]*W[:,4]                                                  #dust mass flux
    F[:,4] = W[:,3]*W[:,4]**2                                              #dust momentum flux
    return(F)


def HLL_solver(WLin, WRin, vf, GAMMA):
    # transform lab frame to face frame
    WL = np.copy(WLin)
    WR = np.copy(WRin)
    WL[:,1] = WLin[:,1] - vf       # subtract face velocity from gas velocity
    WR[:,1] = WRin[:,1] - vf
    WL[:,4] = WLin[:,4] - vf
    WR[:,4] = WRin[:,4] - vf
        
    UL = prim2cons(WL, GAMMA)
    UR = prim2cons(WR, GAMMA)
    
    fL = prim2flux(WL, GAMMA)
    fR = prim2flux(WR, GAMMA)
    
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
    fHLL_lab[:,2] += 0.5*(fHLL[:,0] + FB*fHLL[:,3])*vf**2 + (fHLL[:,1]+FB*fHLL[:,4])*vf
    fHLL_lab[:,4] += fHLL[:,3]*vf
    
    return fHLL_lab

"""_HLLC = HLLC(gamma=GAMMA)
def HLLC_solver(Wl, Wr):
    Ul, Ur = prim2cons(Wl), prim2cons(Wr)
    flux = _HLLC(Ul.T, Ur.T)

    return flux.T
"""

def max_wave_speed(U, GAMMA):
    W = cons2prim(U, GAMMA)
    return np.max(np.abs(W[:,1]) + np.sqrt(GAMMA*W[:,2]/W[:,0]))





def solve_euler(Npts, IC, tout, Ca = 0.5, fixed_v = 0.0, mesh_type = "fixed", 
                dust_scheme = "explicit", b_type = "periodic", 
                dust_gas_ratio = 1.0,
                gravity = 0.0,
                dust_reflect = False,  #ignore reflection of dust velocity unless set to True
                GAMMA=5./3.): 
    """Test schemes using an Explicit TVD RK integration"""
    # Setup up the grid
    reconstruction = Arepo2
    stencil = reconstruction.STENCIL
    order = reconstruction.ORDER
    
    shape = Npts + 2*stencil
    dx0 = 1. / Npts
    xc = np.linspace(-dx0*stencil + dx0*0.5, 1+ dx0*stencil - dx0*0.5, shape)
    dx = (xc[2:] - xc[:-2])*0.5
    
    # Reconstruction function:
    R = reconstruction(xc, 0)
    
    
    def boundary(Q):
        if b_type == "periodic":
            Qb = np.empty([shape, NHYDRO])
            Qb[stencil:-stencil] = Q
            Qb[ :stencil] = Qb[Npts:Npts+stencil]
            Qb[-stencil:] = Qb[stencil:2*stencil]
        
        elif b_type == "reflecting":
            Qb = np.empty([shape, NHYDRO])
            Qb[stencil:-stencil] = Q
            Qb[0] = Qb[3]
            Qb[1] = Qb[2]
            Qb[-1] = Qb[-4]
            Qb[-2] = Qb[-3]
            
            #flip signs for velocities
            Qb[0,1] = - Qb[3,1]
            Qb[1,1] = - Qb[2,1]
            Qb[-1,1] = - Qb[-4,1]
            Qb[-2,1] = - Qb[-3,1]
            
            if dust_reflect == True:
                Qb[0,4] = - Qb[3,4]
                Qb[1,4] = - Qb[2,4]
                Qb[-1,4] = - Qb[-4,4]
                Qb[-2,4] = - Qb[-3,4]
            
        
        elif b_type == "flow":
            Qb = np.empty([shape, NHYDRO])
            Qb[stencil:-stencil] = Q
            Qb[0] = Qb[2]
            Qb[1] = Qb[2]
            Qb[-2] = Qb[-3]
            Qb[-1] = Qb[-3]
            
        elif b_type == "inflowL_and_reflectR":
            Qb = np.empty([shape, NHYDRO])
            Qb[stencil:-stencil] = Q
            #   inflow on left
            Qb[0] = Qb[2]
            Qb[1] = Qb[2]
            #   reflect on right
            Qb[-1] = Qb[-4]
            Qb[-2] = Qb[-3]
            Qb[-1,1] = -Qb[-4,1]
            Qb[-2,1] = -Qb[-3,1]
            Qb[-1,4] = -Qb[-4,4]
            Qb[-2,4] = -Qb[-3,4]
        
        return Qb

    
    def time_diff_W(W, gradW, vf):# ###, FB):

        dWdt = np.zeros_like(W)
        
        rho_g = W[:, 0]
        grad_rho_g = gradW[:, 0]
        
        v_g = W[:, 1] - vf
        grad_v_g = gradW[:, 1]
        
        P = W[:, 2]
        grad_P = gradW[:, 2]
        
        
        rho_d = W[:, 3]
        grad_rho_d = gradW[:, 3]
        
        v_d = W[:, 4] - vf
        grad_v_d = gradW[:, 4]
        
        dWdt[:,0] = v_g*grad_rho_g + rho_g*grad_v_g
        dWdt[:,1] = grad_P/rho_g + v_g*grad_v_g 
        dWdt[:,2] = GAMMA*P*grad_v_g + v_g * grad_P
        dWdt[:,3] =  v_d*grad_rho_d + rho_d*grad_v_d
        dWdt[:,4] = v_d*grad_v_d 
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
    W = IC(xc[stencil:-stencil], dust_gas_ratio= dust_gas_ratio, gravity=gravity, GAMMA=GAMMA)
    U = prim2cons(W, GAMMA)
    Q = U * dx[1:-1].reshape(-1,1)
    t = 0
    while t < tout:
        # 0) Calculate new timestep
        dtmax = Ca * min(dx) / max_wave_speed(U, GAMMA)
        dt = min(dtmax, tout-t)
        
        #1. Apply Boundaries
        Ub = boundary(U)

        #2. Compute Primitive variables
        W = cons2prim(U, GAMMA)
        Wb = cons2prim(Ub, GAMMA)

        #3. Reconstruct the edge states
        gradW = R.reconstruct(Wb, xc)
        
        xe = 0.5*(xc[1:] + xc[:-1])
        Wm = Wb[1:-1] + gradW*(xe[0:-1] - xc[1:-1]).reshape(-1,1)
        Wp = Wb[1:-1] + gradW*(xe[1:] - xc[1:-1]).reshape(-1,1)
        WL = Wp[:-1]; WR = Wm[1:]
        
        #4. Correct velocities
        if mesh_type == "Lagrangian":
            vl = WL[:,1]
            vr = WR[:,1]
            vf =  0.5 * (vl+vr)
            vc = Wb[1:-1,1]
        else:
            vf = fixed_v
            vc = np.full_like(Wb[1:-1,1], fixed_v)
            
        #5. Compute first flux
        flux_0 =              HLL_solver(WL, WR, vf, GAMMA)
        F0 = dt*np.diff(flux_0, axis=0)
        
        # 7a. Compute edge states W at t+td, starting with centre
        Wb = boundary(W)[1:-1]#match gradient extent
        WL = np.copy(Wb[:-1]); WR = np.copy(Wb[1:])
        
        dWdt = time_diff_W(Wb, gradW, vc)
        
        # 7b. Update mesh
        dxold = np.copy(dx)
        xc, dx = update_mesh(xc, dt, W)
        
        #7c. predict cell centre
        Ws = boundary(W)[1:-1]
        
        #7d. include drag terms (explicit first...)
        rho_d = Ws[:, 3]
        rho_g = Ws[:, 0]
        Ws[:,1] += FB*K*rho_d*(Ws[:, 4] - Ws[:, 1])*dt
        #Ws[:,4] -=    K*rho_g*(Ws[:, 4] - Ws[:, 1])*dt       #dust
        Ws += dWdt*dt
        
        #7e. Include constant gravity term, if applicable
        Ws[:,1] += gravity*dt #either 0.0 or 1.0
        Ws[:,4] += gravity*dt
        #7f. Reconstruct the edge states        
        xe = 0.5*(xc[1:] + xc[:-1])
        Wm = Ws + gradW*(xe[0:-1] - xc[1:-1]).reshape(-1,1)
        Wp = Ws + gradW*(xe[1:] - xc[1:-1]).reshape(-1,1)
        WL = Wp[:-1]; WR = Wm[1:]
        
        #8. Compute second flux
        flux_1 =                  HLL_solver(Wp[:-1], Wm[1:], vf, GAMMA)

        #9. Update Q
        F1 = dt*np.diff(flux_1, axis=0)
        
        # 10. Time average fluxes (both used dt, so just *0.5)
        Qold = np.copy(Q)
        flux = - 0.5*(F1+F0)
        
        Q = Qold + flux
        Utemp = Q/dx[1:-1].reshape(-1,1)
        
        #11. Recompute Q for gas / dust momenta...
        p_g = Qold[:,1].copy()
        p_d = Qold[:,4].copy()
        f_g = flux[:,1].copy()
        f_d = flux[:,4].copy()
        rho_d = Utemp[:,0].copy()
        rho_g = Utemp[:,3].copy()
        rho = FB*rho_d + rho_g
        eps_g = rho_g / rho 
        eps_d = rho_d / rho 
        
        dp_0 = K*dt*(Qold[:,0]*Qold[:,4] - Qold[:,3]*Qold[:,1])/dxold[1:-1].reshape(-1)
        dp_1 = K*dt*(Q[:,0]*Q[:,4] - Q[:,3]*Q[:,1])/dx[1:-1].reshape(-1)
        
        dp = 0.5*(dp_0 + dp_1)
        
        Q[:,4] -= dp
        Q[:,1] += FB*dp
        
        # Include const gravity term
        Q[:,4] += gravity*dt*Q[:,3]
        Q[:,1] += gravity*dt*Q[:,0]
        
        #12. Update U
        U = Q/dx[1:-1].reshape(-1,1)
    
        t = min(tout, t+dt)
    xc = xc[stencil:-stencil]
    return xc, cons2prim(U, GAMMA)





def _test_convergence(IC, pmin=4, pmax=10, figs_evol=None, fig_err=None, t_final=3.0):
    N = 2**np.arange(pmin, pmax+1, GAMMA=5./3.)
    scheme = Arepo2
    errs_gas = []
    errs_dust = []
    c=None
    label=scheme.__name__
    for Ni in N:
        print (scheme.__name__, Ni)
        _, W0 = solve_euler(Ni, IC, 0, Ca = 0.4, mesh_type = "Lagrangian",  fixed_v = 1.0)
        x, W = solve_euler(Ni, IC, t_final, Ca = 0.4, mesh_type = "Lagrangian", fixed_v = 1.0)
        true = IC(x, t=t_final)
        if figs_evol is not None:
            c = figs_evol[0].plot(x, W[:,0], c=c, 
                                  label="gas")[0].get_color()
            figs_evol[0].plot(x, W[:,3], 
                                  label="dust "+ str(Ni))
            figs_evol[1].plot(x, W[:,1], c=c)
            figs_evol[1].plot(x, W[:,4], 
                                  label="dust "+ str(Ni))
            figs_evol[2].plot(x, W[:,2], c=c)
            
            figs_evol[0].set_ylabel('Density')
            figs_evol[1].set_ylabel('Velocity')
            figs_evol[2].set_ylabel('Pressure')
            figs_evol[2].set_xlabel('x')
            
            label=None
        
        err = W - true
        errs_gas.append(np.sqrt(np.mean((err[:,1])**2)))
        errs_dust.append(np.sqrt(np.mean((err[:,4])**2)))
    if fig_err is not None:
        fig_err.loglog(N, errs_gas, c=c, label=scheme.__name__ + " gas", ls = "-")
        fig_err.loglog(N, errs_dust, c=c, label=scheme.__name__ + " dust", ls="--")

    if fig_err is not None:
        fig_err.set_xlabel('N')
        fig_err.set_ylabel('L2 velocity error')
        fig_err.plot(N, 1e-4/N**2, label='1/N^2', c='k')
        fig_err.legend()
    if figs_evol is not None:
        figs_evol[0].legend(loc='best',frameon=False)
        figs_evol[0].plot(x, true[:,0], c="black", linestyle="-", 
                                  label="gas")[0].get_color()
        figs_evol[0].plot(x, true[:,3], c="black", 
                                  label="dust", linestyle = "--")
        figs_evol[1].plot(x, true[:,1], c="black", linestyle="-", 
                                  label="gas")[0].get_color()
        figs_evol[1].plot(x, true[:,4], c="black", 
                                  label="dust", linestyle = "--")

def init_wave(xc, cs0=1.0, rho0=1.0, v0=1.0, drho=1e-6, t=0, dust_gas_ratio = 1.0, GAMMA=5./3.):
    if t == 0:
        kx = 2*np.pi*xc
    
        W = np.full([len(xc), NHYDRO], np.nan)
        #gas
        W[:,0] = rho0 + drho*np.sin(kx)
        W[:,1] = v0 + drho*cs0*np.sin(kx)
        W[:,2] = (rho0*cs0**2/GAMMA) * (W[:,0]/rho0)**GAMMA
    
        #dust
        W[:,3] = W[:,0]
        W[:,4] = W[:,1]
    
    else:
        sol = DustyWaveSolver(K=K, delta=drho, feedback=FB)(t)
        
        x = xc - v0*t
        W = np.full([len(x), NHYDRO], np.nan)
        
        W[:,0] = sol.rho_gas(x)
        W[:,1] = v0 + sol.v_gas(x)
        W[:,2] = (rho0*cs0**2/GAMMA) * (W[:,0]/rho0)**GAMMA
        
        W[:,3] = sol.rho_dust(x)
        W[:,4] = v0 + sol.v_dust(x)
    return W

def init_sod(xc, dust_gas_ratio = 1.0, gravity=0.0, GAMMA=5./3.):
    Pl = 1.0
    Pr = 0.1
    rhol = 1.0
    rhor = 0.125
    vl = 0
    vr = 0
    
    #Change once you've messed with boundaries
    idl = (xc < 0.5) 
    
    W = np.full([len(xc), NHYDRO], np.nan)
    
    W[idl, 0] = rhol
    W[idl, 1] = vl
    W[idl, 2] = Pl
    
    W[~idl, 0] = rhor
    W[~idl, 1] = vr
    W[~idl, 2] = Pr
    
    W[:,3] = W[:,0]
    W[:,4] = W[:,1]
    return(W)

def _test_sod(Nx=256, t_final=0.1, gravity=0.0):
    IC = init_sod
    
    f, subs = plt.subplots(5, 1)
    plt.suptitle("Dustyshock test")
    
    xL, WL = solve_euler(Nx, IC, t_final, Ca=0.4, 
                         mesh_type = "Lagrangian", fixed_v = 0.0, b_type="flow")
    xF, WF = solve_euler(Nx, IC, t_final, Ca=0.4, 
                         mesh_type= "Fixed", fixed_v=0.0, b_type="flow")
    xI, WI = solve_euler(Nx, IC, 0.0, Ca=0.4, 
                         mesh_type = "Fixed", fixed_v=0.0, b_type="flow")
    
    for i in range(0,5):
        subs[i].plot(xL, WL[:,i], c="b", label="Lagrangian")
        subs[i].plot(xF, WF[:,i], c="g", ls="--", label="Fixed")
        subs[i].plot(xI, WI[:,i], c="k", label="IC")
    
    subs[0].set_ylabel('Density')
    subs[1].set_ylabel('Velocity')
    subs[2].set_ylabel('Pressure')
    subs[3].set_ylabel('Dust Density')
    subs[4].set_ylabel('Dust Velocity')
    
    subs[4].set_xlabel('x')

    subs[0].legend(loc='best')
    
    """subs[0].set_xlim(0, 0.5)
    subs[1].set_xlim(0, 0.5)
    subs[2].set_xlim(0, 0.5)
    subs[3].set_xlim(0, 0.5)
    subs[4].set_xlim(0, 0.5)"""


def init_dustybox(xc, dust_gas_ratio = 1.0, gravity=0.0, GAMMA=5./3.):
    W = np.full([len(xc), NHYDRO], np.nan)
    P = 1.0
    rho_g = 1.0
    rho_d = rho_g*dust_gas_ratio 
    v_d = 1.0
    v_g = 0.0
    
    W[:,0] = rho_g
    W[:,1] = v_g
    W[:,2] = P
    W[:,3] = rho_d
    W[:,4] = v_d
    
    return(W)
    
def _test_dustybox_time(Nx=256, t_final= 1.0):
    IC = init_dustybox
    
    plt.figure()
    for ratio in [0.01, 0.1, 1., 10., 100.]:
        plt.title("Dustybox velocity over time")
        ts = []
        vLs = []
        vFs = []
        for t in np.linspace(0, t_final, 10):
            ts.append(t)
            xL, WL = solve_euler(Nx, IC, t, Ca=0.4,
                                 mesh_type = "Lagrangian", b_type = "periodic",
                                 dust_gas_ratio = ratio)
            """xF, WF = solve_euler(Nx, IC, t, Ca=0.4,
                                 mesh_type = "fixed", b_type = "periodic",
                                 dust_gas_ratio = ratio)"""
            vL = np.mean(WL[:,4])
            #vF = np.mean(WF[:,4])
            vLs.append(vL)
            #vFs.append(vF)
        plt.plot(ts, vLs, label="Lagrangian ratio=" + str(ratio))
        #plt.plot(ts, vFs, ls="--")
        plt.xlabel("t")
        plt.ylabel("Dust velocity")
        plt.xlim(-1e-5, 2)
        plt.ylim(-1e-5, 1)
        plt.legend()


def analytical_dustybox_feedback(t, dust_gas_ratio=1.0):
    P = 1.0
    rho_g = 1.0
    rho_d = rho_g*dust_gas_ratio
    rho = rho_g + rho_d
    ed = rho_d / rho
    eg = rho_g / rho 
    exp = np.exp(-K*rho*t)
    
    v_d = 1.0
    v_g = 0.0
    p_d = v_d*rho_d
    p_g = v_g*rho_g
    
    p_dust_out = p_g*(ed-ed*exp) + p_d*(ed+eg*exp)
    v_dust_out = p_dust_out / rho_d
    print(v_dust_out)
    return(v_dust_out)

def _test_dustybox_convergence(pmin = 4, pmax=10, t_final=1.0):
    IC = init_dustybox
    N = 2**np.arange(pmin, pmax+1)
    
    plt.figure()
    for ratio in [0.01, 0.1, 1., 10.]:
        true = analytical_dustybox_feedback(t_final, ratio)
        plt.title("Dustybox velocity error convergence")
        vLs = []
        vFs = []
        for Ni in N:
            xL, WL = solve_euler(Ni, IC, t_final, Ca=0.4,
                                 mesh_type = "Lagrangian", b_type = "periodic",
                                 dust_gas_ratio = ratio)
            vL = true - np.mean(WL[:,4])
            if vL == 0:
                vL = 1e-20
            vLs.append(vL)
            print(vL)
        plt.plot(N, vLs, label="Lagrangian ratio=" + str(ratio))
        plt.xlabel("N")
        plt.ylabel("Dust velocity error")
        plt.legend()
    plt.yscale("log")
    plt.xscale("log")


def init_const_gravity(xc, dust_gas_ratio=1.0, gravity=-1.0, GAMMA=5./3.):
    W = np.full([len(xc), NHYDRO], np.nan)
    H = (gravity*GAMMA)
    P_0 = 1.0/GAMMA
    rho_0 = P_0*GAMMA
    
    rho_g = rho_0*np.exp(xc*H)
    P = P_0*np.exp(xc*H)
    v_g=0
    v_d=0
    #insert rho dust in as a constant, since gas is in equilibrium already.
    rho_d = rho_0*np.exp(xc*H)[-1]*dust_gas_ratio
    
    W[:,0] = rho_g
    W[:,1] = v_g
    W[:,2] = P
    W[:,3] = rho_d
    W[:,4] = v_d
    
    return(W)


def _test_const_gravity(t_final=1.0, Nx=256, gravity=-1.0, dust_gas_ratio=1.0, Ca=0.4, GAMMA=5./3.):
    IC = init_const_gravity
    f, subs = plt.subplots(5, 1)
    
    xI, WI = solve_euler(Nx, IC, 0, Ca=Ca,
                         mesh_type="fixed", b_type = "reflecting",
                         dust_gas_ratio= dust_gas_ratio, gravity = gravity)
    
    for t in [2.0, 3.0, 5.0, 10.0]:
        x, W = solve_euler(Nx, IC, t, Ca=Ca, 
                           mesh_type = "fixed", b_type = "reflecting",
                           dust_gas_ratio = dust_gas_ratio, gravity=gravity)
        for i in range(0,5):
            subs[i].plot(x, W[:,i], label=str(t))
        
        #   Compute terminal velocity of dust
        H = (gravity*GAMMA)
        x_true, v_true = dust_settling_sol(H, v_0=W[-1,4])
        #subs[4].plot(xI, v_terminal, c = "k", ls="--", label="terminal velocity")
        subs[4].plot(x_true, v_true, c="k", ls="--", label="analytical solution")
    
    #   Plot IC 
    #for i in range(0,5):
    #    subs[i].plot(xI, WI[:,i], c="k", label="IC")
        
    
    subs[0].set_ylabel('Density')
    subs[1].set_ylabel('Velocity')
    subs[2].set_ylabel('Pressure')
    subs[3].set_ylabel('Dust Density')
    subs[4].set_ylabel('Dust Velocity')
    
    subs[4].set_xlabel('x')

    subs[0].legend(loc='best')


def init_dusty_shock_Jtype(xc, dust_gas_ratio = 1.0, gravity=0.0, GAMMA=1.0001):
    M = 2.0     
    
    P = 1.0
    rho = 1.0
    rho_d = dust_gas_ratio*rho
    
    c_s = np.sqrt(P/rho)
    v_s = c_s*M
    v_post = v_s/((1+dust_gas_ratio)*M**2)
    
    dv = v_s - v_post
    
    W = np.full([len(xc), NHYDRO], np.nan)
    
    W[:, 0] = rho
    W[:, 1] = dv
    W[:, 2] = P
    
    W[:,3] = rho_d
    W[:,4] = dv
    return(W)


def _test_dusty_shocks(t_final=1, Nx=256, Ca=0.4):
    plt.figure()
    linestyles = [":", "--", "-"]
    D = [0.01, 0.1, 1.0]
    v_s = 2.0 #since P=rho=1, c_s=1, thus v_s=Mach number
    for i in range(0,3):
        x, W = solve_euler(Nx, init_dusty_shock_Jtype, t_final, Ca=Ca,
                           mesh_type = "fixed", b_type = "flow", dust_reflect = True,
                           dust_gas_ratio = D[i], GAMMA=1.00001)
        plt.plot(x, W[:,1]/v_s, c="r", ls=linestyles[i], label="Gas; D=" + str(D[i]))
        plt.plot(x, W[:,4]/v_s, c="k", ls=linestyles[i], label="Dust; D=" + str(D[i]))
    plt.legend(loc="best")

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    """_test_convergence(init_wave, 
                      figs_evol=plt.subplots(3, 1)[1],
                      fig_err=plt.subplots(1)[1],
                      t_final = 3.0)
    """
    #_test_sod(t_final=0.2, Nx=569)
    
    #_test_dustybox_time(Nx=256, t_final= 2.0)
    
    #_test_dustybox_convergence(t_final=0.5)
    
    #_test_const_gravity()
    
    _test_dusty_shocks()
    
    plt.show()