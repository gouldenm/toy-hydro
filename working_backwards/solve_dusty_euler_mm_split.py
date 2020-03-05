from __future__ import print_function
import numpy as np

from HLLC import HLLC
from dustywave_sol import DustyWaveSolver

GAMMA = 5/3.
NHYDRO = 5

i_rho_g = 0
i_vel_g = 1
i_pre_g = 2

i_rho_d = 3
i_vel_d = 4

K = 10.0

FB = 1

def prim2cons(W):
    U = np.full((len(W), NHYDRO), np.nan) #conserved state vector
    # Gas
    U[:,0] = W[:,0] # density
    U[:,1] = W[:,0]*W[:,1] # momentum
    U[:,2] = W[:,2]/(GAMMA-1) + (W[:,0]*W[:,1]**2)/2 # energy
    # Dust
    U[:,3] = W[:,3] # density
    U[:,4] = W[:,3]*W[:,4] # momentum
    return(U)
	
def cons2prim(U):
    W = np.full((len(U), NHYDRO), np.nan) #primitive state vector
    # Gas
    W[:,0] = U[:,0] # density
    W[:,1] = U[:,1]/U[:,0] # velocity
    W[:,2] = (GAMMA-1)*(U[:,2] - (U[:,1]**2/U[:,0])/2)	# pressure
    # Dust
    W[:,3] = U[:,3] # density
    W[:,4] = U[:,4]/U[:,3] # velocity
    return(W)
	
def prim2flux(W):
    F = np.full((len(W), NHYDRO), np.nan)
    # Gas
    F[:,0] = W[:,0]*W[:,1] # mass 
    F[:,1] = W[:,0]*W[:,1]**2 + W[:,2] # momentum 
    F[:,2] = W[:,1]*(W[:,2]/(GAMMA-1) + (W[:,0]*W[:,1]**2)/2 + W[:,2]) # energy
    # Dust
    F[:,3] = W[:,3]*W[:,4]  # mass 
    F[:,4] = W[:,3]*W[:,4]**2 # momentum 
    return(F)


def compute_gradients(xc, xe, Q):
    Qm = Q[:-2]
    Q0 = Q[1:-1]
    Qp = Q[2:]

    dx = xc[2:] - xc[:-2]

    Qmax = np.maximum(np.maximum(Qp, Qm), Q0)
    Qmin = np.minimum(np.minimum(Qp, Qm), Q0)

    grad = (Qp - Qm) / dx.reshape(-1,1)

    dQ = grad*(xe[2:-1] - xc[1:-1]).reshape(-1,1)
    Qp = Q0 + dQ

    pos = Qp > Qmax ; neg = Qp < Qmin
    with np.errstate(all='ignore'):
        phir = np.where(pos, (Qmax - Q0)/dQ, np.where(neg, (Qmin - Q0)/dQ, 1))
        
    dQ = grad*(xe[1:-2] - xc[1:-1]).reshape(-1,1)
    Qm = Q0 + dQ
    
    pos = Qm > Qmax ; neg = Qm < Qmin
    with np.errstate(all='ignore'):
        phil = np.where(pos, (Qmax - Q0)/dQ, np.where(neg, (Qmin - Q0)/dQ, 1))

    alpha = np.maximum(0, np.minimum(1, np.minimum(phir, phil)))
    grad *= alpha

    return grad


def compute_time_diff_W(W, gradW, vframe):
    
    dWdt = np.zeros_like(W)

    rho_g = W[:, i_rho_g]
    grad_rho_g = gradW[:, i_rho_g]
    
    v_g = W[:, i_vel_g] - vframe
    grad_v_g = gradW[:, i_vel_g]

    P = W[:, i_pre_g]
    grad_P = gradW[:, i_pre_g]

    dWdt[:,i_rho_g] = - v_g*grad_rho_g - rho_g  *grad_v_g
    dWdt[:,i_vel_g] =                  - v_g    *grad_v_g - (1/rho_g)*grad_P
    dWdt[:,i_pre_g] =                  - GAMMA*P*grad_v_g -   v_g    *grad_P

    rho_d = W[:, i_rho_d]
    grad_rho_d = gradW[:, i_rho_d]

    v_d = W[:, i_vel_d] - vframe
    grad_v_d = gradW[:, i_vel_d]

    dWdt[:,i_rho_d] = - v_d*grad_rho_d - rho_d  *grad_v_d
    dWdt[:,i_vel_d] =                  - v_d    *grad_v_d 

    # Drag forces
    #   Ignore in operator split formalism
    
    return dWdt

def wrapflux_moving_mesh(riemann):
    
    def compute_flux(WL, WR, vf):
        WL, WR = WL.copy(), WR.copy()

        WL[:,1] -= vf ; WR[:,1] -= vf
        WL[:,4] -= vf ; WR[:,4] -= vf

        flux = riemann(WL, WR)
        
        flux[:,2] += 0.5*flux[:,0]*vf**2 + flux[:,1]*vf
        flux[:,1] +=     flux[:,0]*vf

        flux[:,4] +=     flux[:,3]*vf

        return flux

    return compute_flux

def dust_solver(WL, WR):
    """Solve the Riemann Problem for dust"""

    # Dust signal speed: Roe-average
    R = np.sqrt(WL[:,3]/WR[:,3])
    f = R /(1 + R)

    Sd = f*WL[:,4] + (1-f)*WR[:,4]
    
    # Compute the conserved quantities
    UL = np.full([len(WL), 2], np.nan)
    UL[:,0] = WL[:,i_rho_d]
    UL[:,1] = WL[:,i_rho_d]*WL[:,i_vel_d]

    UR = np.full([len(WR), 2], np.nan)
    UR[:,0] = WR[:,i_rho_d]
    UR[:,1] = WR[:,i_rho_d]*WR[:,i_vel_d]

    # Upwind the advection
    Sd = Sd.reshape(-1,1)
    f_dust = Sd*np.where(Sd > 0, UL, UR)

    return f_dust

@wrapflux_moving_mesh
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

    # Overwrite the dust flux
    fHLL[:,3:] = dust_solver(WL, WR)
	      
    return fHLL

_HLLC = HLLC(gamma=GAMMA)
@wrapflux_moving_mesh
def HLLC_solver(WL, WR):
    UL, UR = prim2cons(WL), prim2cons(WR)
    flux_HLLC = _HLLC(UL[:,:3].T, UR[:,:3].T)

    flux = np.zeros_like(WL)
    flux[:,:3] = flux_HLLC.T
    flux[:,3:] = dust_solver(WL, WR)

    return flux


def max_wave_speed(U, vf):
    W = cons2prim(U)
    return np.maximum(np.abs(W[:,1]-vf) + np.sqrt(GAMMA*W[:,2]/W[:,0]), 
                      np.abs(W[:,4]-vf))

        
def solve_euler(Npts, IC, tout, Ca = 0.7, lagrangian=False, HLLC=True):
    """Test schemes using an Explicit TVD RK integration"""
    # Setup up the grid
    stencil = 2
         
    xe = np.linspace(0.0, 1.0, Npts+1)
    xc = 0.5*(xe[1:] + xe[:-1])

    def boundary(xc, Q):
        # Add periodic boundaries to Q
        Qb = np.empty([Npts+2*stencil, NHYDRO])
        Qb[stencil:-stencil] = Q
        Qb[ :stencil] = Qb[Npts:Npts+stencil]
        Qb[-stencil:] = Qb[stencil:2*stencil]

        # Add periodic boundaries for cell centres and compute interfaces
        xc_b = np.empty(Npts+2*(stencil+1))
        xc_b[(stencil+1):-(stencil+1)] = xc
        xc_b[ :(stencil+1)] = xc[-(stencil+1):] - 1
        xc_b[-(stencil+1):] = xc[ :(stencil+1)] + 1

        xe = 0.5*(xc_b[1:] + xc_b[:-1])
        xc_b = xc_b[1:-1]

        return xc_b, xe, Qb

    def apply_drag_forces_split(Q, dx, dt):
        """Operator split drag forces"""
        mg, md = Q[:,i_rho_g], Q[:,i_rho_d]

        m = mg + FB*md
        fac = np.expm1( -K * m * dt / dx)
        
        pg = Q[:,i_vel_g]
        pd = Q[:,i_vel_d]
        
        v_com = (pg + FB*pd) / m
        delta_pd = (mg*pd - md*pg) * fac / m

        Q[:,i_vel_d] += delta_pd
        if FB:
            Q[:,i_vel_g] -= delta_pd
            # Add change in dust K.E. to conserve total energy
            dE_k = delta_pd*(Q[:,i_vel_d] - 0.5*delta_pd) / Q[:,i_rho_d]
            Q[:,i_pre_g] -= delta_pd


    def RK2_prim(xc_in, Q, dt):

        # Initial drag step:
        dx = np.diff(boundary(xc_in, Q)[1][stencil:-stencil])
        apply_drag_forces_split(Q, dx, dt/2)

        #1. Apply Boundaries
        xc, xe, Qb = boundary(xc_in, Q)
        dx = np.diff(xe).reshape(-1, 1)

        #2. Compute Primitive variables
        Ub = Qb / dx
        Wb = cons2prim(Ub)

        #3. Compute gradients
        grad = compute_gradients(xc, xe, Wb)

        #4. Set interface velocities:
        if lagrangian:
            vc = Wb[:,1].copy()
        else:
            vc = np.zeros_like(Wb[:,1])
        f = (xe[1:-1] - xc[:-1]) / (xc[1:]-xc[:-1])
        vf = f*vc[1:] + (1-f)*vc[:-1]

        #5. Compute edge states:
        Wp = Wb[1:-1] + grad*(xe[2:-1] - xc[1:-1]).reshape(-1,1)
        Wm = Wb[1:-1] + grad*(xe[1:-2] - xc[1:-1]).reshape(-1,1)

        #6. Compute first fluxes:
        if HLLC:
            flux_0 = HLLC_solver(Wp[:-1], Wm[1:], vf[1:-1])
        else:
            flux_0 = HLL_solver(Wp[:-1], Wm[1:], vf[1:-1])

        #7. Move the mesh and compute new face locations:
        xc = xc_in + vc[stencil:-stencil]*dt
        xc, xe, _ = boundary(xc, Q)
        dx = np.diff(xe).reshape(-1, 1)

        #8. Predict edge states at t+dt
        dWdt = compute_time_diff_W(Wb[1:-1], grad, vc[1:-1])
        Wp = Wb[1:-1] + dt*dWdt + grad*(xe[2:-1] - xc[1:-1]).reshape(-1,1)
        Wm = Wb[1:-1] + dt*dWdt + grad*(xe[1:-2] - xc[1:-1]).reshape(-1,1)

        #9. Compute second fluxes
        if HLLC:
            flux_1 = HLLC_solver(Wp[:-1], Wm[1:], vf[1:-1])
        else:
            flux_1 = HLL_solver(Wp[:-1], Wm[1:], vf[1:-1])

        #10. Update Conserved quantities
        xc = xc[stencil:-stencil]
        xe = xe[stencil:-stencil]

        Q = Q - 0.5*dt*np.diff(flux_0 + flux_1, axis=0) 

        # Final drag step:
        apply_drag_forces_split(Q, np.diff(xe), dt/2)

        return xc, xe, Q

    # Set the initial conditions
    dx = np.diff(xe).reshape(-1,1)
    W = IC(xe)
    U = prim2cons(W)
    Q = U * dx

    t = 0
    while t < tout:
        vf = 0
        if lagrangian:
            vf = U[:,1] / U[:,0]
        dtmax = Ca * np.min(dx / max_wave_speed(U, vf))
        dt = min(dtmax, tout-t)

        xc, xe, Q = RK2_prim(xc, Q, dt)
        dx = np.diff(xe).reshape(-1,1)

        t = min(tout, t+dt)

    return xc, xe, cons2prim(Q/dx)
                
def _test_convergence(IC, pmin=3, pmax=9, t_final=3.0,
                      figs_evol=None, fig_err=None):

    N = 2**np.arange(pmin, pmax+1)
    for lagrangian in [True, False]:
        err_gas = []
        err_dust = []
        c=None
        if lagrangian:
            scheme = label='Moving'
            ls = '-'
        else:
            scheme = label='Fixed'
            ls = '--'
        print (label)
        for Ni in N:
            print ('\t', Ni)
            x, xe, W = solve_euler(Ni, IC, t_final, Ca = 0.4, 
                                    lagrangian=lagrangian)

            if figs_evol is not None:
                figs_evol[0].plot(x, W[:,0], ls=ls, label=str(Ni))
                figs_evol[1].plot(x, W[:,1], ls=ls)
                figs_evol[2].plot(x, W[:,2], ls=ls)
                figs_evol[3].plot(x, W[:,3], ls=ls)
                figs_evol[4].plot(x, W[:,4], ls=ls)
                
                figs_evol[0].set_ylabel('Density')
                figs_evol[1].set_ylabel('Velocity')
                figs_evol[2].set_ylabel('Pressure')
                figs_evol[3].set_ylabel('Dust Density')
                figs_evol[4].set_ylabel('Dust Velocity')

                figs_evol[2].set_xlabel('x')

                label=None
            
            err = W - IC(x, t=t_final)
            err_gas.append(np.sqrt(np.mean(err[:,1]**2)))
            err_dust.append(np.sqrt(np.mean(err[:,4]**2)))
        if fig_err is not None:
            c = fig_err.loglog(N, err_gas, c=c, ls='-', 
                               label=scheme)[0].get_color()
            fig_err.loglog(N, err_dust, c=c, ls='--')

    if fig_err is not None:
        fig_err.set_xlabel('N')
        fig_err.set_ylabel('L2 velocity error')
        fig_err.plot(N, 1e-4/N**2, label='1/N^2', c='k')
        fig_err.legend()
    if figs_evol is not None:
        W = IC(x, t=t_final)
        figs_evol[0].plot(x, W[:,0], ls=':', c='k', label='Exact')
        figs_evol[1].plot(x, W[:,1], ls=':', c='k')
        figs_evol[2].plot(x, W[:,2], ls=':', c='k')
        figs_evol[3].plot(x, W[:,3], ls=':', c='k')
        figs_evol[4].plot(x, W[:,4], ls=':', c='k')
        figs_evol[0].legend(loc='best',frameon=False, ncol=2)


def init_box(xe, v0=1, t=0):
    W = np.full([len(xe) - int(t==0), NHYDRO], np.nan)
    W[:,0] = 1.0
    W[:,2] = 0.6
    W[:,3] = 1.0

    v_com = 0.5 * FB + v0
    fac = np.expm1(-K*(1+FB)*t)    
    delta_pg =  -fac / (1+FB)

    W[:,1] = v0 + FB* delta_pg
    W[:,4] = v0 + 1 - delta_pg

    return W
    

def init_wave(xe, cs0=1.0, rho0=1.0, v0=1.0, drho=1e-6, t=0):

    if t == 0:
        kx = 2*np.pi*(xe + (v0-cs0)*t) 

        W = np.full([len(xe)-1, NHYDRO], np.nan)
        W[:,0] = rho0 - drho*np.diff(np.cos(kx)) / np.diff(kx)
        W[:,1] = v0 - drho*cs0*np.diff(np.cos(kx)) / np.diff(kx)
        W[:,2] = (rho0*cs0**2/GAMMA) * (W[:,0]/rho0)**GAMMA

        W[:,3] = W[:,0]
        W[:,4] = W[:,1]

    else:
        sol = DustyWaveSolver(K=K, delta=drho,vf=v0, 
                              GAMMA=GAMMA,feedback=FB)(t)

        x=xe
        W = np.full([len(x), NHYDRO], np.nan)

        W[:,0] = sol.rho_gas(x)
        W[:,1] = sol.v_gas(x)
        W[:,2] = sol.P(x)
        W[:,3] = sol.rho_dust(x)
        W[:,4] = sol.v_dust(x)

    return W


def _run_and_plot_sod(Nx=256, t_final=0.1):
    f, subs = plt.subplots(5, 1)
    
    IC = init_sod

    # Lagrangian
    for HLLC in [True, False]:
        if HLLC:
            c = 'b'
            C = 'C'
        else:
            c = 'g'
            C = ''
        x, xe, W = solve_euler(Nx, IC, t_final, Ca = 0.4, 
                               lagrangian=True, HLLC=HLLC)
        x -= 0.5
        

        subs[0].plot(x, W[:,0], c=c, label='Moving, HLL'+C)
        subs[1].plot(x, W[:,1], c=c, )
        subs[2].plot(x, W[:,2], c=c, )
        subs[3].plot(x, W[:,3], c=c, )
        subs[4].plot(x, W[:,4], c=c, )

        # Fixed
        x, xe, W = solve_euler(Nx, IC, t_final, Ca = 0.4, 
                               lagrangian=False, HLLC=HLLC)
        x -= 0.5

        subs[0].plot(x, W[:,0], c=c,  ls='--', label='Fixed, HLL'+C)
        subs[1].plot(x, W[:,1], c=c,  ls='--')
        subs[2].plot(x, W[:,2], c=c, ls='--')
        subs[3].plot(x, W[:,3], c=c, ls='--')
        subs[4].plot(x, W[:,4], c=c, ls='--')
    
    # Add IC:
    W = IC(xe)
    subs[0].plot(x, W[:,0], c='k', label='IC')
    subs[1].plot(x, W[:,1], c='k')
    subs[2].plot(x, W[:,2], c='k')
    subs[3].plot(x, W[:,3], c='k')
    subs[4].plot(x, W[:,4], c='k')
    

    subs[0].set_ylabel('Density')
    subs[1].set_ylabel('Velocity')
    subs[2].set_ylabel('Pressure')
    subs[3].set_ylabel('Dust Density')
    subs[4].set_ylabel('Dust Velocity')

    subs[4].set_xlabel('x')

    subs[0].legend(loc='best')

    subs[0].set_xlim(0, 0.5)
    subs[1].set_xlim(0, 0.5)
    subs[2].set_xlim(0, 0.5)
    subs[3].set_xlim(0, 0.5)
    subs[4].set_xlim(0, 0.5)

    subs[1].set_ylim(-0.1, 0.7)

def init_sod(xe):
    Pl = 1.0
    rhol = 1.0
    vl = 0.0

    Pr = 0.1975
    rhor = 0.25
    vr = 0

    xc = 0.5*(xe[1:] + xe[:-1])
    idx = (0.25 < xc) & (xc <= 0.75)
    
    W = np.full([len(xc), NHYDRO], np.nan)

    W[idx, 0] = rhol
    W[idx, 1] = vl
    W[idx, 2] = Pl

    W[~idx, 0] = rhor
    W[~idx, 1] = vr
    W[~idx, 2] = Pr

    W[:,3] = W[:,0]
    W[:,4] = W[:,1]

    return W

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    np.seterr(invalid='raise')
    _test_convergence(init_wave, 
                      figs_evol=plt.subplots(5, 1)[1],
                      fig_err=plt.subplots(1)[1])

    #_run_and_plot_sod()

    plt.show()
    
