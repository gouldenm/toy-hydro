from __future__ import print_function
import numpy as np

GAMMA = 5/3.
NHYDRO = 5

K = 0.1

FB = 0

from dustywave_sol import DustyWaveSolver

def compute_gradient(xc, xe, Q):
    dx = (xc[2:] - xc[:-2]).reshape(-1,1)
    """Reconstruct the left/right states"""
    Qm = Q[:-2]
    Q0 = Q[1:-1]
    Qp = Q[2:]
    
    Qmax = np.maximum(np.maximum(Qp, Qm), Q0)
    Qmin = np.minimum(np.minimum(Qp, Qm), Q0)
        
    #Not the least squares estimate, but what is used in AREPO code release
    grad = (Qp - Qm) / dx

    dQ = grad*(xe[2:-1] - xc[1:-1]).reshape(-1,1)
    Qp = Q0 + dQ

    pos = Qp > Qmax ; neg = Qp < Qmin
    phir = np.where(pos, (Qmax - Q0)/dQ, np.where(neg, (Qmin - Q0)/dQ, 1))
        
    dQ = grad*(xe[1:-2] - xc[1:-1]).reshape(-1,1)
    Qm = Q0 + dQ

    pos = Qm > Qmax ; neg = Qm < Qmin
    phil = np.where(pos, (Qmax - Q0)/dQ, np.where(neg, (Qmin - Q0)/dQ, 1))

    alpha = np.maximum(0, np.minimum(1, np.minimum(phir, phil)))
    grad *= alpha
        
    return grad


def prim2cons(W):
    U = np.full((len(W), NHYDRO), np.nan) #conserved state vector
    U[:,0] = W[:,0] #gas density
    U[:,1] = W[:,0]*W[:,1] #gas momentum
    U[:,2] = W[:,2]/(GAMMA-1) + (W[:,0]*W[:,1]**2)/2    #gas energy
    #   Dust
    U[:,3] = W[:,3]                                        #dust density
    U[:,4] = W[:,3]*W[:,4]                             #dust momentum
    return(U)
    
def cons2prim(U):
    W = np.full((len(U), NHYDRO), np.nan) #primitive state vector
    W[:,0] = U[:,0] #gas density
    W[:,1] = U[:,1]/U[:,0] #gas velocity
    W[:,2] = (GAMMA-1)*(U[:,2] - (U[:,1]**2/U[:,0])/2)  #gas pressure
    #   Dust
    W[:,3] = U[:,3]                                     #dust density
    W[:,4] = U[:,4]/U[:,3]                             #dust velocity
    return(W)
    
def prim2flux(W):
    F = np.full((len(W), NHYDRO), np.nan)
    F[:,0] = W[:,0]*W[:,1] #mass flux
    F[:,1] = W[:,0]*W[:,1]**2 + W[:,2] #momentum flux
    F[:,2] = W[:,1] * (W[:,2]/(GAMMA-1) + (W[:,0]*W[:,1]**2)/2 + W[:,2]) #energy flux
    #   Dust
    F[:,3] = W[:,3]*W[:,4]                                                  #dust mass flux
    F[:,4] = W[:,3]*W[:,4]**2                                              #dust momentum flux
    return(F)


def HLL_solver(WL, WR, vf):
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
    
     # Correct to lab frame
    fHLL_lab = np.copy(fHLL)
    fHLL_lab[:,1] += fHLL[:,0]*vf
    fHLL_lab[:,2] += 0.5*fHLL[:,0]*vf**2 + fHLL[:,1]*vf
    
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
    fHLL_lab[:,4] += fHLL[:,3]*vf
    
    return fHLL_lab



def max_wave_speed(U):
    W = cons2prim(U)
    return np.max(np.abs(W[:,1]) + np.sqrt(GAMMA*W[:,2]/W[:,0]))


def time_diff_W(W, gradW, vf):# ###, FB):
    dWdt = np.zeros_like(W)
    
    rho_g = W[:, 0]
    grad_rho_g = gradW[:, 0]
    
    v_g = W[:, 1] - vf
    grad_v_g = gradW[:, 1]
    
    P = W[:, 2]
    grad_P = gradW[:, 2]
    
    dWdt[:,0] = v_g*grad_rho_g + rho_g*grad_v_g
    dWdt[:,1] =                  v_g*grad_v_g        + (1/rho_g)*grad_P
    dWdt[:,2] =                  GAMMA*P*grad_v_g    + v_g * grad_P
    
    # Dust
    rho_d = W[:, 3]
    grad_rho_d = gradW[:, 3]
    
    v_d = W[:, 4] - vf
    grad_v_d = gradW[:, 4]
    
    dWdt[:,3] =  v_d*grad_rho_d + rho_d*grad_v_d
    dWdt[:,4] =                   v_d*grad_v_d #+ K*rho_g*v_d - K*rho_g*v_g
    
    dWdt *= -1
    
    return(dWdt)


def solve_euler(Npts, IC, tout, Ca = 0.7, fixed_v = 0.0, lagrangian = False, 
                dust_scheme = "explicit"):
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
    
    def update_stage_both(Q, dt, xcin):
        #1. APply boundaries
        xc, xe, Qb = boundary(xcin, Q)
        dx = np.diff(xe).reshape(-1, 1)
        
        #2. Compute prim variables
        Ub = Qb / dx
        Wb = cons2prim(Ub)
        
        #3. Compute gradients
        grad = compute_gradient(xc, xe, Wb)
        
        #4. Set interface velocities:
        if lagrangian:
            vc = Wb[:,1].copy()
        else:
            vc = np.full_like(Wb[:,1], fixed_v)
        f = (xe[1:-1] - xc[:-1]) / (xc[1:]-xc[:-1])
        vf = f*vc[1:] + (1-f)*vc[:-1]
        
        #5. Compute state at cell edges
        Wm = Wb[1:-1] + grad*(xe[2:-1] - xc[1:-1]).reshape(-1,1)
        Wp = Wb[1:-1] + grad*(xe[1:-2] - xc[1:-1]).reshape(-1,1)
        WL = Wp[:-1]; WR = Wm[1:]
        
        #6. Compute flux at t
        #6.a Transform lab-frame to face frame.
        #       Gas
        WL[:,1] -= vf[1:-1] #since Wm, Wp use [1:-1] as well     
        WR[:,1] -= vf[1:-1]
        #       Dust
        WL[:,4] -= vf[1:-1]
        WR[:,4] -= vf[1:-1]
        
        #6.b Use HLL solver
        F0 =                        HLL_solver(WL, WR, vf[1:-1]) #still takes in vf to correct output for frame change
        
        #7. Move the mesh + compute new face location
        xc = xcin + vc[stencil:-stencil]*dt
        xc, xe, _ = boundary(xc, Q)
        dx = np.diff(xe).reshape(-1, 1)
        
        #8. Predict edge states at t+dt
        #8a. first, compute  dWdt at centres (noting that we differentiate L/R bc of eq (35 - 36) in Springel
        # TODO: if things are breaking, it might be because you need to use centre velocities here instead -- though I'm not convinced...
        WLc = np.copy(Wb[1:-2]); WRc = np.copy(Wb[2:-1])
        dWdtL = time_diff_W(WLc, grad[:-1], vf[1:-1])
        dWdtR = time_diff_W(WRc, grad[1:], vf[1:-1])
        
        WLc += dWdtL*dt
        WRc += dWdtR*dt
        
        
        #8b. Apply drag forces using Exp Euler method (again, keep L/R separate even tho it's center cos they have different vf associated...)
        for i in [0,1]:
            Wc = [WLc, WRc][i]
            dWdt = [dWdtL, dWdtR][i]
            
            rho = Wc[:,0] + FB*Wc[:,3]
            
            v_com = (Wc[:,0]*Wc[:,1] + FB*Wc[:,3]*Wc[:,4])/rho
            dV = (Wb[1+i:-2+i,4] - Wb[1+i:-2+i,1]) * np.exp(-K*rho*dt) 
            da = (dWdt[:,4] - dWdt[:,1]) *-np.expm1(-dt*K*rho)/(K*rho)
            
            Wc[:,1] = v_com - FB*Wc[:,3]*(dV + da)/rho
            Wc[:,4] = v_com +    Wc[:,0]*(dV + da)/rho
        
        
        #8c. Reconstruct edge states (nb other than time change, I think these should be the same as in previous step?)
        WL = WL + dWdtL
        WR = WR + dWdtR
        
        
        #9. Compute flux at t+dt
        F1 =                       HLL_solver(WL, WR, vf[1:-1])
        
        
        #10.  Update conserved quantities (those that don't include drag term)
        Q[:,0] = Q[:,0] - 0.5*dt*np.diff(F0 + F1, axis=0)[:,0]
        Q[:,2] = Q[:,2] - 0.5*dt*np.diff(F0 + F1, axis=0)[:,2]
        Q[:,3] = Q[:,3] - 0.5*dt*np.diff(F0 + F1, axis=0)[:,3]
        
        #11. Compute drag terms using second order exp RK method
        dx = np.diff(xe[stencil:-stencil])
        rho_g = Q[:,0] / dx
        rho_d = Q[:,3] / dx
        rho = rho_g + FB*rho_d 
        
        eps_g = rho_g / rho 
        eps_d = rho_d / rho 
        
        f_g0 = -np.diff(F0[:,1]) ; f_g1 = -np.diff(F1[:,1])
        f_d0 = -np.diff(F0[:,4]) ; f_d1 = -np.diff(F1[:,4])

        m_com = Q[:,1]
        if FB:
            m_com += Q[:,4]

        df   = eps_g*f_d0 - eps_d*f_g0 
        dfdt = (eps_g*(f_d1-f_d0) - eps_d*(f_g1-f_g0)) / dt

        dV = (eps_g*Q[:,4] - eps_d*Q[:,1]) * np.exp(-K*rho*dt) 
        dV += (df - dfdt/(K*rho)) *-np.expm1(-dt*K*rho)/(K*rho)
        dV += dfdt*dt/(K*rho)
        
        m_d = eps_d * m_com + dV
        m_g = eps_g * m_com - dV*FB
        
        #12. Update conserved quantities that include drag term
        Q[:,1] = m_g
        Q[:,4] = m_d
        
        # Return:
        xc = xc[stencil:-stencil]
        xe = xe[stencil:-stencil]

        return xc, xe, Q
    
    
    # Set the initial conditions
    dx = np.diff(xe).reshape(-1,1)
    W = IC(xe)
    U = prim2cons(W)
    Q = U * dx

    t = 0
    while t < tout:
        # 1) Calculate new timestep
        dtmax = Ca * min(dx) / max_wave_speed(U)
        dt = min(dtmax, tout-t)
        
        xc, xe, Q = update_stage_both(Q, dt, xc)
        dx = np.diff(xe).reshape(-1,1)
        
        t = min(tout, t+dt)
        
    return xc, xe, cons2prim(U)




def _test_convergence(IC, pmin=3, pmax=9, tout=3.0,
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
            x, xe, W = solve_euler(Ni, IC, tout, Ca = 0.1, 
                                    lagrangian=lagrangian)

            if figs_evol is not None:
                figs_evol[0].plot(x, W[:,0]-1, ls=ls, label=str(Ni))
                figs_evol[1].plot(x, W[:,1], ls=ls)
                figs_evol[2].plot(x, W[:,2]-0.6, ls=ls)
                figs_evol[3].plot(x, W[:,3]-1, ls=ls)
                figs_evol[4].plot(x, W[:,4], ls=ls)
                
                figs_evol[0].set_ylabel('Density')
                figs_evol[1].set_ylabel('Velocity')
                figs_evol[2].set_ylabel('Pressure')
                figs_evol[3].set_ylabel('Dust Density')
                figs_evol[4].set_ylabel('Dust Velocity')

                figs_evol[2].set_xlabel('x')

                label=None
            
            err = W - IC(x, t=tout)
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
        W = IC(x, t=tout)
        figs_evol[0].plot(x, W[:,0]-1, ls=':', c='k', label='Exact')
        figs_evol[1].plot(x, W[:,1], ls=':', c='k')
        figs_evol[2].plot(x, W[:,2]-0.6, ls=':', c='k')
        figs_evol[3].plot(x, W[:,3]-1, ls=':', c='k')
        figs_evol[4].plot(x, W[:,4], ls=':', c='k')
        figs_evol[0].legend(loc='best',frameon=False, ncol=2)


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
        sol = DustyWaveSolver(K=K, delta=drho,feedback=FB)(t)

        x = xe - v0*t
        W = np.full([len(x), NHYDRO], np.nan)

        W[:,0] = sol.rho_gas(x)
        W[:,1] = v0 + sol.v_gas(x)
        W[:,2] = (rho0*cs0**2/GAMMA) * (W[:,0]/rho0)**GAMMA
        W[:,3] = sol.rho_dust(x)
        W[:,4] = v0 + sol.v_dust(x)

    return W


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    _test_convergence(init_wave, 
                      figs_evol=plt.subplots(5, 1)[1],
                      fig_err=plt.subplots(1)[1])

    plt.show()
    