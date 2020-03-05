from __future__ import print_function
import numpy as np
from dusty_shock_adiabatic import *
import matplotlib.pyplot as plt

NHYDRO = 5
HLLC = True
plot_every_step = None#True

def reconstruct(Q, xc):
    dx = xc[2:] - xc[:-2]
    xe = 0.5*(xc[1:] + xc[:-1])
    """Reconstruct the left/right states"""
    Qm = Q[:-2]
    Q0 = Q[1:-1]
    Qp = Q[2:]
    limit = 2.0
    Qmax = np.maximum(np.maximum(Qp, Qm), Q0)
    Qmin = np.minimum(np.minimum(Qp, Qm), Q0)
    
    #Not the least squares estimate, but what is used in AREPO code release
    grad = (Qp - Qm) / dx.reshape(-1,1)
    dQ = limit*grad*(xe[1:] - xc[1:-1]).reshape(-1,1)
    Qp = Q0 + dQ

    pos = Qp > Qmax ; neg = Qp < Qmin
    phir = np.where(pos, (Qmax - Q0)/dQ, np.where(neg, (Qmin - Q0)/dQ, 1))
        
    dQ = limit*grad*(xe[0:-1] - xc[1:-1]).reshape(-1,1)
    Qm = Q0 + dQ

    pos = Qm > Qmax ; neg = Qm < Qmin
    phil = np.where(pos, (Qmax - Q0)/dQ, np.where(neg, (Qmin - Q0)/dQ, 1))

    alpha = np.maximum(0, np.minimum(1, np.minimum(phir, phil)))
    grad *= alpha
    
    return grad


def prim2cons(W, GAMMA, FB):
    U = np.full((len(W), NHYDRO), np.nan) #conserved state vector
    U[:,0] = W[:,0] #gas density
    U[:,1] = W[:,0]*W[:,1] #gas momentum
    U[:,2] = W[:,2]/(GAMMA-1) + (W[:,0]*W[:,1]**2)/2.    + FB*(W[:,3]*W[:,4]**2)/2.   #gas energy + dust KE
    U[:,3] = W[:,3]                                        #dust density
    U[:,4] = W[:,3]*W[:,4]                             #dust momentum
    return(U)
    
def cons2prim(U, GAMMA, FB):
    W = np.full((len(U), NHYDRO), np.nan) #primitive state vector
    W[:,0] = U[:,0] #gas density
    W[:,1] = U[:,1]/U[:,0] #gas velocity
    W[:,2] = (GAMMA-1)*(U[:,2] - (U[:,1]**2/U[:,0])/2.   - FB*(U[:,4]**2/U[:,3])/2. )  #gas pressure
    W[:,3] = U[:,3]                                     #dust density
    W[:,4] = U[:,4]/U[:,3]                             #dust velocity
    return(W)
    
def prim2flux(W, GAMMA, FB):
    F = np.full((len(W), NHYDRO), np.nan)
    F[:,0] = W[:,0]*W[:,1] #mass flux
    F[:,1] = W[:,0]*W[:,1]**2 + W[:,2] #momentum flux
    F[:,2] = W[:,1] * (W[:,2]/(GAMMA-1) + (W[:,0]*W[:,1]**2)/2 + W[:,2]) #\
             #+  FB* W[:,4] * (W[:,3]*W[:,4]**2)/2.                 #gas energy flux + dust energy flux
    F[:,3] = W[:,3]*W[:,4]                                                  #dust mass flux
    F[:,4] = W[:,3]*W[:,4]**2                                              #dust momentum flux
    return(F)


def HLL_solve(WLin, WRin, vf, GAMMA, FB):
    # transform lab frame to face frame
    WL = np.copy(WLin)
    WR = np.copy(WRin)
    WL[:,1] = WLin[:,1] - vf       # subtract face velocity from gas velocity
    WR[:,1] = WRin[:,1] - vf
    WL[:,4] = WLin[:,4] - vf
    WR[:,4] = WRin[:,4] - vf
        
    # RB: These conserved quantities will include the dust K.E in
    #     the total energy. We shouldn't include that in the HLL flux
    UL = prim2cons(WL, GAMMA, 0) # RB: Replace FB with 0 here
    UR = prim2cons(WR, GAMMA, 0) # RB: Replace FB with 0 here
    
    # RB: Again, use FB = 0 here so that the energy flux is just the
    #     gas energy.
    fL = prim2flux(WL, GAMMA, 0) # RB: Replace FB with 0 here
    fR = prim2flux(WR, GAMMA, 0) # RB: Replace FB with 0 here
    
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
    #   Calculate DUST flux in frame of face (note if vL < 0 < vR, then fHLL = 0.)
    f_dust = (WL[:,4] > 0).reshape(-1,1) * fL[:,3:] + (WR[:,4] < 0).reshape(-1,1) * fR[:,3:] 
    fHLL[:, 3:] = f_dust
    
    #   ### Compute change in energy due to dust KE flux... ###
    F_dust_energy_L = FB* (WL[:,3]*WL[:,4]**3)/2.
    F_dust_energy_R = FB* (WR[:,3]*WR[:,4]**3)/2.
    F_dust_energy = (WL[:,4] > 0) * F_dust_energy_L + (WR[:,4] < 0) * F_dust_energy_R
    
    fHLL[:,2] += F_dust_energy
    
    # Correct to lab frame
    fHLL_lab = np.copy(fHLL)
    fHLL_lab[:,1] += fHLL[:,0]*vf
    fHLL_lab[:,2] += 0.5*(fHLL[:,0] + FB*fHLL[:,3])*vf**2 + (fHLL[:,1]+FB*fHLL[:,4])*vf
    fHLL_lab[:,4] += fHLL[:,3]*vf
    
    return fHLL_lab




def HLLC_solve(WLin, WRin, vf, GAMMA, FB):
    # transform lab frame to face frame
    WL = np.copy(WLin)
    WR = np.copy(WRin)
    WL[:,1] = WLin[:,1] - vf       # subtract face velocity from gas velocity
    WR[:,1] = WRin[:,1] - vf
    WL[:,4] = WLin[:,4] - vf
    WR[:,4] = WRin[:,4] - vf
        
    UL = prim2cons(WL, GAMMA, 0) # RB: Replace FB with 0 here
    UR = prim2cons(WR, GAMMA, 0) # RB: Replace FB with 0 here
    
    fL = prim2flux(WL, GAMMA, 0) # RB: Replace FB with 0 here
    fR = prim2flux(WR, GAMMA, 0) # RB: Replace FB with 0 here
    
    #   Compute signal speeds
    pR, pL = WR[:,2].reshape(-1,1), WL[:,2].reshape(-1,1)
    uR, uL = WR[:,1].reshape(-1,1), WL[:,1].reshape(-1,1)
    rhoR, rhoL = WR[:,0].reshape(-1,1), WL[:,0].reshape(-1,1)
    
    csl = np.sqrt(GAMMA*pL/rhoL)
    csr = np.sqrt(GAMMA*pR/rhoR)
    
    Sm = (uL - csl)
    Sp = (uR + csr)
    
    Sstar = ( pR - pL + rhoL*uL*(Sm-uL) - rhoR*uR*(Sp - uR) ) / (rhoL*(Sm-uL) - rhoR*(Sp-uR))
    
    #   Compute star fluxes using single mean pressure in the star region (Toro 10.42, 10.44, 10.26)
    pLR = 0.5 * ( pL + pR + rhoL*(Sm - uL)*(Sstar - uL) + rhoR*(Sp - uR)*(Sstar - uR) )
    
    f_starL = np.zeros_like(fL)
    f_starL = Sstar*(Sm*UL - fL)
    f_starL[:,1] += Sm.flatten()*pLR.flatten()
    f_starL[:,2] += Sm.flatten()*Sstar.flatten()*pLR.flatten()
    f_starL = f_starL / (Sm - Sstar)
    
    f_starR = np.zeros_like(fR)
    f_starR = Sstar*(Sp*UR - fR)
    f_starR[:,1] += Sp.flatten()*pLR.flatten()
    f_starR[:,2] += Sp.flatten()*Sstar.flatten()*pLR.flatten()
    f_starR = f_starR / (Sp - Sstar)
    

    # Left / Right states
    fHLL = np.zeros_like(fL)
    
    # RB: Note that you set the cases with S == 0 twice #MG: Cheers, fixed it!
    indexL =                                 0 < Sm.flatten()
    indexLstar = (Sm.flatten()    <= 0 ) * ( 0 < Sstar.flatten())
    indexRstar = (Sstar.flatten() <= 0 ) * ( 0 < Sp.flatten())
    indexR =         Sp.flatten() <= 0
    
    fHLL[indexL,:3] = fL[indexL,:3]
    fHLL[indexLstar,:3] = f_starL[indexLstar,:3]
    fHLL[indexRstar,:3] = f_starR[indexRstar,:3]
    fHLL[indexR,:3] = fR[indexR,:3]
    
    # ### ### ### DUST ### ### ###
    #    Calculate signal speed for dust
    #   Calculate DUST flux in frame of face (note if vL < 0 < vR, then fHLL = 0.)
    f_dust = (WL[:,4] > 0).reshape(-1,1) * fL[:,3:] + (WR[:,4] < 0).reshape(-1,1) * fR[:,3:] 
    fHLL[:, 3:] = f_dust
    
    #   ### Compute change in energy due to dust KE flux... ###
    F_dust_energy_L = FB* (WL[:,3]*WL[:,4]**3)/2.
    F_dust_energy_R = FB* (WR[:,3]*WR[:,4]**3)/2.
    F_dust_energy = (WL[:,4] > 0) * F_dust_energy_L + (WR[:,4] < 0) * F_dust_energy_R
    
    fHLL[:,2] += F_dust_energy
    
    # Correct to lab frame
    fHLL_lab = np.copy(fHLL)
    fHLL_lab[:,1] += fHLL[:,0]*vf
    fHLL_lab[:,2] += 0.5*(fHLL[:,0] + FB*fHLL[:,3])*vf**2 + (fHLL[:,1]+FB*fHLL[:,4])*vf
    fHLL_lab[:,4] += fHLL[:,3]*vf
    
    return fHLL_lab


def max_wave_speed(U, GAMMA, FB):
    W = cons2prim(U, GAMMA, FB)
    max_gas = np.max(np.abs( W[:,1]) + np.sqrt(GAMMA*W[:,2]/W[:,0]))
    max_dust = np.max(np.abs(W[:,4]))
    return max(max_gas, max_dust)





def solve_euler(Npts, IC, boundary, tout, Ca = 0.5, fixed_v = 0.0, mesh_type = "fixed", 
                dust_scheme = "explicit",
                dust_gas_ratio = 1.0,
                gravity = 0.0,
                dust_reflect = False,  #ignore reflection of dust velocity unless set to True
                mach=1.0,
                GAMMA=5./3.,
                xend=1.0,
                FB = 1.0,
                K = 1.0): 
    if HLLC:
        HLL_solver = HLLC_solve
    else:
        HLL_solver = HLL_solve
    """Test schemes using an Explicit TVD RK integration"""
    stencil = 2
    shape = Npts + 2*stencil
    
    dx0 = xend / Npts
    xc = np.linspace(-dx0*stencil + dx0*0.5, xend+ dx0*stencil - dx0*0.5, shape)
    dx = (xc[2:] - xc[:-2])*0.5
    
    def time_diff_W(W, gradW, vf):
        dWdt = np.zeros_like(W)
        
        rho_g = W[:, 0]
        grad_rho_g = gradW[:, 0]
        
        v_g = W[:, 1] - vf
        grad_v_g = gradW[:, 1]
        
        P = W[:, 2]
        grad_P = gradW[:, 2]
        
        dWdt[:,0] = -v_g*grad_rho_g - rho_g*grad_v_g
        dWdt[:,1] = -grad_P/rho_g - v_g*grad_v_g 
        dWdt[:,2] = -GAMMA*P*grad_v_g - v_g * grad_P
        
        # ### Dust
        rho_d = W[:, 3]
        grad_rho_d = gradW[:, 3]
        
        v_d = W[:, 4] - vf
        grad_v_d = gradW[:, 4]
        
        dWdt[:,3] = - v_d*grad_rho_d - rho_d*grad_v_d
        dWdt[:,4] = -v_d*grad_v_d 
        return(dWdt)
    
    def update_mesh(xc, dt, vc):
    #  Modify x coordinate based on velocity of cell centre
        xc = xc + vc*dt
        dx = (xc[2:] - xc[:-2])*0.5
        return(xc, dx)
    
    
    #########################################################################################
    # Set the initial conditions
    W = IC(xc[stencil:-stencil], K, dust_gas_ratio= dust_gas_ratio, 
           gravity=gravity, GAMMA=GAMMA, FB=FB, mach=mach)
    U = prim2cons(W, GAMMA, FB)
    Q = U * dx[1:-1].reshape(-1,1)
    t = 0
    if plot_every_step:
        f, subs = plt.subplots(5, 1, sharex=True)
        subs[0].set_ylim(-1, 25)
        subs[0].set_ylabel('Density')
        subs[1].set_ylim(-1,5)
        subs[1].set_ylabel('Velocity')
        subs[2].set_ylabel('Energy')
        subs[3].set_ylim(0, 1.5)
        subs[3].set_ylabel('Dust Density')
        subs[4].set_ylabel('Dust velocity')
    
    while t < tout:
        print(t)
        # 0) Calculate new timestep
        dtmax = Ca * min(dx) / max_wave_speed(U, GAMMA, FB)
        dt = min(dtmax, tout-t)
        
        #1. Apply Boundaries
        Ub = boundary(U,shape)
        
        #2. Compute Primitive variables
        W = cons2prim(U, GAMMA, FB)
        Wb = cons2prim(Ub, GAMMA, FB)
        
        #3+4 Compute Gradients (and reconstruct the edge states)
        gradW = reconstruct(Wb, xc)
        
        xe = 0.5*(xc[1:] + xc[:-1])
        Wm = Wb[1:-1] + gradW*(xe[0:-1] - xc[1:-1]).reshape(-1,1)
        Wp = Wb[1:-1] + gradW*(xe[1:] - xc[1:-1]).reshape(-1,1)
        WL = Wp[:-1]; WR = Wm[1:]
        
        #5. Set interface velocities
        if mesh_type == "Lagrangian":
            vc = Wb[:,1]
        else:
            vc = np.full_like(Wb[:,1], fixed_v)
        xe = 0.5*(xc[1:] + xc[:-1])
        f = (xe - xc[:-1]) / (xc[1:] - xc[:-1])
        vf = f*vc[1:] + (1-f)*vc[:-1]
        
        #6. Compute first flux
        flux_0 =              HLL_solver(WL, WR, vf[1:-1], GAMMA, FB)
        F0 = dt*np.diff(flux_0, axis=0)
        
        # 7. Move the mesh
        dxold = np.copy(dx)
        xc, dx = update_mesh(xc, dt, vc)
        
        # 8. Predict edge states W at t+td, starting with centre
        #8a. compute time diff
        Wb = boundary(W,shape)[1:-1]#match gradient extent
        WL = np.copy(Wb[:-1]); WR = np.copy(Wb[1:])
        dWdt = time_diff_W(Wb, gradW, vc[1:-1])
        
        #8b. predict cell centre, INCLUDING DRAG
        Ws = boundary(W,shape)[1:-1]
        rho_d = Ws[:, 3]
        rho_g = Ws[:, 0]
        Ws[:,1] += FB*K*rho_d*(Ws[:, 4] - Ws[:, 1])*dt
        Ws[:,4] -=    K*rho_g*(Ws[:, 4] - Ws[:, 1])*dt       #dust
        Ws += dWdt*dt
        
        #8c. Include constant gravity term, if applicable
        Ws[:,1] += gravity*dt #either 0.0 or 1.0
        Ws[:,4] += gravity*dt        
        
        #8d. Reconstruct the edge states        
        xe = 0.5*(xc[1:] + xc[:-1])
        Wm = Ws + gradW*(xe[0:-1] - xc[1:-1]).reshape(-1,1)
        Wp = Ws + gradW*(xe[1:] - xc[1:-1]).reshape(-1,1)
        WL = Wp[:-1]; WR = Wm[1:]
        
        #9. Compute second flux
        flux_1 =                  HLL_solver(Wp[:-1], Wm[1:], vf[1:-1], GAMMA, FB)
        F1 = dt*np.diff(flux_1, axis=0)
        
        # 10. Time average fluxes (both used dt, so just *0.5)
        Qold = np.copy(Q)
        
        flux = - 0.5*(F1+F0)
        Q = Qold + flux
        
        Utemp = Q/dx[1:-1].reshape(-1,1)
        
        #10a. Recompute Q for gas / dust momenta...
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
        
        #11. Update U
        U = Q/dx[1:-1].reshape(-1,1)
        
        if plot_every_step:
            W = cons2prim(U, GAMMA, FB)
            for i in range(0,5):
                subs[i].plot(xc[2:-2], W[:,i], label=str(t))
            plt.pause(0.5)
        t = min(tout, t+dt)
    xc = xc[stencil:-stencil]
    return xc, cons2prim(U, GAMMA, FB)
