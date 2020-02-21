from __future__ import print_function
import numpy as np
from dustywave_sol2 import *
from dust_settling_sol import *
from dusty_shock import *

NHYDRO = 5
HLLC = False
plot_every_step = True

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
    F[:,2] = W[:,1] * (W[:,2]/(GAMMA-1) + (W[:,0]*W[:,1]**2)/2 + W[:,2]) \
             +  FB* W[:,4] * (W[:,3]*W[:,4]**2)/2.                 #gas energy flux + dust energy flux
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




def HLLC_solve(WLin, WRin, vf, GAMMA, FB):
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
    
    #   Compute signal speeds
    pR, pL = WR[:,2].reshape(-1,1), WL[:,2].reshape(-1,1)
    uR, uL = WR[:,1].reshape(-1,1), WL[:,1].reshape(-1,1)
    rhoR, rhoL = WR[:,0].reshape(-1,1), WL[:,0].reshape(-1,1)
    
    csl = np.sqrt(GAMMA*pL/rhoL)
    csr = np.sqrt(GAMMA*pR/rhoR)
    
    Sm = (uL - csl)
    Sp = (uR + csr)
    
    Sstar = ( pR - pL + rhoL*uL*(Sm-uL) - rhoR*uR*(Sp - uR) ) / \
            (rhoL*(Sm-uL) - rhoR*(Sp-uR))
    
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
    
    indexL =                                 0 <= Sm.flatten()
    indexLstar = (Sm.flatten()    <= 0 ) * ( 0 <= Sstar.flatten())
    indexRstar = (Sstar.flatten() <= 0 ) * ( 0 <= Sp.flatten())
    indexR =         Sp.flatten() <= 0
    
    fHLL[indexL,:3] = fL[indexL,:3]
    fHLL[indexLstar,:3] = f_starL[indexLstar,:3]
    fHLL[indexRstar,:3] = f_starR[indexRstar,:3]
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


def max_wave_speed(U, GAMMA, FB):
    W = cons2prim(U, GAMMA, FB)
    max_gas = np.max(np.abs( W[:,1]) + np.sqrt(GAMMA*W[:,2]/W[:,0]))
    max_dust = np.max(np.abs(W[:,4]))
    return max(max_gas, max_dust)





def solve_euler(Npts, IC, tout, Ca = 0.5, fixed_v = 0.0, mesh_type = "fixed", 
                dust_scheme = "explicit", b_type = "periodic", 
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
    # Setup up the grid
    reconstruction = Arepo2
    stencil = 2
    xe = np.linspace(0.0, xend, Npts+1)
    xc = 0.5*(xe[1:] + xe[:-1])
    
    shape = Npts + 2*stencil
    
    def boundary(Q, xc):
        #TODO: Fix xc_b in each case here (or, better, remove the cases outside...)
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
            #Create xc first
            xc_b = np.empty(shape+2) #2 additional boundary cells to make xe
            xc_b[(stencil+1):-(stencil+1)] = xc
            xc_b[ :(stencil+1)] = xc[-(stencil+1):] - 1
            xc_b[-(stencil+1):] = xc[ :(stencil+1)] + 1
    
            xe = 0.5*(xc_b[1:] + xc_b[:-1])
            xc_b = xc_b[1:-1]
            
            Qb = np.empty([shape, NHYDRO])
            Qb[stencil:-stencil] = Q
            #   inflow both gas and dust on left
            Qb[0] = Qb[2]
            Qb[1] = Qb[2]
            
            #   outflowflow dust on right
            Qb[-2] = Qb[-3]
            Qb[-1] = Qb[-3]
            #   reflect gas on right
            Qb[-1,:3] = Qb[-4,:3]
            Qb[-2,:3] = Qb[-3,:3]
            Qb[-1,1] = -Qb[-4,1]
            Qb[-2,1] = -Qb[-3,1]
        return Qb, xc_b, xe

    
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
    
    
    #########################################################################################
    # Set the initial conditions
    
    # Reconstruction function:
    R = reconstruction(xc, 0)
    
    W = IC(xc, dust_gas_ratio= dust_gas_ratio, 
           gravity=gravity, GAMMA=GAMMA, FB=FB, mach=mach)
    U = prim2cons(W, GAMMA, FB)
    dx = np.diff(xe).reshape(-1,1)
    Q = U * dx
    t = 0
    if plot_every_step:
        f, subs = plt.subplots(5, 1, sharex=True)
        subs[0].set_ylim(-1, 25)
        subs[0].set_ylabel('Density')
        subs[1].set_ylim(-1,3)
        subs[1].set_ylabel('Velocity')
        subs[2].set_ylabel('Energy')
        subs[3].set_ylabel('Dust Density')
        subs[4].set_ylabel('Dust velocity')
    
    
    while t < tout:
        print(t)
        # 0) Calculate new timestep
        dtmax = Ca * min(dx) / max_wave_speed(U, GAMMA, FB)
        dt = min(dtmax, tout-t)
        
        #1. Apply Boundaries
        Ub, xc_b, xe = boundary(U, xc)
        
        #2. Compute Primitive variables
        W = cons2prim(U, GAMMA, FB)
        Wb = cons2prim(Ub, GAMMA, FB)
        
        #3+4 Compute Gradients (and reconstruct the edge states)
        gradW = R.reconstruct(Wb, xc_b)
        
        Wm = Wb[1:-1] + gradW*(xe[1:-2] - xc_b[1:-1]).reshape(-1,1)
        Wp = Wb[1:-1] + gradW*(xe[2:-1] - xc_b[1:-1]).reshape(-1,1)
        WL = Wp[:-1]; WR = Wm[1:]
        
        #5. Set interface velocities
        if mesh_type == "Lagrangian":
            vc = Wb[:,1]
        else:
            vc = np.full_like(Wb[:,1], fixed_v)
        f = (xe[1:-1] - xc_b[:-1]) / (xc_b[1:] - xc_b[:-1])
        vf = f*vc[1:] + (1-f)*vc[:-1]
            
        #6. Compute first flux
        flux_0 =              HLL_solver(WL, WR, vf[1:-1], GAMMA, FB)
        F0 = dt*np.diff(flux_0, axis=0)
        
        # 7. Move the mesh
        xc_moved = xc + vc[stencil:-stencil]*dt
        Wb, xc_b_moved, xe_moved = boundary(W, xc_moved)
        dxold = np.diff(xe).reshape(-1,1)
        dx = np.diff(xe_moved).reshape(-1,1)
        
        # 8. Predict edge states W at t+td, starting with centre
        #8a. compute time diff
        WL = np.copy(Wb[1:-2]); WR = np.copy(Wb[2:-1])
        dWdt = time_diff_W(Wb[1:-1], gradW, vc[1:-1])
        
        #8b. predict cell centre, INCLUDING DRAG
        Ws = np.copy(Wb)[1:-1]
        rho_d = Ws[:, 3]
        rho_g = Ws[:, 0]
        Ws[:,1] += FB*K*rho_d*(Ws[:, 4] - Ws[:, 1])*dt
        Ws[:,4] -=    K*rho_g*(Ws[:, 4] - Ws[:, 1])*dt       #dust
        Ws += dWdt*dt
        
        #8c. Include constant gravity term, if applicable
        Ws[:,1] += gravity*dt #either 0.0 or 1.0
        Ws[:,4] += gravity*dt        
        
        #8d. Reconstruct the edge states    
        Wm = Ws + gradW*(xe_moved[1:-2] - xc_b_moved[1:-1]).reshape(-1,1)
        Wp = Ws + gradW*(xe_moved[2:-1] - xc_b_moved[1:-1]).reshape(-1,1)
        WL = Wp[:-1]; WR = Wm[1:]
        
        #9. Compute second flux
        flux_1 =                  HLL_solver(Wp[:-1], Wm[1:], vf[1:-1], GAMMA, FB)
        F1 = dt*np.diff(flux_1, axis=0)
        
        # 10. Time average fluxes (both used dt, so just *0.5)
        Qold = np.copy(Q)
        
        flux = - 0.5*(F1+F0)
        Q = Qold + flux
        
        #10a. Recompute Q for gas / dust momenta...
        dp_0 = K*dt*(Qold[:,0]*Qold[:,4] - Qold[:,3]*Qold[:,1])/dxold[2:-2].reshape(-1)
        dp_1 = K*dt*(Q[:,0]*Q[:,4] - Q[:,3]*Q[:,1])/dx[2:-2].reshape(-1)
        
        dp = 0.5*(dp_0 + dp_1)
        
        Q[:,4] -= dp
        Q[:,1] += FB*dp
        
        # Include const gravity term
        Q[:,4] += gravity*dt*Q[:,3]
        Q[:,1] += gravity*dt*Q[:,0]
        
        #11. Update U
        U = Q/dx[2:-2].reshape(-1,1)
        
        if plot_every_step:
            W = cons2prim(U, GAMMA, FB)
            for i in range(0,5):
                subs[i].plot(xc_b_moved[2:-2], W[:,i], label=str(t))
            plt.pause(2)
        
            print("Q,", Q[-10:,3])
            print("dxnew,", dx[-11:-1])
            print("W", W[-10:,3])
            print("\n")
        t = min(tout, t+dt)
        xc = xc_b_moved[stencil:-stencil]
        
    return xc, cons2prim(U, GAMMA, FB)




def init_dusty_shock_Jtype(xc, dust_gas_ratio = 1.0, gravity=0.0, GAMMA=1.0001, FB=1.0, mach=1.1):
    M = mach
    
    P = 1.0
    rho = 1.0
    rho_d = dust_gas_ratio*rho
    
    c_s = np.sqrt(P/rho)
    v_s = c_s*M
    v_post = v_s/((1+ FB*dust_gas_ratio)*M**2)
    
    dv = v_s - v_post
    
    W = np.full([len(xc), NHYDRO], np.nan)
    W[:, 0] = rho
    W[:, 1] = dv
    W[:, 2] = P
    W[:, 3] = rho_d
    W[:, 4] = dv
    return(W)


def init_dusty_shock_Ctype(xc, dust_gas_ratio = 1.0, gravity=0.0, GAMMA=1.0001, FB=1.0, mach=0.96):
    M = mach
    
    P = 1.0
    rho = 1.0
    rho_d = dust_gas_ratio*rho
    
    c_s = np.sqrt(P/rho)
    v_s = c_s*M
    v_post = v_s/((1+dust_gas_ratio)*M**2)
    
    dv = v_s - v_post
    
    W = np.full([len(xc), NHYDRO], np.nan)
    
    #Set one cell on left to shock velocity; all cells on right to post-shock velocity
    W[:, 0] = rho
    W[:, 2] = P
    W[:,3] = rho_d
   
    W[:, 1] = dv
    
    W[:,4] = dv
    return(W)



def _test_dusty_shocks_mach(t_final=5.0, Nx=500, Ca=0.2, FB = 0.0, K=0.):
    machs=[1.1]#machs = [ 2.0, 3, 4, 5]
    times = [1]#[1, 0.015, 0.007525, 0.006]
    for i in range(0, len(machs)):
        mach = machs[i]
        t_final = times[i]
        x, W = solve_euler(Nx, init_dusty_shock_Jtype, t_final, Ca=Ca,
                           mesh_type = "Lagrangian", b_type = "inflowL_and_reflectR",
                           dust_gas_ratio = 1.0, GAMMA=1.00001, xend=2.0, 
                           FB=FB, K=K, mach=mach)
        f, subs = plt.subplots(2, 1, sharex=True)
        subs[0].plot(x, W[:,1], c="r", label="Gas")
        #subs[0].plot(x, W[:,4], c="k", label="Dust")
        
        #plt.show()
        true = shock(mach, 1.0, {'drag_type':'power_law', 'drag_const':1.0}, 10., 1000., 
                     t=t_final, FB=FB, Kin=K, offset=10)
        #subs[0].plot(true["xi"], true["wd"], c="gray", ls="--", label="True Dust")
        subs[0].plot(true["xi"], true["wg"], c="pink", ls="--", label="True Gas" )
        
        subs[0].set_ylabel("v")
        subs[0].legend(loc="best")
        f.suptitle("J-type shock, t=" + str(t_final) + ", M=" + str(mach))
       
        subs[1].plot(x, W[:,0], c="r", label="Gas")
        #subs[1].plot(x, W[:,3], c="k", label="Dust")
       
        subs[1].plot(true["xi"], true["rhog"], c="pink", ls="--", label="True Gas")
        #subs[1].plot(true["xi"], true["rhod"], c="gray", ls="--", label="True Dust")
        subs[1].set_xlabel("pos")
        subs[1].set_ylabel("rho")
        #plt.legend(loc="best")


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
    
    #_test_dusty_shocks(t_final=6)
    
    for t in [2]:#[0.5, 0.55, 0.6, 0.8, 1.0, 2.0, 3.0, 3.6, 4]:
        _test_dusty_shocks_mach(t_final=t)
    plt.show()