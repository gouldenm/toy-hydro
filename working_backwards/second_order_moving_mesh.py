from __future__ import print_function
from arepo_reconstruction.py import reconstruction
import numpy as np

def prim2cons(W):
    U = np.full((len(W), NHYDRO), np.nan) #conserved state vector
    U[:,0] = W[:,0] #gas density
    U[:,1] = W[:,0]*W[:,1] #gas momentum
    U[:,2] = W[:,2]/(GAMMA-1) + (W[:,0]*W[:,1]**2)/2    #gas energy
    return(U)
    
def cons2prim(U):
    W = np.full((len(U), NHYDRO), np.nan) #primitive state vector
    W[:,0] = U[:,0] #gas density
    W[:,1] = U[:,1]/U[:,0] #gas velocity
    W[:,2] = (GAMMA-1)*(U[:,2] - (U[:,1]**2/U[:,0])/2)  #gas pressure
    return(W)
    
def prim2flux(W):
    F = np.full((len(W), NHYDRO), np.nan)
    F[:,0] = W[:,0]*W[:,1] #mass flux
    F[:,1] = W[:,0]*W[:,1]**2 + W[:,2] #momentum flux
    F[:,2] = W[:,1] * (W[:,2]/(GAMMA-1) + (W[:,0]*W[:,1]**2)/2 + W[:,2]) #energy flux
    return(F)


def HLL_solver(WL, WR, vf):
    # Transform lab-frame to face frame.
    WL[:,1] -= vf       # subtract face velocity from gas velocity
    WR[:,1] -= vf      
    
    UL = prim2cons(WL)
    UR = prim2cons(WR)
    
    fL = prim2flux(WL)
    fR = prim2flux(WR)
    
    csl = np.sqrt(GAMMA*WL[:,2]/WL[:,0])
    csr = np.sqrt(GAMMA*WR[:,2]/WR[:,0])
    
    Sm = (WL[:,1] - csl).reshape(-1,1)
    Sp = (WR[:,1] + csr).reshape(-1,1)

    
    # HLL central state in face frame
    fHLL = (Sp*fL - Sm*fR + Sp*Sm*(UR - UL)) / (Sp - Sm)

    # Left / Right states
    indexL = Sm.reshape(-1) >= 0
    indexR = Sp.reshape(-1) <= 0
    fHLL[indexL] = fL[indexL]
    fHLL[indexR] = fR[indexR]
    
    # Correct to lab frame
    fHLL_lab = np.copy(fHLL)
    fHLL_lab[:,1] += fHLL[:,0]*vf
    fHLL_lab[:,2] += 0.5*fHLL[:,0]*vf**2 + fHLL[:,1]*vf
        
    return fHLL_lab


""" Solve the equations of hydrodynamics for gas only on a moving grid """
def solve_euler(Npts, IC, reconstruction, tout, Ca = 0.7,
                fixed_v = 0.0, mesh_type = "fixed"):
    # Setup up the grid
    stencil = reconstruction.STENCIL
    order = reconstruction.ORDER
    
    shape = Npts + 2*stencil
    dx0 = 1. / Npts
    dx = np.full(Npts, dx0)
    xc = np.linspace(-dx0*stencil + dx0*0.5, 1+ dx0*stencil - dx0*0.5, shape)
    
    # Reconstruction function:
    R = reconstruction(xc, 0)
    
    def boundary(Q):
        Qb = np.empty([shape, NHYDRO])
        Qb[stencil:-stencil] = Q
        Qb[ :stencil] = Qb[Npts:Npts+stencil]
        Qb[-stencil:] = Qb[stencil:2*stencil]

        return Qb

    def update_stage(U, dt, xc):
        #1. Apply Boundaries
        Ub = boundary(U)

        #2. Compute Primitive variables
        Wb = cons2prim(Ub)
        
        #3. Correct velocities
        if mesh_type == "Lagrangian":
            vp = Wb[1:-2,1]
            vm = Wb[2:-1,1]
            vf =  0.5 * (vp+vm)
        elif mesh_type == "fixed":
            vf = fixed_v
        
        #4. Reconstruct the edge states
        Wp = np.full([U.shape[0]+2,NHYDRO], np.nan)
        Wm = np.full([U.shape[0]+2,NHYDRO], np.nan)
        gradW = np.full([U.shape[0]+2,NHYDRO], np.nan)
        for i in range(NHYDRO):
            Wm[:,i], Wp[:,i], gradW[:,i] = R.reconstruct(Wb[:,i], xc)
        
        #5. Compute fluxes
        flux =              HLL_solver(Wp[:-1], Wm[1:], vf)

        #6. Update Q
        return dt*np.diff(flux, axis=0)/dx.reshape(-1,1), gradW
    
    def update_stage_prim(W, dt, xc):
        #1. Apply Boundaries
        Wb = boundary(W)

        #2. Compute Conserved variables
        U = cons2prim(W)
        
        #3. Correct velocities
        if mesh_type == "Lagrangian":
            vp = Wb[1:-2,1]
            vm = Wb[2:-1,1]
            vf =  0.5 * (vp+vm)
        elif mesh_type == "fixed":
            vf = fixed_v
        
        #4. Reconstruct the edge states
        Wp = np.full([U.shape[0]+2,NHYDRO], np.nan)
        Wm = np.full([U.shape[0]+2,NHYDRO], np.nan)
        gradW = np.full([U.shape[0]+2,NHYDRO], np.nan)
        for i in range(NHYDRO):
            Wm[:,i], Wp[:,i], gradW[:,i] = R.reconstruct(Wb[:,i], xc)
        
        #5. Compute fluxes
        flux =                  HLL_solver(Wp[:-1], Wm[1:], vf)

        #6. Update Q
        return dt*np.diff(flux, axis=0)/dx.reshape(-1,1)

    
    def time_diff_W(W, gradW):# ###, FB):
        # TODO: Correct velocities
        dWdt = np.zeros_like(W)
        
        rho_g = W[:, 0]
        grad_rho_g = gradW[:, 0]
        
        v_g = W[:, 1]
        grad_v_g = gradW[:, 1]
        
        P = W[:, 2]
        grad_P = gradW[:, 2]
        
        """
        rho_d = W[:, self.i_rho_d]
        grad_rho_d = gradW[:, self.i_rho_d]
        
        v_d = W[:, self.i_p_d]
        grad_v_d = gradW[:, self.i_p_d]
        """
        dWdt[:,0] = v_g*grad_rho_g + rho_g*grad_v_g
        dWdt[:,1] = grad_P/rho_g + v_g*grad_v_g # ###+ FB*( - self.K*rho_d*v_d + self.K*rho_d*v_g)
        dWdt[:,2] = GAMMA*P*grad_v_g + v_g * grad_P
        """dWdt[:,self.i_rho_d] =  v_d*grad_rho_d + rho_d*grad_v_d
        dWdt[:,self.i_p_d] = v_d*grad_v_d + self.K*rho_g*v_d - self.K*rho_g*v_g"""
        dWdt *= -1
        return(dWdt)
    
    def dt_max_Ca(U):
        W = cons2prim(U)
        if mesh_type == "moving":
            wavespeed = np.abs(np.sqrt(GAMMA*W[:,2]/W[:,0]))
        else:
            wavespeed = np.abs((W[:,1])-fixed_v) + np.sqrt(GAMMA*W[:,2]/W[:,0])
        return np.max(Ca * dx / wavespeed)
    
    def update_mesh(xc, dt, W):
    #  Modify x coordinate based on velocity of cell centre
        if mesh_type == "Lagrangian":
            Wb = boundary(W)
            xc = xc + Wb[:,1]*dt
        else:
            xc = xc + fixed_v*dt
        dx[1:-1] = (xc_ng[3:-1] - xc_ng[1:-3])*0.5
        dx[0], dx[-1] = dx[1], dx[-2]
        return(xc, dx)
    
    
    # Set the initial conditions
    W = IC(xc[stencil:-stencil])
    U = prim2cons(W)

    t = 0
    while t < tout:
        # 1) Find new timestep
        dtmax = dt_max_Ca(U)
        dt = min(dtmax, tout-t)
        if order == 2:
            # 2) Calculate gradient, 
            # 3.) compute face velocity (in HLL solver), 
            # 4.) return flux-updated U
            F1, gradW1 =      update_stage(U , dt, xc)
            U1 = U - F1
            
            # 5.) TODO: Update mesh
            xc, dx = update_mesh(xc, dt, W)
            
            # 6) Compute predicted prim vars
            W = cons2prim(U)
            dWdt = time_diff_W(W, gradW1[1:-1])
            W1 = W + dWdt*dt
            
            # 7) Compute fluxes again
            Fp = update_stage_prim(W1, dt, xc)
            
            # 8) Time average (both used dt, so just *0.5)
            #U = U - 0.5*(F1+Fp)
            Fp, gradWp = update_stage(U1, dt, xc)
            Up = U1 - Fp
            U  = (U + Up)/2.
        else:
            F, grad  = update_stage(U, dt, xc)
            U = U - F
        
        t = min(tout, t+dt)

    xc = xc[stencil:-stencil]
    return xc, cons2prim(U)