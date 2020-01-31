""" A toy model for hydro simulations, used to test things before trying out Arepo.
    Author: Maggie Celeste
    Date: 07/11/2019
    Added moving mesh as of 27/11/2019
    Added dust (with gas drag) as of 10/12/2019
    
    INSTRUCTIONS
    To run a hydro sim:
    1. Initialize a mesh instance
    2. Run mesh.setup(), with whatever conditions you like
    3. Run mesh.solve(), with scheme ("exp" or "approx") and tend (simulation end time -- float)
"""
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate

class mesh:
    def __init__(self, 					
                Npts, IC, tout,						#number of (non-ghost) cells spatial extent of grid
                mesh_type = "fixed", fixed_v = 0,		#type of mesh movement (options = "Fixed" or "Lagrangian"), velocity of fixed grid
                K=0, NHYDRO=3, STENCIL = 2,								#constant in dust-gas coupling, dust:gast ratio
                CFL=0.2, gamma=1.4):
    
    """	Functions to convert from primitive vector to conserved vector and flux"""
    def prim2cons(self, W):
        U = np.full((len(W), NHYDRO), np.nan)					#conserved state vector
        U[:,0] = W[:,0]										#gas density
        U[:,1] = W[:,0]*W[:,1]								#gas momentum
        U[:,2] = W[:,2]/(self.gamma-1) + (W[:,0]*W[:,1]**2)/2	#gas energy
        #U[:,3] = W[:,3]										#dust density
        #U[:,4] = W[:,3]*W[:,4]								#dust momentum
        return(U)
    
    def cons2prim(self, U):
        W = np.full((len(U), NHYDRO), np.nan)					#primitive state vector
        W[:,0] = U[:,0]										#gas density
        W[:,1] = U[:,1]/U[:,0]								#gas velocity
        W[:,2] = (self.gamma-1)*(U[:,2] - (U[:,1]**2/U[:,0])/2)	#gas pressure
        #W[:,3] = U[:,3]										#dust density
        #W[:,4] = U[:,4]/U[:,3]								#dust velocity
        return(W)
    
    def prim2flux(self, W):
        F = np.full((len(W), NHYDRO), np.nan)		#conserved state vector
        F[:,0] = W[:,0]*W[:,1]													#gas mass flux
        F[:,1] = W[:,0]*W[:,1]**2 + W[:,2]										#gas momentum flux
        F[:,2] = W[:,1] * (W[:,2]/(self.gamma-1) + (W[:,0]*W[:,1]**2)/2 + W[:,2])	#gas energy flux
        #F[:,3] = W[:,3]*W[:,4]													#dust mass flux
        #F[:,4] = W[:,3]*W[:,4]**2												#dust momentum flux
        return(F)


    """	Function to impose boundary conditions onto primitive state vectors"""
    def boundary(Q):
        Qb = np.empty([shape, NHYDRO])
        Qb[stencil:-stencil] = Q
        Qb[ :stencil] = Qb[Npts:Npts+stencil]
        Qb[-stencil:] = Qb[stencil:2*stencil]

        return Qb
    
    def reconstruct(self, Q, xc):
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
    
        
    """HLL Riemann Solver; note this also updates self.v, self.lp, and self.lm"""
    def riemann_solver(self, WL, WR, vf):
        fHLL = np.full((Npts+STENCIL, NHYDRO), 0.)
        #	Transform lab-frame to face frame.
        WL[:,1] -= vf		# subtract face velocity from gas velocity
        WR[:,1] -= vf		
        #WL[:,4] -= vf		# subtract face velocity from dust velocity
        #WR[:,4] -= vf
        UL = self.prim2cons(WL)
        UR = self.prim2cons(WR)
        fL = self.prim2flux(WL)
        fR = self.prim2flux(WR)
        
        #	Calculate signal speeds for gas
        csl = np.sqrt(self.gamma*WL[:,2]/WL[:,0])
        csr = np.sqrt(self.gamma*WR[:,2]/WR[:,0])
        lm = (WL[:,1] - csl).reshape(-1,1)
        lp = (WR[:,1] + csr).reshape(-1,1)
        #	Calculate GAS flux in frame of face
        indexL = self.lm >= 0
        indexR = self.lp <= 0
        fHLL[:,:3] = ( lp*fL[:,:3] - lm.reshape*fR[:,:3] + lp*lm*(UR[:,:3] - UL[:,:3]) ) \
                    / (lp-lm)
        fHLL[indexL,:3] = fL[indexL,:3]
        fHLL[indexR,:3] = fR[indexR,:3]
        
        """#	Calculate signal speed for dust
        self.ld = (np.sqrt(WL[:,3])*WL[:,4] + np.sqrt(WR[:,3])*WR[:,4]) / (np.sqrt(WL[:,3]) + np.sqrt(WR[:,3]))
        #	Calculate DUST flux in frame of face (note if vL < 0 < vR, then fHLL = 0.)
        indexL = (self.ld > 1e-15) & np.logical_not(((WL[:,4] < 0) & (WR[:,4] > 0)))
        indexC = (np.abs(self.ld) < 1e-15 ) & np.logical_not(((WL[:,4] < 0) & (WR[:,4] > 0)))
        indexR = (self.ld < -1e-15) & np.logical_not(((WL[:,4] < 0) & (WR[:,4] > 0)))
        fHLL[indexL,3:] = fL[indexL,3:]
        fHLL[indexC,3:] = (fL[indexC,3:] + fR[indexC,3:])/2.
        fHLL[indexR,3:] = fR[indexR,3:]
        
        w_f = self.ld.reshape(-1,1)
        f_dust = w_f*np.where(w_f > 0, UL[:,3:], UR[:,3:]) 
        
        fHLL[:, 3:] = f_dust
        """
        #	Calculate net flux in frame of LAB
        fF = np.copy(fHLL)
        fF[:,1] += fHLL[:,0]*vf
        fF[:,2] += 0.5*fHLL[:,0]*vf**2 + fHLL[:,1]*vf
        #fF[:,4] += fHLL[:,3]*vf
        return(fF)
        
    
    """	Calculate time step duration according to Courant condition"""
    def CFL_condition(self, U, dx):
        W = cons2prim(U)
        if mesh_type == "moving":
            wavespeed = np.abs(np.sqrt(GAMMA*W[:,2]/W[:,0]))
        else:
            wavespeed = np.abs((W[:,1])-fixed_v) + np.sqrt(GAMMA*W[:,2]/W[:,0])
        return np.max(Ca * dx / wavespeed)
    
    """	Move cells (i.e. change x coordinate),rearrange cells if any fall off the grid boundaries
        NB: doesn't work if grid cells exceed spatial extent on both sides """
    def update_mesh(self, dt):
        #  Modify x coordinate based on velocity of cell centre
        self.x += self.v * dt
        self.dx[1:-1] = (self.x[2:] - self.x[:-2])*0.5
        self.dx[0], self.dx[-1] = self.dx[1], self.dx[-2]
        
    
    
    """	Function tying everything together into a hydro solver"""
    def solve(self, tend, scheme="exp",
            early_stop = None, plotsep=None, timestep=None,   #early_stop = steps til stop; plotsep = steps between plots
            feedback=False, order2=False): 
        print("Solving... \n")
        self.tend = tend
        plotcount = 1
        if feedback == False:
            FB = 0
        else:
            FB = 1.
        if plotsep is not None:
            f, ax = plt.subplots(2,1)
            #ax[0].plot(self.pos, self.rho_dust, color='k', label="rho_dust 0")
            ax[0].set_ylabel("Density")
            ax[0].plot(self.pos, self.rho_gas, linestyle="--", color='k', label="rho_gas 0")
            #ax[1].plot(self.pos, self.v_dust, label="v_dust 0")
            ax[1].plot(self.pos, self.v_gas, label="v_gas 0")
            ax[1].set_ylabel("Velocity")
            #ax[1].scatter(self.x, self.W[:,4], color='k')
        
        while self.t < self.tend:
            pos_save = self.x
            rho_gas_save = self.W[:,self.i_rho_g]
            # 0) Compute face velocity
            if self.mesh_type == "Lagrangian":
                self.v = np.copy(self.W[:,1])
                self.vf = (self.v[:-1] + self.v[1:])/2
            
            # 1.A) If second order, reconstruct primitive vector 
            #      (i.e. compute slope-limited gradient in each cell, get WL, WR)
            if order2 == True:
                # i) Get gradients
                gradW = self.spatial_diff_W(self.W, self.x)
                # ii) Get distance from cell -> face
                xp = np.roll(self.x, axis=0, shift=-1)
                dp = xp-self.x 
                # vi) Finally, get reconstructed WL, WR for use in Riemann solver
                WL = self.W[:-1] + gradW[:-1] * 0.5 * dp[:-1].reshape(-1,1)
                WR = self.W[1:] - gradW[1:] * 0.5 * dp[:-1].reshape(-1,1)
            
            # 1.B) If first order, just copy W for WL / WR:
            else:
                WL = np.copy(self.W[:-1])
                WR = np.copy(self.W[1:])
                
            # 2. Compute fluxes at time t
            fF = self.riemann_solver(WL, WR, self.vf)
            
            # 3. Compute Courant condition from fluxes
            if timestep is not None:
                dt = timestep
            else:
                dt = self.CFL_condition()
                dt = min(self.tend-self.t, dt)
            
            
            # 4) Update mesh.
            self.update_mesh(dt)
            
            # 5. A) If second order, compute predicted flux at time t+dt
            if order2 == True:
                # *** Compute time derivatives of primitive variables (got from Euler equations) ***				
                dWdt = self.time_diff_W(self.W, self.spatial_diff_W(self.W, self.x, limit=True), FB)
                
                W_int = self.W + dt * dWdt
                Q_int = self.Q.copy()
                Q_int[1:-1] -=  np.diff(fF, axis=0)*dt
                W_int2 = self.cons2prim(Q_int/self.dx.reshape(-1,1))
                W_int2 = self.boundary_set(W_int)
                gradW_int = self.spatial_diff_W(W_int, self.x)
                
                xp = np.roll(self.x, axis=0, shift=-1)
                dp = xp-self.x 
                
                WL_int = W_int[:-1] + gradW_int[:-1] * 0.5 * dp[:-1].reshape(-1,1)
                WR_int = W_int[1:] - gradW_int[1:] * 0.5 * dp[:-1].reshape(-1,1)
                
                
                # 3.B Compute intermediate fluxes
                fF_int = self.riemann_solver(WL_int, WR_int, self.vf)
                
            # 5. B) If only first order, just replace intermediate terms with originals...
            else:
                fF_int = np.copy(fF)
                W_intL = np.copy(WL)
                W_intR = np.copy(WR)
            
            
            # 6) Perform time integration
            Qold = np.copy(self.Q)		#nb use old dx here to get old Q
            Unew = np.full_like(Qold, np.nan) 
            
            L = - np.diff(fF, axis=0)
            L_int = - np.diff(fF_int, axis=0)
            
            p_g = Qold[1:-1, self.i_p_g]
            p_d = Qold[1:-1, self.i_p_d]
            
            # Average of first order source terms + second order source terms
            f_g = 0.5*(L[:,self.i_p_g] + L_int[:,self.i_p_g])
            f_d = 0.5*(L[:,self.i_p_d] + L_int[:,self.i_p_d])
            
            # Average gas density flux
            L_rho_g = 0.5*(L[:,self.i_rho_g] + L_int[:,self.i_rho_g])
            
            #  Gas density, Gas Energy density, dust density
            Unew[1:-1,self.i_rho_g] = (Qold[1:-1,self.i_rho_g] + L_rho_g*dt) / self.dx[1:-1]
            Unew[1:-1,2:4] = (Qold[1:-1,2:4] + 0.5*(L[:,2:4]+L_int[:,2:4])*dt) / self.dx[1:-1].reshape(-1,1)
            self.Q = Unew * self.dx.reshape(-1,1)
            
            #	Group terms for clarity
            rho_d = Unew[1:-1, self.i_rho_d]
            rho_g = Unew[1:-1, self.i_rho_g]
            rho = FB*rho_d + rho_g	
            eps_g = rho_g / rho
            eps_d = rho_d / rho
            
            if scheme == "exp":
                exp_term = np.exp(-self.K*rho*dt)
                #  Compute dust momentum
                self.Q[1:-1, self.i_p_d] = (eps_g*p_d - eps_d*p_g) * exp_term                        \
                                        + (eps_g*f_d - eps_d*f_g) * (1-exp_term) / (self.K*rho)   \
                                        + eps_d * (FB*p_d + p_g)                                   \
                                        + eps_d * (FB*f_d + f_g) * dt
                #  Compute gas momentum
                self.Q[1:-1, self.i_p_g] = (FB*f_d + f_g) * dt        \
                                        + (FB*p_d + p_g)           \
                                        - FB*self.Q[1:-1, self.i_p_d]
            elif scheme == "explicit":
                self.Q[1:-1,self.i_p_d] = (p_d + f_d*dt + self.K*rho_d*dt*p_g)\
                                        / (1+self.K*rho_g*dt)
                
                self.Q[1:-1, self.i_p_g] = (FB*f_d+f_g)*dt            \
                                        + (FB*p_d + p_g)           \
                                        - FB*self.Q[1:-1, self.i_p_d]
            
            # 7) Save the updated primitive variables
            U = self.Q / self.dx.reshape(-1,1)
            self.W[1:-1] = self.cons2prim(U[1:-1])
            
            
            # 8) Compute edge states
            self.W = self.boundary_set(self.W)
            self.t+=dt	
