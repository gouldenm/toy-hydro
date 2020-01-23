""" A toy model for hydro simulations, used to test things before trying out Arepo.
	Author: Maggie Celeste
	Date: 07/11/2019
	Added moving mesh as of 27/11/2019
	Added dust (with gas drag) as of 10/12/2019
	Added feedback of dust onto gas as of 17/1/2020
	Brought up to 2nd order as of 21/2020
	
	INSTRUCTIONS
	To run a hydro sim:
	1. Initialize a mesh instance
	2. Run mesh.setup(), with whatever conditions you like
	3. Run mesh.solve(), with scheme ("exp" or "approx") and tend (simulation end time -- float)
"""
import numpy as np
import matplotlib.pyplot as plt

class mesh:
	def __init__(self, 					
				 nx, xend,								#number of (non-ghost) cells spatial extent of grid
				 mesh_type = "Fixed", fixed_v = 0,		#type of mesh movement (options = "Fixed" or "Lagrangian"), velocity of fixed grid
				 K=0,									#constant in dust-gas coupling, dust:gast ratio
				 CFL=0.5, gamma=1.4):
		#define indices
		self.i_rho_g = 0
		self.i_p_g = 1
		self.i_E_g = 2
		self.i_rho_d = 3
		self.i_p_d = 4
		
		#	Establishing grid:
		self.nx, self.xend, self.CFL, self.gamma = nx, xend, CFL, gamma
		self.K = K
		self.mesh_type = mesh_type
		self.v = np.full(nx+2, fixed_v)								#velocity of cell centres -- defaults to 0 for Eulerian mesh
		self.vf = np.full(nx+1, fixed_v)							#velocity of cell faces -- defaults to same as cell centres
		self.dx = np.full(nx+2, self.xend/(self.nx) ) 				#size of cell -- initially uniform across all cells
		self.x = (np.arange(-0.5, nx+1))*(self.dx[0])				#position of mesh-generating point
		self.cm= (np.arange(-0.5, nx+1))*(self.dx[0])				#position of cell centre
		self.t = 0.0												#current time		
		
		#	Attributes that will be used later:
		self.boundary, self.IC = None, None							#Boundary type, initial condition type
		self.tend = None
		self.W = np.full((self.nx+2, 5), np.nan)					#primitive vector (lab frame)
		self.gradW = np.full((self.nx+2, 5), 0.0)						#primitive vector gradients (lab frame)
		self.Q = np.full((self.nx+2, 5), np.nan)					#Integrated conserved vector
		self.lm = np.full(nx+1,np.nan)								#Left signal velocity	
		self.lp = np.full(nx+1,np.nan)								#Right signal velocity
		self.ld = np.full(nx+1,np.nan)								#Dust signal velocity
	
	
	"""	Functions to convert from primitive vector to conserved vector and flux"""
	def prim2cons(self, W):
		U = np.full((len(W), 5), np.nan)					#conserved state vector
		U[:,0] = W[:,0]										#gas density
		U[:,1] = W[:,0]*W[:,1]								#gas momentum
		U[:,2] = W[:,2]/(self.gamma-1) + (W[:,0]*W[:,1]**2)/2	#gas energy
		U[:,3] = W[:,3]										#dust density
		U[:,4] = W[:,3]*W[:,4]								#dust momentum
		return(U)
	
	def cons2prim(self, U):
		W = np.full((len(U), 5), np.nan)					#primitive state vector
		W[:,0] = U[:,0]										#gas density
		W[:,1] = U[:,1]/U[:,0]								#gas velocity
		W[:,2] = (self.gamma-1)*(U[:,2] - (U[:,1]**2/U[:,0])/2)	#gas pressure
		W[:,3] = U[:,3]										#dust density
		W[:,4] = U[:,4]/U[:,3]								#dust velocity
		return(W)
	
	def prim2flux(self, W):
		F = np.full((len(W), 5), np.nan)		#conserved state vector
		F[:,0] = W[:,0]*W[:,1]													#gas mass flux
		F[:,1] = W[:,0]*W[:,1]**2 + W[:,2]										#gas momentum flux
		F[:,2] = W[:,1] * (W[:,2]/(self.gamma-1) + (W[:,0]*W[:,1]**2)/2 + W[:,2])	#gas energy flux
		F[:,3] = W[:,3]*W[:,4]													#dust mass flux
		F[:,4] = W[:,3]*W[:,4]**2												#dust momentum flux
		return(F)
	
	
	"""	Function to impose boundary conditions onto primitive state vectors"""
	def boundary_set(self, q):
		if self.boundary == "flow":
			q[0,:] = q[1,:]
			q[-1,:] = q[-2,:]
		elif self.boundary == "periodic":
			q[0,:] = q[-2,:]
			q[-1,:] = q[1,:]
		return(q)
	
	
	"""		Generate primitive vectors from Riemann-style Left/Right split		"""
	def get_W_LRsplit(self, cutoff, 				#cutoff tells where L/R boundary is
					  rhoL, PL, vL, rhoR, PR, vR,	#gas conditions on left, right
					  rhoLd, vLd, rhoRd, vRd):		#dust conditions on left, right
		for i in range(0, self.nx + 2):
			if self.x[i] <= cutoff*self.xend:
				self.W[i,0] = rhoL
				self.W[i,1] = vL
				self.W[i,2] = PL
				self.W[i,3] = rhoLd
				self.W[i,4] = vLd
			else:
				self.W[i,0] = rhoR
				self.W[i,1] = vR
				self.W[i,2] = PR
				self.W[i,3] = rhoRd
				self.W[i,4] = vRd
	
	
	"""		Generate primitive vectors from soundwave criteria		"""
	def get_W_soundwave(self, rhoB, drho,vB,		#background density, density wave amplitude, bulk velocity (all for gas)
						rhoBd, drhod, vBd,			#^^^^ for dust
						l, c_s):					#wavelength, sound speed, 
		C = c_s**2*rhoB**(1-self.gamma)/(self.gamma)  	#P = C*rho^gamma
		k = 2*np.pi/l
		for i in range(0, self.nx+2):
			#if self.x[i] <= l:
			self.W[i,0] = rhoB + drho*np.sin(k*self.x[i])
			self.W[i,1] = vB + c_s* (drho/rhoB)*np.sin(k*self.x[i])
			self.W[i,2] = C*self.W[i,0]**self.gamma
			self.W[i,3] = rhoBd + drhod*np.sin(k*self.x[i])
			self.W[i,4] = vBd + c_s* (drhod/rhoBd)*np.sin(k*self.x[i])
		
	"""		Set up Primitive vectors		"""
	def setup(self,
			  vB=0, rhoB=1.0, drho=1e-3, l=0.2, c_s=1.0, 					#gas variables for soundwave
			  vBd=0, rhoBd=1.0, drhod=1e-3, 								#dust variables for soundwave
			  vL=0, vR=0, rhoL=1.0, rhoR=0.1, PL=1.0, PR=0.125, cutoff=0.5, #gas variables for Riemann split
			  vLd=0, vRd=0, rhoLd=1.0, rhoRd=0.1,							#dust variables for Riemann split
			  IC="soundwave", boundary="periodic"):							#initial conditions type = "soundwave" or "LRsplit", boundary="periodic" or "flow"
		self.IC, self.boundary = IC, boundary
		if self.IC == "soundwave":
			self.get_W_soundwave(rhoB, drho, vB, rhoBd, drhod, vBd, l, c_s)
		elif self.IC == "LRsplit":
			self.get_W_LRsplit(cutoff, rhoL, PL, vL, rhoR, PR, vR, rhoLd, vLd, rhoRd, vRd)
			
		# Compute conserved quantities.
		U = self.prim2cons(self.W)
		self.Q = U * self.dx.reshape(-1,1)
		
	"""HLL Riemann Solver; note this also updates self.v, self.lp, and self.lm"""
	def riemann_solver(self, WL_in, WR_in, vf):
		fHLL = np.full((self.nx+1, 5), 0.)
		WL, WR = np.copy(WL_in), np.copy(WR_in)
		
		UL = self.prim2cons(WL)
		UR = self.prim2cons(WR)
		fL = self.prim2flux(WL)
		fR = self.prim2flux(WR)
		
		#	Calculate signal speeds for gas
		csl = np.sqrt(self.gamma*WL[:,2]/WL[:,0])
		csr = np.sqrt(self.gamma*WR[:,2]/WR[:,0])
		self.lm = WL[:,1] - csl
		self.lp = WR[:,1] + csr
		#	Calculate GAS flux in frame of face
		indexL = self.lm >= 0
		indexR = self.lp <= 0
		fHLL[:,:3] = ( self.lp.reshape(-1,1)*fL[:,:3] - self.lm.reshape(-1,1)*fR[:,:3] + self.lp.reshape(-1,1)*self.lm.reshape(-1,1)*(UR[:,:3] - UL[:,:3]) ) \
					 / (self.lp.reshape(-1,1)-self.lm.reshape(-1,1))
		fHLL[indexL,:3] = fL[indexL,:3]
		fHLL[indexR,:3] = fR[indexR,:3]
		
		#	Calculate signal speed for dust
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
		
		#	Calculate net flux in frame of LAB
		fF = np.copy(fHLL)
		fF[:,1] += fHLL[:,0]*vf
		fF[:,2] += 0.5*fHLL[:,0]*vf**2 + fHLL[:,1]*vf
		fF[:,4] += fHLL[:,3]*vf
		return(fF)
		
	
	"""	Calculate time step duration according to Courant condition"""
	def CFL_condition(self):
		dtm = self.CFL * self.dx[:-1] / np.absolute(self.lm)
		dtp = self.CFL * self.dx[1: ] / np.absolute(self.lp)
		dt = min(min(dtm), min(dtp))
		return(dt)
	
	
	"""	Move cells (i.e. change x coordinate),rearrange cells if any fall off the grid boundaries
		NB: doesn't work if grid cells exceed spatial extent on both sides """
	def update_mesh(self, dt):
		#  Modify x coordinate based on velocity of cell
		self.x += self.v * dt
		#  Update cell widths (so that each face is halfway between cell-generating points
		self.dx[1:-1] = (self.x[2:] - self.x[:-2])*0.5
		self.dx[0], self.dx[-1] = self.dx[1], self.dx[-2]
		#  Update positions of centre of mass
		self.cm[1:] = self.x[:-1] + 0.5*(self.x[1:]-self.x[:-1]) + self.dx[1:]*0.5
		self.cm[0] = self.x[0]
		
		
		#  Check if any (non-ghost) cells exeed the spatial extent of the grid in + x direction
		right_limit, left_limit = 0, 0
		i = 1
		while i < self.nx + 1:
			if self.x[i] > self.xend:
				right_limit = i
				break
			i+=1
		#  then work backwards to see if cells exceed spatial extent of the grid in -x direction...
		while i > 0:
			if self.x[i] < 0:
				left_limit = i
				break
			i-=1
		
		#  If any cells lie outside spatial extent in periodic regime, roll grid so they don't anymore
		if self.boundary == "periodic":
			if right_limit != 0:
				self.W[1:-1,:] = np.roll(self.W[1:-1,:], axis=0,shift = self.nx - right_limit)
				self.W = self.boundary_set(self.W)	#correct ghost cells
				self.x -= self.x[-2] - self.xend			#shift coordinates to match rolled grid
			
			elif left_limit != 0:
				self.W[1:-1,:] = np.roll(self.W[1:-1,:], axis=0,shift = -left_limit)
				self.W = self.boundary_set(self.W)
				self.x += (0 - self.x[1])
		"""
		elif self.boundary == "flow":
			if right_limit != 0:
				#Delete overhanging cells from right end by rolling + overwriting
				overhang = self.nx - right_limit
				self.W[1:-1,:] = np.roll(self.W[1:-1,:], axis=0,shift = overhang)
				self.W[: overhang, :] = self.W[overhang:overhang*2,:]
				self.W = self.boundary_set(self.W)	#correct ghost cells
				self.x -= self.x[-2] - self.xend			#shift coordinates to match rolled grid
			
			elif left_limit != 0:
				self.W[1:-1,:] = np.roll(self.W[1:-1,:], axis=0,shift = -left_limit)
				self.W[-left_limit: -1, :] = self.W[-left_limit*2:-left_limit,:]
				self.W = self.boundary_set(self.W)	#correct ghost cells
				self.x -= self.x[-2] - self.xend			#shift coordinates to match rolled grid
		"""
	
	def time_diff_W(self, W, gradW, FB):
		dWdt = np.zeros_like(W)
		rho_g = W[:, self.i_rho_g]
		grad_rho_g = gradW[:, self.i_rho_g]
		
		v_g = W[:, self.i_p_g]
		grad_v_g = gradW[:, self.i_p_g]
		
		P = W[:, self.i_E_g]
		grad_P = gradW[:, self.i_E_g]
		
		rho_d = W[:, self.i_rho_d]
		grad_rho_d = gradW[:, self.i_rho_d]
		
		v_d = W[:, self.i_p_d]
		grad_v_d = gradW[:, self.i_p_d]
		
		dWdt[:,self.i_rho_g] = v_g*grad_rho_g + rho_g*grad_v_g
		dWdt[:,self.i_p_g] = grad_P/rho_g + v_g*grad_v_g + FB*( - self.K*rho_d*v_d + self.K*rho_d*v_g)
		dWdt[:,self.i_E_g] = self.gamma*P*grad_v_g + v_g * grad_P
		dWdt[:,self.i_rho_d] =  v_d*grad_rho_d + rho_d*grad_v_d
		dWdt[:,self.i_p_d] = v_d*grad_v_d + self.K*rho_g*v_d - self.K*rho_g*v_g
		return(dWdt)
	
	"""	Function tying everything together into a hydro solver"""
	def solve(self, tend, scheme="exp",
		      early_stop = None, plotsep=None, timestep=None,   #early_stop = steps til stop; plotsep = steps between plots
		      feedback=False,
		      order2=False): 
		print("Solving... \n")
		self.tend = tend
		plotcount = 1
		if plotsep is not None:
			f, ax = plt.subplots(2,1)
			ax[0].plot(self.pos, self.rho_dust, color='k', label="rho_dust 0")
			ax[0].set_ylabel("Density")
			ax[0].plot(self.pos, self.rho_gas, linestyle="--", color='k', label="rho_gas 0")
			ax[1].plot(self.pos, self.v_dust, label="v_dust 0")
			ax[1].plot(self.pos, self.v_gas, label="v_gas 0")
			ax[1].set_ylabel("Velocity")
			#ax[1].scatter(self.x, self.W[:,4], color='k')
		
		while self.t < self.tend:
			# 0. A) Set feedback flag
			if feedback == True:
					FB = 1
			else:
				FB = 0
				
			# 0. B) Compute face velocity
			if self.mesh_type == "Lagrangian":
				self.v = np.copy(self.W[:,1])
				self.vf = (self.v[:-1] + self.v[1:])/2
			
			# 1.A) If second order, reconstruct primitive vector (i.e. compute slope-limited gradient in each cell, get WL, WR)
			if order2 == True:
				# Start by getting min / max W, to be used later...
				Wr = np.roll(self.W, axis=0, shift=-1)
				Wl = np.roll(self.W, axis=0, shift=1)
				maxW = np.maximum(np.maximum(Wr, self.W), Wl)
				minW = np.minimum(np.minimum(Wr, self.W), Wl)
				
				#  Compute least squares gradient by minimising difference between adjacent cell value and value got from extrapolating this cell's value w/ gradient
				#  First, compute separation between mesh-generating points of cell i and i+1 (ie to right of mesh point)
				dR = 0.5*(self.x[1:] - self.x[:-1])
				#  Then compute initial estimate of gradient:
				self.gradW[1:-1] = (self.W[1:-1] - self.W[0:-2])/dR[0:-1].reshape(-1,1) + (self.W[2:] - self.W[1:-1])/dR[1:].reshape(-1,1)
				
				# *** Slope limiting: Routine from Springel 2010
				# A) Compute change in prim variable from mesh-generating point to right face
				dWr = np.zeros_like(self.W)
				dWr[1:-1] = dR[1:].reshape(-1,1)*self.gradW[1:-1]
				
				# B) Get sign of change from mesh point -> right face
				index_gt0 = (dWr > 0)
				index_lt0 = (dWr < 0)
				index_eq0 = (dWr == 0)
				
				# C) Determine if W is a maximum / minimum with an inappropriate gradient...
				psir = np.zeros_like(dWr)
				psir[index_gt0] = (maxW[index_gt0] - self.W[index_gt0])/dWr[index_gt0]
				psir[index_lt0] = (minW[index_lt0] - self.W[index_lt0])/dWr[index_lt0]
				psir[index_eq0] = 1.
				
				# D) Repeat for left face
				dWl = np.zeros_like(self.W)
				dWl[1:-1] = -dR[0:-1].reshape(-1,1)*self.gradW[1:-1]
				
				index_gt0 = (dWl > 0)
				index_lt0 = (dWl < 0)
				index_eq0 = (dWl == 0)
				
				psil = np.zeros_like(dWl)
				psil[index_gt0] = (maxW[index_gt0] - self.W[index_gt0])/dWl[index_gt0]
				psil[index_lt0] = (minW[index_lt0] - self.W[index_lt0])/dWl[index_lt0]
				psil[index_eq0] = 1.
				
				# E) Now apply the slope limiting factor + correct boundaries
				alpha = np.minimum(np.ones_like(psir), np.minimum(psir, psil))
				self.gradW = alpha*self.gradW
				self.gradW = self.boundary_set(self.gradW)
				
				# F) Finally, get reconstructed WL, WR from gradients, to use in Riemann solver...
				WL = self.W[:-1] + self.gradW[:-1]*(self.x[1:]-self.x[:-1]).reshape(-1,1)*0.5
				WR = self.W[1:] - self.gradW[1:]*(self.x[1:]-self.x[:-1]).reshape(-1,1)*0.5
				
			#  1.B) If first order, just copy W for WL / WR:
			else:
				WL = np.copy(self.W[:-1])
				WR = np.copy(self.W[1:])
			
			# 2) Transform lab-frame to face frame (Note: gradients should be unaffected by frame change... At least according to Springel...
			WL[:,1] -= self.vf		# subtract face velocity from gas velocity
			WR[:,1] -= self.vf		
			WL[:,4] -= self.vf		# subtract face velocity from dust velocity
			WR[:,4] -= self.vf
			
			# 3. A) Compute flux at time t
			fF = self.riemann_solver(WL, WR, self.vf)
			
			# 3. B) Compute Courant condition + timestep
			if timestep is not None:
				dt = timestep
			
			else:
				dt = self.CFL_condition()
				dt = min(self.tend-self.t, dt)
			
			# 4. A) If second order, compute predicted flux at time t+dt
			if order2 == True:
				# *** Compute time derivatives of primitive variables (got from Euler equations) ***
				gradWL = self.gradW[:-1]
				gradWR = self.gradW[1:]
				
				dWdtL = self.time_diff_W(WL, gradWL, FB)
				dWdtR = self.time_diff_W(WR, gradWR, FB)
				
				WL_int, WR_int = np.copy(WL), np.copy(WR)
				
				# 3.A Compute intermediate primitive vector (i.e. predicted without transfer between cells)
				W_intL = WL + dt * dWdtL
				W_intR = WR + dt * dWdtR
				
				# 3.B Compute intermediate fluxes
				fF_int = self.riemann_solver(W_intL, W_intR, self.vf)
			
			# 4. B) If only first order, just replace intermediate terms with originals...
			else:
				fF_int = np.copy(fF)
				W_intL = np.copy(WL)
				W_intR = np.copy(WR)
			
			# 5) Update mesh.
			self.update_mesh(dt)
			
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
			
			if scheme == "implicit":
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
				print("not working yet")
			
			# 6) Save the updated primitive variables
			U = self.Q / self.dx.reshape(-1,1)
			self.W[1:-1] = self.cons2prim(U[1:-1])
			
			
			# 7) Compute edge states
			self.W = self.boundary_set(self.W)
			self.t+=dt	
			if plotsep is not None:
				if plotcount % plotsep == 0:
					ax[0].set_title("time="+str(self.t))
					ax[0].plot(self.pos, self.rho_dust, label="rho_dust")#, alpha=self.t/tend*0.5)
					ax[1].plot(self.pos, self.v_dust, label="v_dust")#, alpha=self.t/tend*0.5)
					#ax[0].scatter(self.pos, self.rho_dust, color="red", alpha=self.t/tend*0.5)
					#ax[1].scatter(self.pos, self.v_dust, color="red", alpha=self.t/tend*0.5)
					
					ax[0].plot(self.pos, self.rho_gas, linestyle="--", label="rho_gas")#, alpha=self.t/tend*0.5)
					ax[1].plot(self.pos, self.v_gas, "b", linestyle="--", label="v_gas")#, alpha=self.t/tend*0.5)
					#ax[0].scatter(self.pos, self.rho_gas, color="b", alpha=self.t/tend*0.5)
					#ax[1].scatter(self.pos, self.v_gas, color="b", alpha=self.t/tend*0.5)
					ax[0].grid()
					ax[0].legend()
					ax[1].legend()
					plt.pause(0.1)
					
			if early_stop:
				if early_stop == plotcount:
					print("Stopped simulation early at time t=", self.t+dt)
					break
					
			plotcount+=1
			
		
		
	@property
	def pos(self):
		return(self.x[1:-1])
	
	@property
	def pressure(self):
		return (self.W[1:-1,2])
	
	@property
	def rho_gas(self):
		return (self.W[1:-1,0])
	
	@property
	def rho_dust(self):
		return (self.W[1:-1,3])
	
	@property
	def v_gas (self):
		return (self.W[1:-1,1])
	
	@property
	def v_dust (self):
		return (self.W[1:-1,4])
	
	@property
	def energy(self):
		return (self.W[1:-1,2]/(gamma-1) + (self.W[1:-1,0]*self.W[1:-1,1]**2)/2)
	
	@property
	def time(self):
		return self.t

"""
eg = mesh(200, 1.0, mesh_type = "Lagrangian", K =1.0, CFL = 0.5)
eg.setup(IC="soundwave", boundary="periodic", vB=0, rhoB=1.0, drho=1e-3, l=1.0, c_s=1.0)
eg.solve(tend=1.0, scheme = "exp", order2=True)"""

