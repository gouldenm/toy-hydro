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

class mesh:
	def __init__(self, 					
				 nx, xend,								#number of (non-ghost) cells spatial extent of grid
				 mesh_type = "Fixed", fixed_v = 0,		#type of mesh movement (options = "Fixed" or "Lagrangian"), velocity of fixed grid
				 K=0,									#constant in dust-gas coupling, dust:gast ratio
				 CFL=0.5, gamma=1.4):
		#define indices
		self.rho_g = 0
		self.v_g = 1
		self.P = 2
		self.rho_d = 3
		self.v_d = 4
		
		#	Establishing grid:
		self.nx, self.xend, self.CFL, self.gamma = nx, xend, CFL, gamma
		self.K = K
		self.mesh_type = mesh_type
		self.v = np.full(nx+2, fixed_v)								#velocity of cell centres -- defaults to 0 for Eulerian mesh
		self.vf = np.full(nx+1, fixed_v)							#velocity of cell faces -- defaults to same as cell centres
		self.dx = np.full(nx+2, self.xend/(self.nx) ) 			#size of cell -- initially uniform across all cells
		self.x = (np.arange(-0.5, nx+1))*(self.dx[0])					#position of cell centre
		self.t = 0.0												#current time		
		
		#	Attributes that will be used later:
		self.boundary, self.IC = None, None							#Boundary type, initial condition type
		self.tend = None
		self.W = np.full((self.nx+2, 5), np.nan)					#primitive vector (lab frame)
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
		#	Transform lab-frame to face frame.
		WL[:,1] -= vf		# subtract face velocity from gas velocity
		WR[:,1] -= vf		
		WL[:,4] -= vf		# subtract face velocity from dust velocity
		WR[:,4] -= vf
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

		#f_dust = fHLL[:,3:] 
		
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
		#  Modify x coordinate based on velocity of cell centre
		self.x += self.v * dt
		self.dx[1:-1] = (self.x[2:] - self.x[:-2])*0.5
		self.dx[0], self.dx[-1] = self.dx[1], self.dx[-2]
		
		
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
				#print("rolled left")
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
	
	
	
	
	"""	Function tying everything together into a hydro solver"""
	def solve(self, scheme, tend, 
		      early_stop = None, plotsep=None, timestep=None,   #early_stop = steps til stop; plotsep = steps between plots
		      feedback=True): 
		print("Solving... \n")
		self.tend = tend
		plotcount = 1
		if plotsep is not None:
			f, ax = plt.subplots(2,1)
			ax[0].plot(self.x, self.W[:,3], color='k')
			ax[1].plot(self.x, self.W[:,4], color='k')
			ax[0].scatter(self.x, self.W[:,3], color='k')
			ax[1].scatter(self.x, self.W[:,4], color='k')
		
		while self.t < self.tend:
			# 1) Compute face velocity
			if self.mesh_type == "Lagrangian":
				self.v = np.copy(self.W[:,1])
				self.vf = (self.v[:-1] + self.v[1:])/2

			# 2) Compute fluxes
			#WL = 0.5*(self.W[:-1] + self.W[1:])
			#WR  = WL.copy()
			WL, WR = np.copy(self.W)[:-1,:], np.copy(self.W)[1:,:]
			fF = self.riemann_solver(WL, WR, self.vf)
			Qold = np.copy(self.Q)		#nb use old dx here to get old Q
			Unew = np.full_like(Qold, np.nan) 
			
			# 3) Compute Courant condition
			if timestep is not None:
				dt = timestep
			
			else:
				dt = self.CFL_condition()
				dt = min(self.tend-self.t, dt)
			
			# 4) Update mesh.
			self.update_mesh(dt)
			
			# 5) First order time integration using Euler's method
			L = - np.diff(fF, axis=0)
			Unew[1:-1,:4] = (Qold[1:-1,:4] + L[:,:4]*dt) / self.dx[1:-1].reshape(-1,1)
			self.Q = Unew * self.dx.reshape(-1,1)	#nb use new dx here to get new Q
			if scheme == "approx":
				# must include source term for the dust
				self.Q[1:-1,4] = (Qold[1:-1,4] + L[:,4]*dt + self.K*Unew[1:-1,3]*dt*self.Q[1:-1,1])\
								/ (1+self.K*Unew[1:-1,0]*dt)
				#self.Q[1:-1, 4] = (Qold[1:-1,4]*(
			elif scheme == "exp":
				new_exp_term = np.exp(-self.K*Unew[1:-1,0]*dt)
				self.Q[1:-1,4] = Qold[1:-1,4]*new_exp_term \
								 + Unew[1:-1,3]/Unew[1:-1,0] * self.Q[1:-1,1] * (1 - new_exp_term)\
								 + (1-new_exp_term)/self.K * (L[:,4]/Unew[1:-1,0] - L[:,1]*Unew[1:-1,3]/Unew[1:-1,0]**2)\
								 + Unew[1:-1,3]/Unew[1:-1,0]*L[:,1]*dt*new_exp_term
			
			# 6) Save the updated primitive variables
			U = self.Q / self.dx.reshape(-1,1)
			self.W[1:-1] = self.cons2prim(U[1:-1])
			
			# 7) Compute edge states
			self.W = self.boundary_set(self.W)
			self.t+=dt	
			if plotsep is not None:
				if plotcount % plotsep == 0:
					ax[0].plot(self.x, self.W[:,3])
					ax[1].plot(self.x, self.W[:,4])
					ax[0].scatter(self.x, self.W[:,3])
					ax[1].scatter(self.x, self.W[:,4])
					ax[0].grid()
					plt.pause(0.5)
					
			if early_stop:
				if early_stop == plotcount:
					print("Stopped simulation early at time t=", self.t+dt)
					break
					
			plotcount+=1
			print(plotcount)
			
		
		
	@property
	def position(self):
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
eg.solve(tend=1.0, scheme = "approx", plotsep=100, early_stop = 500)
plt.show()
"""
