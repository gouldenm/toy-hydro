""" A toy model for hydro simulations, used to test things before trying out Arepo.
	Author: Maggie Celeste
	Date: 07/11/2019
	UPDATE: Added moving mesh as of 27/11/2019
	
	To run a hydro sim:
	1. Initialize a mesh instance
	2. Run mesh.setup(), with whatever conditions you like
	3. Run mesh.solve()
"""
import numpy as np
import matplotlib.pyplot as plt


"""	Functions to convert from primitive vector to conserved vector and flux"""
def prim2cons(W, gamma):
	U = np.full((len(W), 5), np.nan)					#conserved state vector
	U[:,0] = W[:,0]										#gas density
	U[:,1] = W[:,0]*W[:,1]								#gas momentum
	U[:,2] = W[:,2]/(gamma-1) + (W[:,0]*W[:,1]**2)/2	#gas energy
	U[:,3] = W[:,3]										#dust density
	U[:,4] = W[:,3]*W[:,4]								#dust momentum
	return(U)

def cons2prim(U, gamma):
	W = np.full((len(U), 5), np.nan)					#primitive state vector
	W[:,0] = U[:,0]										#gas density
	W[:,1] = U[:,1]/U[:,0]								#gas velocity
	W[:,2] = (gamma-1)*(U[:,2] - (U[:,1]**2/U[:,0])/2)	#gas pressure
	W[:,3] = U[:,3]										#dust density
	W[:,4] = U[:,4]/U[:,3]								#dust velocity
	return(W)

def prim2flux(W, gamma):
	F = np.full((len(W), 5), np.nan)		#conserved state vector
	F[:,0] = W[:,0]*W[:,1]													#gas mass flux
	F[:,1] = W[:,0]*W[:,1]**2 + W[:,2]										#gas momentum flux
	F[:,2] = W[:,1] * (W[:,2]/(gamma-1) + (W[:,0]*W[:,1]**2)/2 + W[:,2])	#gas energy flux
	F[:,3] = W[:,3]*W[:,4]													#dust mass flux
	F[:,4] = W[:,3]*W[:,4]**2												#dust momentum flux
	return(F)


"""	Function to impose boundary conditions onto primitive state vectors"""
def boundary(q,boundary):
   if boundary == "flow":
       q[0,:] = q[1,:]
       q[-1,:] = q[-2,:]
   elif boundary == "periodic":
       q[0,:] = q[-2,:]
       q[-1,:] = q[1,:]
   return(q)
	

class mesh:
	def __init__(self, 					
				 nx, tend, xend,					#number of (non-ghost) cells, simulation cutoff time, spatial extent of grid
				 mesh_type = "Fixed", fixed_v = 0,	#type of mesh movement (options = "Fixed" or "Lagrangian"), velocity of fixed grid
				 K=0, ratio=1,						#constant in dust-gas coupling, dust:gast ratio
				 CFL=0.5, gamma=1.4):
		#	Establishing grid:
		self.nx, self.tend, self.xend, self.CFL, self.gamma = nx, tend, xend, CFL, gamma
		self.K, self.ratio = K, ratio								
		self.mesh_type = mesh_type
		self.v = np.full(nx+2, fixed_v)								#velocity of cell centres -- defaults to 0 for Eulerian mesh
		self.vf = np.full(nx+1, fixed_v)							#velocity of cell faces -- defaults to same as cell centres
		self.dx = np.full(nx+2, self.xend/(self.nx-1) ) 			#size of cell -- initially uniform across all cells
		self.x = np.arange(0, nx+2)*(self.dx[0])					#position of cell centre
		self.t = 0.0												#current time		
		
		#	Attributes that will be used later:
		self.boundary, self.IC = None, None							#Boundary type, initial condition type
		self.W = np.full((self.nx+2, 5), np.nan)					#primitive vector (lab frame)
		self.Q = np.full((self.nx+2, 5), np.nan)					#Integrated conserved vector
		#self.fF = np.full((nx+1, 5), np.nan)						#Net flux across face (lab frame)
		self.lm = np.full(nx+1,np.nan)								#Left signal velocity	
		self.lp = np.full(nx+1,np.nan)								#Right signal velocity
		self.ld = np.full(nx+1,np.nan)								#Dust signal velocity
	
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
			#else:
			#	self.W[i,0] = rhoB
			#	self.W[i,1] = vB
			#	self.W[i,2] = C*rhoB**self.gamma
			#	self.W[i,3] = rhoBd
			#	self.W[i,4] = vBd
		
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
		U = prim2cons(self.W, self.gamma)
		self.Q = U * self.dx.reshape(-1,1)
		
	"""HLL Riemann Solver; note this also updates self.v, self.lp, and self.lm"""
	def riemann_solver(self, WL, WR, vf):
		fHLL = np.full((self.nx+1, 5), 0.)
		
		#	Transform lab-frame to face frame.
		WL[:,1] -= vf		# subtract face velocity from gas velocity
		WR[:,1] -= vf		
		WL[:,4] -= vf		# subtract face velocity from dust velocity
		WR[:,4] -= vf
		UL = prim2cons(WL, self.gamma)
		UR = prim2cons(WR, self.gamma)
		fL = prim2flux(WL, self.gamma)
		fR = prim2flux(WR, self.gamma)
		
		#	Calculate signal speeds for gas
		csl = np.sqrt(self.gamma*WL[:,2]/WL[:,0])
		csr = np.sqrt(self.gamma*WR[:,2]/WR[:,0])
		self.lm = WL[:,1] - csl
		self.lp = WR[:,1] + csr
		#	Calculate signal speed for dust
		self.ld = (np.sqrt(WL[:,3])*WL[:,4] + np.sqrt(WR[:,3])*WR[:,4]) / (np.sqrt(WL[:,3]) + np.sqrt(WR[:,3]))
		#	Calculate GAS flux in frame of face
		indexL = self.lm >= 0
		indexR = self.lp <= 0
		fHLL[:,:3] = ( self.lp.reshape(-1,1)*fL[:,:3] - self.lm.reshape(-1,1)*fR[:,:3] + self.lp.reshape(-1,1)*self.lm.reshape(-1,1)*(UR[:,:3] - UL[:,:3]) ) \
					 / (self.lp.reshape(-1,1)-self.lm.reshape(-1,1))
		fHLL[indexL,:3] = fL[indexL,:3]
		fHLL[indexR,:3] = fR[indexR,:3]
		
		#	Calculate DUST flux in frame of face (note if vL < 0 < vR, then fHLL = 0.)
		indexL = self.ld > 0
		indexC = self.ld == 0
		indexR = self.ld < 0
		fHLL[indexL,3:] = fL[indexL,3:]
		fHLL[indexC,3:] = (fL[indexC,3:] + fR[indexC,3:])/2.
		fHLL[indexR,3:] = fR[indexR,3:]
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
		
		#hopefully we can cut this out later, but...
		#st = 1/(self.K * self.W[:,0] * self.W[:,3])
		dt = min(min(dtm), min(dtp))#, min(st))
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
				self.W = boundary(self.W, self.boundary)	#correct ghost cells
				self.x -= self.x[-2] - self.xend			#shift coordinates to match rolled grid
			
			elif left_limit != 0:
				self.W[1:-1,:] = np.roll(self.W[1:-1,:], axis=0,shift = -left_limit)
				self.W = boundary(self.W, self.boundary)
				self.x += (0 - self.x[1])
	
	
	"""	Function tying everything together into a hydro solver"""
	def solve(self):
		print("\n\n")
		plt.pause(0.1)
		plotcount = 1
		while self.t < self.tend:
			# 1) Compute primitive
			Uold = self.Q / self.dx.reshape(-1,1)
			self.W = cons2prim(Uold, self.gamma)
		
			# 2) Compute edge states
			self.W = boundary(self.W, self.boundary)
			
			# 3) Compute face velocity
			if self.mesh_type == "Lagrangian":
				self.v = np.copy(self.W[:,1])
				self.vf = (self.v[:-1] + self.v[1:])/2

			# 4) Compute fluxes
			WL, WR = np.copy(self.W)[:-1,:], np.copy(self.W)[1:,:]
			fF = self.riemann_solver(WL, WR, self.vf)
			Qold = Uold * self.dx.reshape(-1,1)		#nb use old dx here to get old Q
			Unew = np.copy(Uold) 
			
			# 5) Compute Courant condition
			dt = self.CFL_condition()
			dt = min(self.tend-self.t, dt)
			
			# 6) Update mesh.
			self.update_mesh(dt)
			
			# 7) First order time integration using Euler's method
			L = - np.diff(fF, axis=0)
			Unew[1:-1,:4] = (Qold[1:-1,:4] + L[:,:4]*dt) / self.dx[1:-1].reshape(-1,1)
			self.Q = Unew * self.dx.reshape(-1,1)	#nb use new dx here to get new Q
			# must include source term for the dust
			self.Q[1:-1,4] = (Qold[1:-1,4] + L[:,4]*dt + self.K*Uold[1:-1,3]*dt*self.Q[1:-1,1])/(1+self.K*Uold[1:-1,0]*dt)
			#self.Q[1:-1, 4] = (Qold[1:-1,4]*(
			
			if plotcount % 20000 == 0:
				break
				print(t)
				#plt.plot(self.x, self.W[:,0] , label="w="+str(self.v[0]) + ", t=" + str(t+dt))
			self.t+=dt	
			plotcount+=1
			
	@property
	def pressure(self):
		return (self.W[:,2])
	
	@property
	def density(self):
		return (self.W[:,0])
	
	@property
	def energy(self):
		return (self.W[:,2]/(gamma-1) + (self.W[:,0]*self.W[:,1]**2)/2)
		
		
"""t=0.3
grid = mesh(500, t, 1.0,  K=0)#,mesh_type="Lagrangian")
grid.setup(boundary = "flow", IC = "LRsplit", vLd=0.5, vRd=0.225)
#plt.plot(grid.x, grid.W[:,0] ,label="Initial W[v]")
grid.solve()
#plt.plot(grid.x, grid.W[:,0] , label="Gas density L")
plt.plot(grid.x, grid.W[:,0] , label="Gas density K=0" )
plt.plot(grid.x, grid.W[:,3] , label="Dust density, K=0")
plt.legend()
plt.pause(0.5)

grid = mesh(500, t, 1.0, K=20)
grid.setup(boundary = "flow", IC = "LRsplit", vLd=0.5, vRd=0.225)
grid.solve()
plt.plot(grid.x, grid.W[:,0] , label="Gas density K=2" )
plt.plot(grid.x, grid.W[:,3] , label="Dust density, K =2")
plt.legend()
plt.pause(0.5)


grid = mesh(500, t, 1.0, K=20)
grid.setup(boundary = "flow", IC = "LRsplit", vLd=0.5, vRd=0.225)
grid.solve()
plt.plot(grid.x, grid.W[:,0] , label="Gas density K=20" )
plt.plot(grid.x, grid.W[:,3] , label="Dust density, K =20")
plt.legend()
plt.pause(0.5)




grid = mesh(250, 0.2, 1.0,  fixed_v = 2.0)
grid.setup(boundary = "flow", IC = "LRsplit", vL = 2, vR = 2)
grid.solve()
plt.plot(grid.x-0.4, grid.W[:,0] , label="w="+str(grid.v[0]) )
plt.legend()
plt.pause(0.5)



grid = mesh(500, 0.2, 2.0,  fixed_v = -1)
grid.setup(boundary = "flow", IC = "LRsplit",cutoff=0.25)
grid.solve()
plt.plot(grid.x, grid.W[:,0] , label="w="+str(grid.v[0]) )
plt.legend()
plt.pause(0.5)


plt.xlabel("Position")
plt.ylabel("Pressure")
plt.pause(0.5)



#
#Relative motion = sound speed, should match initial conditions
"""
gridsound = mesh(300, 10.0, 1.0, mesh_type="Lagrangian", K=1.0)
gridsound.setup(drhod=1e-4, drho=1e-4, vB=0, vBd=0, l=1.0)
gridsound.solve()
f, ax = plt.subplots(2,1)
ax[0].plot(gridsound.x, gridsound.W[:,0], "k-",  label="Gas $\rho$")
ax[0].plot(gridsound.x, gridsound.W[:,3], "r-", label="Dust $\rho$")
ax[1].plot(gridsound.x, gridsound.W[:,1], "k-",  label="Gas $v$")
ax[1].plot(gridsound.x, gridsound.W[:,4], "r-", label="Dust $v$")


gridsound = mesh(300, 10.0, 1.0, mesh_type="Lagrangian", K=1.0)
gridsound.setup(drhod=1e-4, drho=1e-4, vB=0, vBd=0, l=1.0)
gridsound.solve()
plt.figure()
plt.plot(gridsound.x, gridsound.W[:,0], label="Gas, w=c_s", color="k")
plt.plot(gridsound.x, gridsound.W[:,3], label="dust t=0.05")
plt.show()

"""
#Relative motion = sound speed, should match initial conditions
gridsound = mesh(500, 0.05, 1.0, fixed_v = 1.0, K=100.0)
gridsound.setup(drhod=0, l=1.0)
gridsound.solve()
plt.plot(gridsound.x, gridsound.W[:,0], label="Gas, w=c_s", color="k")
plt.plot(gridsound.x, gridsound.W[:,3], label="dust t=0.05")
plt.legend()
plt.pause(1.0)

gridsound.tend = 0.1
gridsound.solve()
plt.plot(gridsound.x, gridsound.W[:,3], label="dust t=0.1")
plt.legend()
plt.pause(1.0)

gridsound.tend = 0.3
gridsound.solve()
plt.plot(gridsound.x, gridsound.W[:,3], label="dust t=0.3")
plt.legend()
plt.pause(1.0)

gridsound.tend = 0.5
gridsound.solve()
plt.plot(gridsound.x, gridsound.W[:,3], label="dust t=0.5")
plt.legend()
plt.pause(1.0)

gridsound.tend = 0.75
gridsound.solve()
plt.plot(gridsound.x, gridsound.W[:,3], label="dust t=0.75s")
plt.legend()
plt.pause(0.5)

gridsound.tend = 1.0
gridsound.solve()
plt.plot(gridsound.x, gridsound.W[:,3], label="dust t=1.0s")
plt.legend()
plt.pause(0.5)

gridsound.tend = 1.5
gridsound.solve()
plt.plot(gridsound.x, gridsound.W[:,3], label="dust t=1.5")
plt.legend()
plt.pause(0.5)

gridsound.tend = 2.0
gridsound.solve()
plt.plot(gridsound.x, gridsound.W[:,3], label="dust t=2.0")
plt.legend()
plt.pause(0.5)

gridsound.tend = 5.0
gridsound.solve()
plt.plot(gridsound.x, gridsound.W[:,3], label="dust t=5.0")
plt.legend()
plt.pause(0.5)

gridsound.tend = 10.0
gridsound.solve()
plt.plot(gridsound.x, gridsound.W[:,3], label="dust t=10.0")
plt.legend()
plt.pause(0.5)

gridsound.tend = 15.0
gridsound.solve()
plt.plot(gridsound.x, gridsound.W[:,3], label="dust t=15.0")
plt.legend()
plt.pause(0.5)
"""



"""
gridsound = mesh(500, t, 1.0, fixed_v = 0)
gridsound.setup(vB=-1)
gridsound.solve()
plt.plot(gridsound.x, gridsound.W[:,0], label="vB=-1, w="+str(gridsound.v[0]))
plt.legend()
plt.pause(0.5)

#Random faster speed for comparison
gridsound = mesh(500, t, 1.0, fixed_v = 2.0)
gridsound.setup()
gridsound.solve()
plt.plot(gridsound.x, gridsound.W[:,0], label="w="+str(gridsound.v[0]))
plt.legend()
plt.pause(0.5)



plt.figure()

gridsound = mesh(2000, 1.0, 1.0, fixed_v = 0.0)
gridsound.setup()
gridsound.solve()
plt.plot(gridsound.x, gridsound.W[:,2], label="w="+str(gridsound.v[0]))
plt.legend()
plt.pause(0.5)

gridsound = mesh(2000, 1.0, 1.0, fixed_v = 1.0)
gridsound.setup(vB=1.0)
gridsound.solve()
plt.plot(gridsound.x, gridsound.W[:,2], label="w="+str(gridsound.v[0]))
plt.legend()
plt.pause(0.5)

gridsound = mesh(2000, 1.0, 1.0, fixed_v = 1.0)
gridsound.setup()
gridsound.solve()
plt.plot(gridsound.x, gridsound.W[:,2], label="w="+str(gridsound.v[0]))
plt.legend()
plt.pause(0.5)

gridsound = mesh(2000, 1.0, 1.0, fixed_v = 2.0)
gridsound.setup()
gridsound.solve()
plt.plot(gridsound.x, gridsound.W[:,2], label="w="+str(gridsound.v[0]))
plt.legend()
plt.pause(0.5)

gridsound = mesh(2000, 1.0, 1.0, fixed_v = 2.0)
gridsound.setup()
gridsound.solve()
plt.plot(gridsound.x, gridsound.W[:,2], label="w="+str(gridsound.v[0]))
plt.legend()
plt.pause(0.5)
"""
#plt.show()

