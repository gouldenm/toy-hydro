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
	U = np.full((len(W), 3), np.nan)		#conserved state vector
	U[:,0] = W[:,0]
	U[:,1] = W[:,0]*W[:,1]
	U[:,2] = W[:,2]/(gamma-1) + (W[:,0]*W[:,1]**2)/2
	return(U)

def cons2prim(U, gamma):
	W = np.full((len(U), 3), np.nan)		#primitive state vector
	W[:,0] = U[:,0]
	W[:,1] = U[:,1]/U[:,0]
	W[:,2] = (gamma-1)*(U[:,2] - (U[:,1]**2/U[:,0])/2)
	return(W)

def prim2flux(W, gamma):
	F = np.full((len(W), 3), np.nan)		#conserved state vector
	F[:,0] = W[:,0]*W[:,1]
	F[:,1] = W[:,0]*W[:,1]**2 + W[:,2]
	F[:,2] = W[:,1] * (W[:,2]/(gamma-1) + (W[:,0]*W[:,1]**2)/2 + W[:,2])
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
				 CFL=0.5, gamma=1.4):
		#	Establishing grid:
		self.nx, self.tend, self.xend, self.CFL, self.gamma = nx, tend, xend, CFL, gamma
		self.mesh_type = mesh_type
		self.v = np.full(nx+2, fixed_v)								#velocity of cell centres -- defaults to 0 for Eulerian mesh
		self.vf = np.full(nx+1, fixed_v)							#velocity of cell faces -- defaults to same as cell centres
		self.dx = np.full(nx+2, self.xend/(self.nx-1) ) 	#size of cell -- initially uniform across all cells
		self.x = np.arange(0, nx+2)*(self.dx[0])			#position of cell centre
		
		#	Attributes that will be used later:
		self.boundary, self.IC = None, None							#Boundary type, initial condition type
		self.W = np.full((self.nx+2, 3), np.nan)					#primitive vector (lab frame)
		self.Q = np.full((self.nx+2, 3), np.nan)					#Integrated conserved vector
		self.fF = np.full((nx+1, 3), np.nan)						#Net flux across face (lab frame)
		self.lm = np.full(nx+1,np.nan)								#Left signal velocity	
		self.lp = np.full(nx+1,np.nan)								#Right signal velocity
	
	"""		Generate primitive vectors from Riemann-style Left/Right split		"""
	def get_W_LRsplit(self, cutoff, 				#cutoff tells where L/R boundary is
					  rhoL, PL, vL, rhoR, PR, vR):	#conditions on left, right
		for i in range(0, self.nx + 2):
			if self.x[i] <= cutoff*self.xend:
				self.W[i,0] = rhoL
				self.W[i,1] = vL
				self.W[i,2] = PL
			else:
				self.W[i,0] = rhoR
				self.W[i,1] = vR
				self.W[i,2] = PR
	
	
	"""		Generate primitive vectors from soundwave criteria		"""
	def get_W_soundwave(self, rhoB, drho,		#background density, density wave amplitude
						l, c_s, vB):			#wavelength, sound speed, bulk velocity
		C = c_s**2*rhoB**(1-self.gamma)/(self.gamma)  	#P = C*rho^gamma
		k = 2*np.pi/l
		for i in range(0, self.nx+2):
			if self.x[i] <= l:
				self.W[i,0] = rhoB + drho*np.sin(k*self.x[i])
				self.W[i,1] = vB + c_s* (drho/rhoB)*np.sin(k*self.x[i])
				self.W[i,2] = C*self.W[i,0]**self.gamma
			else:
				self.W[i,0] = rhoB
				self.W[i,1] = vB
				self.W[i,2] = C*rhoB**self.gamma
		
	"""		Set up Primitive vectors		"""
	def setup(self, 
			  vB=0, rhoB=1.0, drho=1e-3, l=0.2, c_s=1.0,					#variables for soundwave
			  vL=0, vR=0, rhoL=1.0, rhoR=0.1, PL=1.0, PR=0.125, cutoff=0.5,	#variables for Riemann split
			  IC="soundwave", boundary="periodic"):							#initial conditions type = "soundwave" or "LRsplit", boundary="periodic" or "flow"
		self.IC, self.boundary = IC, boundary
		if self.IC == "soundwave":
			self.get_W_soundwave(rhoB, drho, l, c_s, vB)
		elif self.IC == "LRsplit":
			self.get_W_LRsplit(cutoff, rhoL, PL, vL, rhoR, PR, vR)
			
		# Compute conserved quantities.
		U = prim2cons(self.W, self.gamma)
		self.Q = U * self.dx.reshape(-1,1)
		
	"""HLL Riemann Solver; note this also updates self.v, self.lp, and self.lm"""
	def riemann_solver(self, WL, WR, vf):
		fHLL = np.full((self.nx+1, 3), 0.)
		
		#	Transform lab-frame to face frame.
		WL[:,1] -= vf		# subtract face velocity from cell to LEFT of face
		WR[:,1] -= vf		# subtract face velocity from cell to RIGHT of face
		UL = prim2cons(WL, self.gamma)
		UR = prim2cons(WR, self.gamma)
		fL = prim2flux(WL, self.gamma)
		fR = prim2flux(WR, self.gamma)
		
		#	Calculate signal speeds
		csl = np.sqrt(self.gamma*WL[:,2]/WL[:,0])
		csr = np.sqrt(self.gamma*WR[:,2]/WR[:,0])
		self.lm = WL[:,1] - csl
		self.lp = WR[:,1] + csr
		
		#	Calculate HLL flux in frame of FACE	
		for i in range(0, self.nx+1):
			if self.lm[i] >= 0:
				fHLL[i,:] = fL[i,:]
			elif self.lm[i] < 0 and 0 < self.lp[i]:
				fHLL[i,:] = ( self.lp[i]*fL[i,:] - self.lm[i]*fR[i,:] + self.lp[i]*self.lm[i]*(UR[i,:] - UL[i,:]) ) / (self.lp[i]-self.lm[i])
			else:
				fHLL[i,:] = fR[i,:]
				
		
		
		#	Calculate net flux in frame of LAB
		self.fF = np.copy(fHLL)
		self.fF[:,1] += fHLL[:,0]*vf
		self.fF[:,2] += 0.5*fHLL[:,0]*vf**2 + fHLL[:,1]*vf
		
	
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
		return 
		
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
		t = 0
		plotcount = 1
		while t < self.tend:
			# 1) Compute primitive
			U = self.Q / self.dx.reshape(-1,1)
			self.W = cons2prim(U, self.gamma)
		
			# 2) Compute edge states
			self.W = boundary(self.W, self.boundary)
			
			# 3) Compute face velocity
			if self.mesh_type == "Lagrangian":
				self.v = np.copy(self.W[:,1])
				self.vf = (self.v[:-1] + self.v[1:])/2

			# 4) Compute fluxes
			WL, WR = np.copy(self.W)[:-1,:], np.copy(self.W)[1:,:]
			self.riemann_solver(WL, WR, self.vf)
			Uold = prim2cons(self.W, self.gamma) 
			Qold = Uold * self.dx.reshape(-1,1)		#nb use old dx here to get old Q
			Unew = np.copy(Uold) 
			
			# 5) Compute Courant condition
			dt = self.CFL_condition()
			dt = min(self.tend-t, dt)
			
			# 6) Update mesh.
			self.update_mesh(dt)
			
			# 7) First order time integration using Euler's method
			L = - np.diff(self.fF, axis=0)
			Unew[1:-1] = (Qold[1:-1] + L*dt) / self.dx[1:-1].reshape(-1,1)
			self.Q = Unew * self.dx.reshape(-1,1)	#nb use new dx here to get new Q
			
			if plotcount % 100000000 == 0:
				plt.plot(self.x, self.W[:,0] , label="w="+str(self.v[0]) + ", t=" + str(t+dt))
			t+=dt	
			plotcount+=1
			
	@property
	def pressure(self):
		return "Do some work"
		
		

grid = mesh(200, 0.2, 1.0,  mesh_type="Lagrangian")
grid.setup(boundary = "flow", IC = "LRsplit")
plt.scatter(grid.x, grid.W[:,0] ,label="Initial W[v]")
grid.solve()
plt.scatter(grid.x, grid.W[:,0] , label="w="+str(grid.v[0]))
plt.legend()
plt.pause(0.5)


grid = mesh(200, 0.2, 1.0)
grid.setup(boundary = "flow", IC = "LRsplit")
grid.solve()
plt.scatter(grid.x, grid.W[:,0] , label="w="+str(grid.v[0]) )
plt.legend()
plt.pause(0.5)
"""



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



t=0.2

plt.figure()

gridsound = mesh(1000,t, 1.0, mesh_type="Lagrangian")
gridsound.setup(vB=-1)
gridsound.solve()
plt.plot(gridsound.x, gridsound.W[:,0], label="Lagrangian")
plt.legend()
plt.pause(0.5)

gridsound = mesh(1000,t, 1.0, mesh_type="Fixed")
gridsound.setup(vB=-1)
gridsound.solve()
plt.plot(gridsound.x, gridsound.W[:,0], label="Eulerian")
plt.legend()
plt.pause(0.5)


#Relative motion = 0, should be identical
plt.figure()
gridsound = mesh(1000, t, 1.0, fixed_v = 0)
gridsound.setup(vB=0)
gridsound.solve()
plt.plot(gridsound.x, gridsound.W[:,0], label="vB=0, w="+str(gridsound.v[0]))
plt.legend()
plt.pause(0.5)

gridsound = mesh(500, t, 1.0, fixed_v = 1)
gridsound.setup(vB=1)
gridsound.solve()
plt.plot(gridsound.x, gridsound.W[:,0], label="vB=1, w="+str(gridsound.v[0]))
plt.legend()
plt.pause(0.5)
"""
"""
#Relative motion = sound speed, should match initial conditions
gridsound = mesh(500, t, 1.0, fixed_v = 1.0)
gridsound.setup()
gridsound.solve()
plt.plot(gridsound.x, gridsound.W[:,0], label="vB=0, w="+str(gridsound.v[0]))
plt.legend()
plt.pause(0.5)

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
plt.show()

