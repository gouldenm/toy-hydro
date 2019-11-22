""" A toy model for hydro simulations, used to test things before trying out Arepo.
	Author: Maggie Celeste
	Date: 07/11/2019
"""
import numpy as np
import matplotlib.pyplot as plt


"""	Functions to convert from primitive vector to conserved vector (and vice versa)"""
def prim2cons(W, gamma=1.4):
	U = np.full((len(W), 3), np.nan)		#conserved state vector
	U[:,0] = W[:,0]
	U[:,1] = W[:,0]*W[:,1]
	U[:,2] = W[:,2]/(gamma-1) + (W[:,0]*W[:,1]**2)/2
	return(U)

def cons2prim(U, gamma=1.4):
	W = np.full((len(U), 3), np.nan)		#primitive state vector
	W[:,0] = U[:,0]
	W[:,1] = U[:,0]/U[:,1]
	W[:,2] = (gamma-1)*(U[:,2] - (W[:,0]*W[:,1]**2)/2)
	return(W)


"""	Function to impose boundary conditions onto primitive state vectors
INPUT = prim state vector with ghost cells undefined / incorrect
OUTPUT = prim state vector with properly defined ghost cells
"""
def boundary2D(q,boundary):
   if boundary == "flow":
       q[0,:] = q[1,:]
       q[-1,:] = q[-2,:]
   elif boundary == "periodic":
       q[0,:] = q[-2,:]
       q[-1,:] = q[1,:]
   return(q)
   
def boundary1D(q,boundary):
   if boundary == "flow":
       q[0] = q[1]
       q[-1] = q[-2]
   elif boundary == "periodic":
       q[0] = q[-2]
       q[-1] = q[1]
   return(q)
	

""" Class to hold all information on the mesh:
	positions of mesh sites + interfaces, primitive variables at those sites, conserved variables
"""
class mesh:
	"""Establish empty vectors that will sit on the mesh + constants"""
	def __init__(self, nx,							#number of (non-ghost) cells
							 tend, xend,						#simulation cutoff time, spatial extent of grid
							 boundary = "periodic", #boundary conditions
							 IC = "soundwave",			#initial conditions flag: options = "soundwave" or "LRsplit"
							 CFL=0.5, gamma=1.4):
		
		self.nx, self.tend, self.xend = nx, tend, xend
		self.CFL, self.gamma = CFL, gamma
		self.IC, self.boundary = IC, boundary
		
		self.x = np.full(nx+2, np.nan)					#position of cell centre
		self.xf = np.full(nx+3, np.nan)				#position of cell face (cell face i is left of cell centre i)
		self.W = np.full((nx+2, 3), np.nan)			#primitive state vectors of cell centres
		self.U = np.full((nx+2, 3), np.nan)			#conserved state vectors of cell centres
		self.Wf = np.full((nx+1, 3), np.nan)		#primitive state vectors of cell faces (requires L and R cell, hence 1 less length)
		self.Uf = np.full((nx+1, 3), np.nan)		#conserved state vectors of cell faces
	
	
	"""Generate primitive vectors from Riemann-style L/R split"""
	def get_W_LRsplit(self,
										cutoff = 0.5,													#cutoff point for Left vs Right volume
										rhoL = 1.0, PL = 1.0, vL = 1e-16,			#left state conditions
										rhoR = 0.1, PR =0.125, vR= 1e-16):		#right state conditions:
		
		for i in range(0, self.nx + 2):
			if self.x[i] <= cutoff*self.xend:
				self.W[i,0] = rhoL
				self.W[i,1] = vL
				self.W[i,2] = PL
			else:
				self.W[i,0] = rhoR
				self.W[i,1] = vR
				self.W[i,2] = PR
	
	
	"""Generate primitive vectors from soundwave criteria"""
	def get_W_soundwave(self,
										 	rho_0 = 1.0, drho = 1e-3,		#background density, density wave amplitude
											l = 0.2,										#wavelength
											c_s = 1.0):			#sound speed
		
		C = c_s**2*rho_0**(1-self.gamma)/(self.gamma)  #P = C*rho^gamma
		k = 2*np.pi/l
		for i in range(0, self.nx+2):
			if self.x[i] <= l:
				self.W[i,0] = rho_0 + drho*np.sin(k*self.x[i])
				self.W[i,1] = c_s* (drho/rho_0)*np.sin(k*self.x[i])
				self.W[i,2] = C*self.W[i,0]**self.gamma
			else:
				self.W[i,0] = rho_0
				self.W[i,1] = 1e-16
				self.W[i,2] = C*rho_0**self.gamma
	
	
	"""Populate vectors in the mesh"""
	def setup(self):
		#grid is initially uniformly spaced by dx:
		dx = self.xend / self.nx 
		#	Cell centres
		for i in range(0, self.nx+2):
			self.x[i] = (i-1.0)*dx
		# Primitive vectors
		if self.IC == "soundwave":
			self.get_W_soundwave()
		elif self.IC == "LRsplit":
			self.get_W_LRsplit()
		#	Conserved vectors
		self.U = prim2cons(self.W)
	
	
	"""Function to move mesh sites"""
	def move(self):
		#check if last (not-ghost) cell exceeds the spatial extent of the grid
		if self.x[-2] > self.xend:
			print(self.W[:,0])
			#if periodic boundaries, roll prim vector elements along, ignoring ghost cells
			if self.boundary == "periodic":
				self.W[1:-1,:] = np.roll(self.W[1:-1,:], axis=0,shift = 1)
				#print(self.W[:,0])
				self.W = boundary2D(self.W, self.boundary)
				#print(self.W[:,0])
			#then subtract the grid excess from cell centres, as we've inserted + deleted a point
			x -= (x[-2] - xend)

grid = mesh(10, 0.2, 0.2)
grid.setup()
grid.move()


""" Function to call HLL Riemann Solver
	INPUT = primitive state vector				####DELETE THIS, face velocity (array of size len(W)-1)
	OUTPUT = flux at face, primitive vector at face
"""
"""def riemann_solver(W, mesh_v="Eulerian"):
	#calculate mesh velocity based on movement type
	if mesh_v="Eulerian":
		vf = np.full(nx+1, 0)
	elif mesh_v="Lagrangian":
		vf = np.full(nx+1, np.nan)
		for i in range(0, nx+1):
			v_L = W[i, 1]
			v_R = W[i+1, 1]
			vf[i] = (v_R + v_L)/2
	#if mesh_v is a float, treat as constant velocity across all faces
	else:
		vf = np.full(nx+1, mesh_v)
	
	#Transform lab-frame velocities to mesh-frame by subtracting mesh velocity
	#	NB: mesh velocity is potentially different at every face, 
	#		so we must handle "left" and "right" cell separately.
	#		To conserve vector operations (instead of loops), duplicate W.
	WL, WR = np.copy(W), np.copy(W)
	WL[:-1,1] -= vf		# subtract face velocity from face to RIGHT => list of left cells
	WR[1:,1] -= vf		# subtract face velocity from face to LEFT => lift of right cells
	#fix periodic boundaries 
	
	#Calculate conserved state at face
		
	





	f = np.full((nx+2, 3), np.nan)		#flux of cell centers
	f_face = np.full((nx+1, 3), np.nan)	#HLL flux at face
	
	#because the face velocity depends on L and R states, copy W into L and R arrays
	WL, WR = np.copy(W), np.copy(W)
	#transform into frame of face 
	WL[:-1,1] -= vf #cell to left of face
	WR[1:,1] -= vf #cell to right of face (=> i+1 for v corresponds to i for w)
	
	#restore boundary conditions
	WL = boundary(WL,"flow")
	WR = boundary(WR,"flow")
	
	#express rho, v, P, E in variables for ease of reading
	U = convert_prim2cons(WL)
    rho, v, P = WL[:,0], WL[:,1], WL[:,2]
	E = U[:,2]

    #calculate flux
    f[:,0] = rho*v
    f[:,1] = rho*v**2 + P
    f[:,2] = (E+P)*v

    #calculate signal speeds (l shorthand for lambda, as in Springel notes and in duffell notes)
    cs = np.sqrt(gamma*P/rho)	#isothermal sound speed
    lm = v - cs					#backward sound signal
    lp = v + cs					#forward sound signal
    
    #calculate conserved vector U at interface convert to prim vector W in frame of face
    U_face = (lp*f[:-1,:] + lm[i]*f[1:,:] - lm[i]*lp[i]*(U[1:,:]-U[:-1,:])/(lp+lm)
    W_face = convert_cons2prim(U_face)
    
    #Transform velocities back into lab frame, and convert back to conserved vector
    W_face[:-1, 1] += vf 
    U_lab = convert_prim2cons(boundary(W_face, "flow"))
    
    
    #Calculate flux of conserved variables at interface
    for i in range(0, nx+1):
        ap = max(0, lp[i], lp[i+1]) 	#forward signal speed used
        am = max(0, - lm[i], -lm[i+1])	#backward signal speed used (nb signs)
        f_face[i,:] = (ap*f[i,:] + am*f[i+1,:] - ap*am*(q[i+1,:] - q[i])) / (ap + am)
    
    return((f_face, ))
"""

"""Main body of the code, tying functions together
"""
"""def execute(nx, 					#number of (non-ghost) cells
			tend, xend,				#simulation cutoff time, spatial extent of grid
			CFL=0.5, gamma=1.4)
	#Populate cell centres + interfaces
	dx = xend / nx 					#initial uniform separation between cells
	x = np.full(nx+2, np.nan)			#cell centers
	xi = np.full(nx+3, np.nan)			#cell interfaces (interface i is left of cell i)
	dx = extent/nx
	for i in range(0, nx+2):			#cell centers
   		x[i] = (i-2.0)*dx
	for i in range(0, nx+3):			#cell interfaces
   		xi[i] = (i-2.0)*dx - 0.5*dx
   	
   	#prepare initial conditions
   	t = 0							#initial time
	W = init_sound(x)				#initial primitive vector
   	
   	"""
   	
   	
   	
   	
   	
