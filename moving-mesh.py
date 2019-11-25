""" A toy model for hydro simulations, used to test things before trying out Arepo.
	Author: Maggie Celeste
	Date: 07/11/2019
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
	W[:,1] = U[:,0]/U[:,1]
	W[:,2] = (gamma-1)*(U[:,2] - (W[:,0]*W[:,1]**2)/2)
	return(W)

def prim2flux(W, gamma):
	F = np.full((len(W), 3), np.nan)		#conserved state vector
	F[:,0] = W[:,0]*W[:,1]
	F[:,1] = W[:,0]*W[:,1]**2 + W[:,2]
	F[:,2] = W[:,1] * (W[:,2]/(gamma-1) + (W[:,0]*W[:,1]**2)/2 + W[:,2])
	return(F)


"""	Function to impose boundary conditions onto primitive state vectors
INPUT = prim state vector with ghost cells undefined / incorrect
OUTPUT = prim state vector with properly defined ghost cells
"""
def boundary(q,boundary):
   if boundary == "flow":
       q[0,:] = q[1,:]
       q[-1,:] = q[-2,:]
   elif boundary == "periodic":
       q[0,:] = q[-2,:]
       q[-1,:] = q[1,:]
   return(q)
	

"""	Class to hold all information on the mesh"""
class mesh:
	"""Establish empty vectors that will sit on the mesh + constants"""
	def __init__(self, nx,					#number of (non-ghost) cells
				 tend, xend,				#simulation cutoff time, spatial extent of grid
				 boundary = "periodic", 	#boundary conditions: options = "flow" or "periodic"
				 IC = "soundwave",			#initial conditions flag: options = "soundwave" or "LRsplit"
				 mesh_type = "Fixed",		#type of mesh movement: options = "Fixed" or "Lagrangian"
				 fixed_v = 0,
				 CFL=0.5, gamma=1.4):
		self.nx, self.tend, self.xend = nx, tend, xend
		self.CFL, self.gamma = CFL, gamma
		self.IC, self.boundary = IC, boundary
		self.mesh_type = mesh_type
		
		self.x = np.full(nx+2, np.nan)				#position of cell centre
		self.cell_widths = np.full(nx+2, np.nan)	#size of cell
		self.W = np.full((nx+2, 3), np.nan)			#primitive state vectors (at cell centres)
		self.fF = np.full((nx+1, 3), np.nan)		#Net flux across face (lab frame)
		self.v = np.full(nx+2, fixed_v)				#velocity of cell centres -- defaults to 0 for Eulerian mesh
		self.vf = np.full(nx+1, fixed_v)			#velocity of cell faces
		self.lm= np.full(nx+1, np.nan)				#Left signal velocity (keep for calculating dt)
		self.lp= np.full(nx+1, np.nan)				#Right signal velocity (keep for calculating dt)
	
	"""Generate primitive vectors from Riemann-style L/R split"""
	def get_W_LRsplit(self,
					  cutoff = 0.5,								#cutoff point for Left vs Right volume
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
						l = 0.2,						#wavelength
						c_s = 1.0):						#sound speed
		
		C = c_s**2*rho_0**(1-self.gamma)/(self.gamma)  #P = C*rho^gamma
		k = 2*np.pi/l
		for i in range(0, self.nx+2):
			if self.x[i] <= self.xend:
				self.W[i,0] = rho_0 + drho*np.sin(k*self.x[i])
				self.W[i,1] = c_s* (drho/rho_0)*np.sin(k*self.x[i])
				self.W[i,2] = C*self.W[i,0]**self.gamma
			else:
				self.W[i,0] = rho_0
				self.W[i,1] = 1e-16
				self.W[i,2] = C*rho_0**self.gamma
	
	
	"""Populate vectors in the mesh"""
	def setup(self):
		#	grid is initially uniformly spaced by dx, st x[1]=0 and x[-2]=xend:
		dx = self.xend / (self.nx -1)
		#	Cell centres
		for i in range(0, self.nx+2):
			self.x[i] = (i-1.0)*dx
		# 	Primitive vectors
		if self.IC == "soundwave":
			self.get_W_soundwave()
		elif self.IC == "LRsplit":
			self.get_W_LRsplit()
	
	
	"""HLL Riemann Solver; note this also updates self.v, self.lp, and self.lm"""
	def riemann_solver(self):
		fHLL = np.full((self.nx+1, 3), np.nan)
		
		if self.mesh_type == "Lagrangian":
			self.v = self.W[:,1]
		#	Get face velocity as average of the adjacent cells
		for i in range(0, self.nx+1):
			v_L = self.v[i]
			v_R = self.v[i+1]
			self.vf[i] = (v_R + v_L)/2
			
		#	Transform lab-frame to face frame.	NB: Face frame may vary for every face => must compute WL, WR separately
		WL, WR = np.copy(self.W)[:-1,:], np.copy(self.W)[1:,:]
		WL[:,1] -= self.vf		# subtract face velocity from cell to LEFT of face
		WR[:,1] -= self.vf		# subtract face velocity from cell to RIGHT of face
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
		self.fF = fHLL
		self.fF[:,1] += fHLL[:,0]*self.vf
		self.fF[:,2] += 0.5*fHLL[:,0]*self.vf**2 + fHLL[:,1]*self.vf
	
	
	"""	Calculate time step duration according to Courant condition; note must be called after Riemann Solver generates v, lp, lm"""
	def CFL_condition(self):
		self.cell_widths[1:-1] = self.x[2:] - self.x[:-2]
		self.cell_widths[0], self.cell_widths[-1] = self.cell_widths[1], self.cell_widths[-2]
		dtm = self.CFL * self.cell_widths[:-1] / np.absolute(self.lm)
		dtp = self.CFL * self.cell_widths[1: ] / np.absolute(self.lp)
		mesh_v = max(self.v)
		if mesh_v > 1e-16:
			dt_mesh = self.CFL * self.cell_widths / np.absolute(self.v)
			dt = min(min(dtm), min(dtp), min(dt_mesh))
		else:
			dt = min(min(dtm), min(dtp))
		return(dt)
	
	
	"""	Move cells (i.e. change x coordinate),rearrange cells if any fall off the grid boundaries
		NB: doesn't work if grid cells exceed spatial extent on both sides """
	def move_cells(self, dt):
		#  Modify x coordinate based on velocity of cell centre
		self.x += self.v * dt
		#  Check if any (non-ghost) cells exeed the spatial extent of the grid in + x direction
		print(self.x)
		self.x += 0.09
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
		#TODO: If any cells lie outside spatial extent in flow regime, just delete and insert extras on the other side to balance
	
	
	"""	Function tying everything together into a hydro solver"""
	def solve(self):
		self.setup()
		t = 0
		plotcount = 0
		while t < self.tend:
			print(t)
			#print(self.W[:,0])
			self.W = boundary(self.W, self.boundary)
			Uold = prim2cons(self.W, self.gamma)
			Unew = np.copy(Uold)
			self.setup()
			self.riemann_solver()
			dt = self.CFL_condition()
			#First order time integration using Euler's method
			for i in range(1, self.nx+1):
				L = - (self.fF[i,:] - self.fF[i-1,:])/self.cell_widths[i]
				Unew[i,:] = Uold[i,:] + L*dt
			Unew = boundary(Unew, self.boundary)
			self.W = cons2prim(Unew, self.gamma)
			if plotcount % 20 == 0:
				#plt.close()
				plt.plot(self.x, self.W[:,2])
				plt.pause(0.5)
			t+=dt	
			plotcount+=1
		

grid = mesh(200, 0.5, 0.2, boundary = "flow", IC = "LRsplit")
grid.solve()

plt.show()
