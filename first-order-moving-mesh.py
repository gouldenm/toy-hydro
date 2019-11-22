""" A toy model for hydro simulations, used to test things before trying out Arepo.
	Author: Maggie Celeste
	Date: 07/11/2019
"""
import numpy as np
import matplotlib.pyplot as plt

#Initialise vectors
nx = 1000							#number of cells (nb total number = nx + 2 ghost cells)
x = np.full(nx+2, np.nan)			#cell centers
xi = np.full(nx+3, np.nan)			#cell interfaces (interface i is left of cell i)
W = np.full((nx+2, 3), np.nan)		#primitive state vector
Wnew = np.full((nx+2, 3), np.nan)	#new primitive state vector

#set up initial conditions
CFL = 0.5							#CFL criterion determining how short timesteps should be
gamma = 1.4							#gamma = heat capacity ratio
t = 0								#start time (should be 0 in most cases...)
tend = 0.2							#simulation ends after a step exceeds this time
extent = 1.0						#spatial extent of grid
cutoff = 0.5						#cutoff point for Left vs Right volume
dx = extent/nx						#initial uniform separation between cells (may change later)
rhoL, PL, vL = 1.0, 1.0, 1e-16 		#left state
rhoR, PR, vR = 0.1, 0.125, 1e-16	#right state

#Populate vectors
for i in range(0, nx+2):			#cell centers
    x[i] = (i-2.0)*dx
for i in range(0, nx+3):			#cell interfaces
    xi[i] = (i-2.0)*dx - 0.5*dx
for i in range(0, nx+2):			#state vectors
    if x[i] <= cutoff:
        W[i,0] = rhoL
        W[i,1] = vL
        W[i,2] = PL
    else:
        W[i,0] = rhoR
        W[i,1] = vR
        W[i,2] = PR

"""	Functions to convert from primitive vector to conserved vector (and vice versa)
	W = primitive vector = [i,[rho, v, P]] for all cells i
	U = conserved vector = [i,[rho, rho*v, rho*e]] for all cells i
"""
def convert_prim2cons(W):
	U = np.full((nx+2, 3), np.nan)		#conserved state vector
	U[:,0] = W[:,0]
	U[:,1] = W[:,0]*W[1]
	U[:,2] = W[:,2]/(gamma-1) + (W[:,0]*W[:,1]**2)/2
	return(U)

def convert_cons2prim(U):
	W = np.full((nx+2, 3), np.nan)		#primitive state vector
	W[:,0] = U[:,0]
	W[:,1] = U[:,0]/U[:,1]
	W[:,2] = (gamma-1)*(U[:,2] - (W[:,0]W[:,1]**2)/2)
	return(W)

"""	Function to impose boundary conditions onto state vectors
	INPUT = state vector with ghost cells undefined
	OUTPUT = state vector with properly defined ghost cells
"""
def boundary(q,boundary):
    if boundary == "flow":
        q[0,:] = q[1,:]
        q[nx+1,:] = q[nx,:]
    elif boundary == "periodic":
        q[0,:] = q[nx,:]
        q[nx+1,:] = q[1,:]
    return(q) 

""" Function to call HLL Riemann Solver
	INPUT = primitive state vector, face velocity (array of size len(W)-1)
	OUTPUT = flux at face, primitive vector at face
"""
def riemann_solver(W, vf):
	f = np.full((nx+2, 3), np.nan)		#flux of cell centers
	fhll = np.full((nx+1, 3), np.nan)	#HLL flux at face
	Uhll = np.full((nx+1, 3), np.nan)	#conserved vector at face
	
	#because the face velocity depends on L and R states, copy W into L and R arrays
	WL, WR = np.copy(W), np.copy(W)
	#transform into frame of face (with variable v for every face)
	WL[:-1,1] -= vf #cell to left of face
	WR[1:,1] -= vf #cell to right of face (=> i+1 for v corresponds to i for w)
	
	#restore boundary conditions
	WL = boundary(WL,"flow")
	WR = boundary(WR,"flow")

    rho, v, P = WL[:,0], WL[:,1], WL[:,2]
	E = convert_prim2cons(WL)[:,2]

    #calculate flux
    f[:,0] = rho*v
    f[:,1] = rho*v**2 + P
    f[:,2] = (E+P)*v

    #calculate signal speeds (l shorthand for lambda, as in Springel notes and in duffell notes)
    cs = np.sqrt(gamma*P/rho)			#isothermal sound speed
    lm = v - cs					#backward sound speed
    lp = v + cs					#forward sound speed
    
    #Then the HLL Riemann solver calculates flux across each interface:
    for i in range(0, nx+1):
        ap = max(0, lp[i], lp[i+1]) 	#forward signal speed used
        am = max(0, - lm[i], -lm[i+1])	#backward signal speed used (nb signs)
        fhll[i,:] = (ap*f[i,:] + am*f[i+1,:] - ap*am*(q[i+1,:] - q[i])) / (ap + am)
    
    return((fhll, maxv))

"""#calculate face velocities
	for i in range(0, nx+1):
		v_L = W[i, 1]
		v_R = W[i+1, 1]
		v_f = (v_R + v_L)/2
"""

""" PERFORM ADVECTION IN A WHILE LOOP -- STOPS WHEN SIMULATION TIME IS EXCEEDED
"""
while t < tend:
    q = boundary(q, "flow")			#impose boundary conditions onto state vectors
    fhll, maxv = riemann_solver(q)		#apply Riemann Solver
    #calculate time step according to CFL criterion
    try:
        dt = CFL*dx/maxv
    except:
        dt = 1e-10
        print("v=0 somewhere")
        
    #perform time integration using Heun's method (predictor - corrector, same as AREPO)
    #NB CURRENTLY ONLY HAS FIRST STEP = EULER = FIRST ORDER
    for i in range(1, nx+1):
        L = - (fhll[i,:] - fhll[i-1,:])/dx
        qnew[i,:] = q[i,:] + L*dt
    qnew = boundary(qnew, "flow")
    q = qnew
    #plt.plot(x, q[:,0])
    #plt.savefig("shock" + str(t) + ".pdf")
    #plt.close()
    t+=dt

plt.plot(x, q[:,0])
#plt.savefig("sod_shock_0p2s.pdf")
plt.show()
