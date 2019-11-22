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
q = np.full((nx+2, 3), np.nan)		#state vector
qnew = np.full((nx+2, 3), np.nan)	#new state vector after advection
f = np.full((nx+2, 3), np.nan)		#flux of cell centers
fhll = np.full((nx+1, 3), np.nan)	#HLL flux across interface

#set up initial conditions
CFL = 0.5							#CFL criterion determining how short timesteps should be
gamma = 1.4                         #gamma = heat capacity ratio
t = 0								#start time (should be 0 in most cases...)
tend = 0.2							#simulation ends after a step exceeds this time
extent = 1.0						#spatial extent of grid
cutoff = 0.1						#cutoff point for Left vs Right volume
dx = extent/nx						#initial uniform separation between cells (may change later)
rhoL, PL, vL = 1.0, 10.0, 1e-16 		#left volume
rhoR, PR, vR = 1.0, 0.125, 1e-16	#right volume

#Populate vectors
for i in range(0, nx+2):			#cell centers
    x[i] = (i-2.0)*dx
for i in range(0, nx+3):			#cell interfaces
    xi[i] = (i-2.0)*dx - 0.5*dx
for i in range(0, nx+2):			#state vectors
    if x[i] <= cutoff:
        q[i,0] = rhoL
        q[i,1] = rhoL*vL
        q[i,2] = PL/(gamma-1) + 0.5*rhoL*vL**2
    else:
        q[i,0] = rhoR
        q[i,1] = rhoR*vR
        q[i,2] = PR/(gamma-1) + 0.5*rhoR*vR**2


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
	INPUT = state vector
	OUTPUT = vector of HLL fluxes across interface, max signal velocity present in array
"""
def riemann_solver(q):
    #Calculate primitive variables (to make eqns clearer, but not strictly necessary)
    rho, E = q[:,0], q[:,2]
    v = q[:,1] / rho
    P = (gamma-1)*(E - 0.5*rho*v**2)

    #calculate flux
    f[:,0] = rho*v
    f[:,1] = rho*v**2 + P
    f[:,2] = (E+P)*v

    #calculate signal speeds (l shorthand for lambda, as in Springel notes and in duffell notes)
    cs = np.sqrt(gamma*P/rho)			#isothermal sound speed
    lm = v - cs							#backward sound speed
    lp = v + cs							#forward sound speed
    maxv = max(max(lm), max(lp))		#max signal speed in grid, used to calculate dt
    
    #Then the HLL Riemann solver calculates flux across each interface:
    for i in range(0, nx+1):
        ap = max(0, lp[i], lp[i+1]) 	#forward signal speed used
        am = max(0, - lm[i], -lm[i+1])	#backward signal speed used (nb signs)
        fhll[i,:] = (ap*f[i,:] + am*f[i+1,:] - ap*am*(q[i+1,:] - q[i])) / (ap + am)
    
    return((fhll, maxv))


""" PERFORM ADVECTION IN A WHILE LOOP -- STOPS WHEN SIMULATION TIME IS EXCEEDED
"""
while t < tend:
    q = boundary(q, "periodic")				#impose boundary conditions onto state vectors
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
    qnew = boundary(qnew, "periodic")
    q = qnew
    #plt.plot(x, q[:,0])
    #plt.savefig("shock" + str(t) + ".pdf")
    #plt.close()
    t+=dt

plt.plot(x, q[:,0])
#plt.savefig("sod_shock_0p2s.pdf")
plt.show()
