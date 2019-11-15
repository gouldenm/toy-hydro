"""
A toy model for hydro simulations, used to test things
before trying out Arepo.
Author: Maggie Celeste
Date: 07/11/2019
"""

import numpy as np
import matplotlib.pyplot as plt

gamma = 1.4
CFL = 0.01
boundary = "flow"
g = 2 #number of ghost cells

#Set up empty vectors
#100 grid points w/ TWO ghost cell either side
#thus 103 cell walls
nx = 500
x = np.full(nx+4, np.nan)
xi = np.full(nx+5, np.nan)

#also set up state vector (old and new)
q = np.full((nx+4, 3), np.nan)
qnew = np.full((nx+4, 3), np.nan)

#vectors used in Runge Kutta integration
q1 = np.full((nx+4, 3), np.nan)
q2 = np.full((nx+4, 3), np.nan)

#flux in cell center and across interface
f = np.full((nx+4, 3), np.nan)
fhll = np.full((nx+3, 3), np.nan)


#now populate grid vectors
#note cell walls indexed st wall i is left of cell i
dx = 1.0/nx
for i in range(0, nx+4):
    x[i] = (i-2.0)*dx

for i in range(0, nx+5):
    xi[i] = (i-2.0)*dx - 0.5*dx

#set up initial conditions and advection velocity array 
"""SOD SHOCK TUBE"""
rhoL, PL, vL = 1.0, 1.0, 1e-16
rhoR, PR, vR = 0.1, 0.125, 1e-16
for i in range(0, nx+4):
    if x[i] <= 0.5:
        q[i,0] = rhoL
        q[i,1] = rhoL*vL
        q[i,2] = PL/(gamma-1) + 0.5*rhoL*vL**2
    else:
        q[i,0] = rhoR
        q[i,1] = rhoR*vR
        q[i,2] = PR/(gamma-1) + 0.5*rhoR*vR**2

#simulation end time
tend = 0.2

#perform advection
t = 0.0


"""RIEMANN SOLVER FUNCTION
INPUT = old state vector
OUTPUT = HLL flux across interface (vector), and
         max velocity -- used to determine max timestep"""
def riemann_solver(q):
    #Calculate primitive variables (to make eqns clearer, but not strictly necessary)
    rho = q[:,0]
    E = q[:,2]
    v = q[:,1] / rho
    P = (gamma-1)*(E - 0.5*rho*v**2)
    
    #calculate flux
    f[:,0] = rho*v
    f[:,1] = rho*v**2 + P
    f[:,2] = (E+P)*v
    
    #calculate signal speeds (l shorthand for lambda, as in Springel notes and in duffell notes)
    cs = np.sqrt(gamma*P/rho)
    lm = v - cs
    lp = v + cs
    
    #Then the HLL Riemann solver calculates average flux across each interface:
    
    for i in range(0, nx+3):
        ap = max(0, lp[i], lp[i+1])
        am = max(0, - lm[i], -lm[i+1])
        fhll[i,:] = (ap*f[i,:] + am*f[i+1,:] - ap*am*(q[i+1,:] - q[i])) / (ap + am)
    
    maxv = max(max(lm), max(lp))
    
    return((fhll, maxv))

"""BOUNDARY CONDITIONS FUNCTION
impose boundary conditions onto state vectors
INPUT = state vector with ghost cells undefined
OUTPUT = state vector with properly defined ghost cells
"""
def boundary(q,boundary):
    #print("Before boundaries:", q[:,0])
    if boundary == "flow":
        q[0,:], q[1,:] = q[2,:], q[2,:]
        q[nx+2,:], q[nx+3,:] = q[nx+1,:],q[nx+1,:]
    elif boundary == "periodic":
        print("FIX PERIODIC BOUNDARIES")
        q[0,:] = q[nx,:]
        q[nx+1,:] = q[1, :]
    #print("After boundaries:", q[:,0])
    return(q)


#PERFORM ADVECTION IN WHILE LOOP
while t < tend:
	q = boundary(q, "flow")
		
	#Integrate in time using Runge-Kutta method (3rd order)    
	fhll, maxv = riemann_solver(q)
		
	#make sure timestep abides by CFL condition
	try:
		dt = CFL*dx/maxv
	except:
		dt = 1e-10
		print("v=0 is max velocity")
	
	print("First step of RK:")
	for i in range(1, nx+3):
		L = - (fhll[i,:] - fhll[i-1,:])/dx
		q1[i,:] = q[i,:] + dt*L
	q1 = boundary(q1, "flow")
	
	print("Second step of RK:")
	fhll1, maxv = riemann_solver(q1)
	for i in range(1, nx+3):
		L1 = - (fhll1[i,:] - fhll1[i-1,:])/dx
		q2[i,:] = 0.75*q[i,:] + 0.25*q1[i,:] + 0.25*dt*L1
	q2 = boundary(q2, "flow")
		
	print("Third step of RK:")
	fhll2, maxv = riemann_solver(q2)
	for i in range(1, nx+3):
		L2 = (fhll2[i,:] - fhll2[i-1,:])/dx
		qnew[i,:] = (1/3)*q[i,:] + (2/3)*q2[i,:] + (2/3)*dt*L2
	
	q = qnew
	plt.plot(x, q[:,0])
	#plt.savefig("shock" + str(t) + ".pdf")
	#plt.close()
	t+=dt

plt.plot(x, q[:,0])
plt.savefig("strong_shock" + str(t) + ".pdf")
plt.show()


#W = [rho, v, p] primitive vector of the cell
#q = [rho, rho*v, E] is the conserved state vector
#F = [rho*v, rho*v**2 + p, (E+p)*v] is the Flux
#rho is fluid density, v velocity, E = P + 1/2rho v**2 is total energy density
