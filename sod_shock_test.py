"""
A toy model for hydro simulations, used to test things
before trying out Arepo.
Author: Maggie Celeste
Date: 07/11/2019
"""

import numpy as np
import matplotlib.pyplot as plt

gamma = 1.4
CFL = 0.5
boundary = "flow"

#Set up empty vectors
#100 grid points w/ ghost cell either side
#thus 103 cell walls
nx = 1000
x = np.full(nx+2, np.nan)
xi = np.full(nx+3, np.nan)

#also set up state vector (old and new), and flux vector
q = np.full((nx+2, 3), np.nan)
qnew = np.full((nx+2, 3), np.nan)
f = np.full((nx+2, 3), np.nan)
fhll = np.full((nx+1, 3), np.nan)


#now populate grid vectors
#note cell walls indexed st wall i is left of cell i
dx = 1.0/nx
for i in range(0, nx+2):
    x[i] = (i-1.0)*dx

for i in range(0, nx+3):
    xi[i] = (i-1.0)*dx - 0.5*dx

#set up initial conditions and advection velocity array 
"""SOD SHOCK TUBE"""
rhoL, PL, vL = 1.0, 1.0, 1e-16
rhoR, PR, vR = 0.125, 0.1, 1e-16
for i in range(0, nx+2):
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
    
    
while t < tend:
    #impose boundary conditions onto state vectors
    if boundary == "flow":
        q[0,:] = q[1,:]
        q[nx+1,:] = q[nx,:]
    elif boundary == "periodic":
        q[0,:] = q[nx,:]
        q[nx+1,:] = q[1, :]
    
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
    
    for i in range(0, nx+1):
        ap = max(0, lp[i], lp[i+1])
        am = max(0, - lm[i], -lm[i+1])
        fhll[i,:] = (ap*f[i,:] + am*f[i+1,:] - ap*am*(q[i+1,:] - q[i])) / (ap + am)
    
    
    
    #make sure timestep abides by CFL condition
    maxv = max(max(lm), max(lp))
    #catch v=0 case, in which case just pick any dt. Note though that we should avoid v=0 by setting min value to be 1e-16 instead.
    try:
        dt = CFL*dx/maxv
    except:
        dt = 1e-10
        print("v=0 across cells")
        
    
    #Use average flux across each interface to determine q_(n+1)
    
    for i in range(1, nx+1):
        L = - (fhll[i,:] - fhll[i-1,:])/dx
        qnew[i,:] = q[i,:] + dt*L
        
    q = qnew
    """plt.plot(x, q[:,0])
    plt.savefig("shock" + str(t) + ".pdf")
    plt.close()
    print(dt)"""
    t+=dt

plt.plot(x, q[:,0])
plt.savefig("sod_shock" + str(t) + ".pdf")
plt.show()


#W = [rho, v, p] primitive vector of the cell
#q = [rho, rho*v, E] is the conserved state vector
#F = [rho*v, rho*v**2 + p, (E+p)*v] is the Flux
#rho is fluid density, v velocity, E = P + 1/2rho v**2 is total energy density
