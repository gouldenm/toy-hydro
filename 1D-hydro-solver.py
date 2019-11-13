"""
A toy model for hydro simulations, used to test things
before trying out Arepo.
Author: Maggie Celeste
Date: 07/11/2019
"""

import numpy as np
import matplotlib.pyplot as plt

gamma = 1.4

#W = [rho, v, p] primitive vector of the cell
#q = [rho, rho*v, E] is the conserved state vector
#F = [rho*v, rho*v**2 + p, (E+p)*v] is the Flux
#rho is fluid density, v velocity, E = u + 1/2rho v**2 is total energy density

#Takes in primitive vector, returns conserved state vector
def prim2vec(W):
    q = np.full(3, np.nan)
    q[0] = W[0]
    q[1] = W[1]*W[0]
    #assuming an ideal gas for EOS:
    q[2] = W[2]/(gamma-1) + 0.5*W[0]*W[1]**2
    
    return(q)

#Takes in conserved state vector, returns primitive vector
def vec2prim(q):
    W = np.full(3, np.nan)
    W[0] = q[0]
    W[1] = q[1]/q[0]
    W[2] = q[2] * (gamma-1) - 0.5*W[0]*W[1]**2 *(gamma-1)
    
    return(W)

#Takes in primitive vector, returns flux of the cell (not of interface)
def prim2flux(W):
    F = np.full(3, np.nan)
    F[0] = W[0]*W[1]
    F[1] = W[0]*W[1]**2 + W[2]
    F[2] = (W[2]/(gamma-1) + 0.5*W[0]*W[1]**2 + W[2])*W[1]
    
    return(F)

#calculates net flux across interface between two cells,
# using HLL method
def Riemann(UL, UR):
    WL = vec2prim(UL)
    WR = vec2prim(UR)
    
    #sound speed on L and R:
    csL = np.sqrt(gamma*WL[1]/WL[0])
    csR = np.sqrt(gamma*WR[1]/WR[0])
    
    #wave speeds, v +- cs, according to Davis (1988)
    SL = WL[2] - csL
    SR = WR[2] + csR
    
    FL = prim2flux(WL)
    FR = prim2flux(WR)
    
    if (0 <= SL):
        Fhll = FL
    
    elif (SL <= 0 and 0 <= SR):
        Fhll = ( SR*FL - SL*FR + SL*SR*(UR-UL) )/(SR-SL)

    elif (SR <= 0):
        Fhll = FR

    else:
        print("Something broken in Riemann solver")

    return(Fhll)


#Integrate state vector using first-order forward Euler method

def time_integration(U, dx, dt, boundarytype="periodic"):
    UpdateU = np.copy(U)
    #compute first cell outside of loop, using ghost cell as boundary
    
    if boundarytype = "periodic":
        Fp = Riemann(U[0], U[1])
        Fm = Riemann(U[-1], U[0])
    for i in range(1, len(U)-1):
        
