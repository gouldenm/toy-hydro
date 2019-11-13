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

def vec2prim(q):
    W = np.full(3, np.nan)
    W[0] = q[0]
    W[1] = q[1]/q[0]
    W[2] = q[2] * (gamma-1) - 0.5*W[0]*W[1]**2

#Takes in primitive vector, returns flux of the cell (not of interface)
def prim2flux(W):
    F = np.full(3, np.nan)
    F[0] = W[0]*W[1]
    F[1] = W[0]*W[1]**2 + W[2]
    F[2] = (W[2]/(gamma-1) + 0.5*W[0]*W[1]**2 + W[2])*W[1]


def Riemann(gamma, WL, WR):
    #sound speed on L and R:
    csL = np.sqrt(gamma*WL[1]/WL[0])
    csR = np.sqrt(gamma*WR[1]/WR[0])
    
    #wave speeds, v +- cs, according to Davis (1988)
    SL = WL[2] - csL
    SR = WR[2] + csR
    
    UL = prim2vec(WL)
    UR = prim2vec(WR)
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


#Integrate using first-order forward Euler method
# function takes prim vector W,
# converts to state vector U,
# integrates over time,
# and returns the updated prim vector W+1

    
