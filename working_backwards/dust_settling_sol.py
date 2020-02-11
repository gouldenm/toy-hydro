from scipy.integrate import odeint
import numpy as np

#   Define function that calculates derivatives; returns x' and v' = x''

def dust_settling_sol(xc, t_final, v_0=0.0, rho_0 = 1.0, K=1.0, g=-1.0):
    def dU_dt(U, t):
        return(U[1], -K*rho_0*np.exp(-U[0])*U[1] + g)
    
    ts = np.linspace(0, t_final, 1000)
    U0 = [xc, v_0]
    U = odeint(dU_dt, U0, ts)
    x = U[:,0]
    v = U[:,1]
    return(v)

print(dust_settling_sol(0, 100.0))
"""

def dust_settling_sol(xc, t_final, v_0=0, rho_0 = 1.0, K=1.0, g=-1.0):
    def dv_dt(v, t):
        return(-K*rho_0*np.exp(-xc)*v + g)
    
    ts = np.linspace(0, t_final, 1000)
    v = odeint(dv_dt, v_0, ts)
    return(v)

print(dust_settling_sol(0, 10)[-1])
"""
