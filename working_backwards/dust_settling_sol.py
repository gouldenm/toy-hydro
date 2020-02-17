from scipy.integrate import solve_ivp as odeint
import numpy as np
import matplotlib.pyplot as plt

#   Define function that calculates derivatives; returns x' and v' = x''
"""
def dust_settling_sol(xc, t_final, v_0=0.0, rho_0 = 1.0, K=10.0, g=-1.0):
    def dU_dt(U, t):
        return(U[1], -K*rho_0*np.exp(-U[0])*U[1] + g)
    
    ts = np.linspace(0, t_final, 10000)
    U0 = [xc, v_0]
    U = odeint(dU_dt, U0, ts)
    x = U[:,0]
    v = U[:,1]
    return(v)

print(dust_settling_sol(0, 100.0))
"""

def dust_settling_sol(H, v_0 = None, rho_0 = 1.0, K=10.0, g=-1.0):
    def dv_dx(x, v):
        return( g/v - K*rho_0*np.exp(x*H))
    
    if v_0 is None:
        v_0 = g/(K*rho_0*np.exp(H*1.0))

    xc = np.linspace(1.0,0, 100000)
    sol = odeint(dv_dx, [xc[0], xc[-1]], [v_0], t_eval=xc[1:-1], method='BDF')

    return (sol.t, sol.y[0])

#H = -1.0*5/3
#dust_settling_sol(H)
#plt.show()