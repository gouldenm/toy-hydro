from scipy.integrate import solve_ivp
import numpy as np
import matplotlib.pyplot as plt

class SolverError(Exception):
    pass

def dustyshock_sol(D, M, rho_g_0, c_s, K):
    rho_d_0 = D * rho_g_0
    v_s = c_s * M
    #C = K / (rho_d_0 * v_s)
    
    def gas_velocity(wd):
        b = D*(wd - 1.) - 1. - M**(-2)
        disc = b**2 - 4/ M**2
        print(disc)
        #if disc < 0.0:
        #    print('Error in gas_velocity: no solution for gas velocity')
        #    #quit()
        wg = 0.5*(-b-np.sqrt(disc))
        
        
        return(wg)
    
    def dwd_dz(wd, z):
        wg = gas_velocity(wd)
        return ( - np.abs(wd-wg))
    
    
    xc = np.linspace(0, 2.0, 100)
    print(xc)
    sol = solve_ivp(dwd_dz, [xc[0], xc[-1]], [0.999], t_eval = xc[1:-1])
    wd = sol.y[0]
    wg = gas_velocity(wd)
    x =  sol.t
    print(x)
    return(x, wd, wg)

sol = dustyshock_sol(1.0, 0.95, 1.0, 1., 1.)

plt.plot(sol[0], sol[1], label="Dust")
plt.plot(sol[0], sol[2], label="Gas")
plt.legend()
plt.show()