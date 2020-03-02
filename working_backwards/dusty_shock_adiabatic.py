#Copyright 2016 Andrew Lehmann
#
#Licensed under the Apache License, Version 2.0 (the "License");
#you may not use this file except in compliance with the License.
#You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
#Unless required by applicable law or agreed to in writing, software
#distributed under the License is distributed on an "AS IS" BASIS,
#WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#See the License for the specific language governing permissions and
#limitations under the License.

from math import sqrt
import numpy as np
from scikits_extract import ode
from scipy.integrate import solve_ivp
from numpy import arange, array
import matplotlib.pyplot as plt


###################################################
class SolverError(Exception):
    pass
###################################################

###################################################
def shock(mach, D_ratio, drag_params, shock_length, shock_step, GAMMA, P_0,
          t=0, Kin=1.0, rhog0=1.0, FB=0, offset=0.0):
    c_s = np.sqrt(GAMMA*P_0 / rhog0)
    v_s =  mach * c_s 
    
    v_g0 = v_s * (GAMMA-1) / (GAMMA+1)

    A0 = (GAMMA+1) / (2*GAMMA)
    B0 = -v_s - P_0 / (rhog0*v_s)
    C0 = (GAMMA-1)/(2*GAMMA)*v_s**2 + P_0/rhog0
    
    vg0 = (-B0 - np.sqrt(B0**2 - 4*A0*C0)) / (2*A0)
    
    A = (1+FB*D_ratio)* (GAMMA+1) / (2*GAMMA)
    B = - ( v_s*(1+FB*D_ratio) + P_0 / (rhog0*v_s))
    C = (GAMMA-1)/(2*GAMMA) *(v_s**2 * (1+FB*D_ratio) ) + P_0/rhog0
    
    v_post = (-B - np.sqrt(B**2 - 4*A*C)) / (2*A)
    
    rhod0 = rhog0*D_ratio
    
    mg = rhog0*v_s
    md = rhod0*v_s
    
    E = 0.5 * v_s**2 * (FB*md+mg) + (GAMMA/(GAMMA-1)) * P_0 * v_s
    
    rhog0in = np.copy(rhog0)
    rhog0 = rhog0 * v_s / vg0
    
    
    def derivs(z, y):
        vd = y[0]
        try:
            vg = gas_velocity(vd, mach, D_ratio)
        except SolverError:
            return -1
        rhog = rhog0in*v_s/(vg)
        rhod = rhod0*v_s/(vd)
        
        
        
        dwdz = -abs(vd -vg) *(Kin*rhog/(vd))
        return(dwdz, 0, 0)
            
    ###################################################
    
    
    ###################################################
    def gas_velocity(vd, mach, D_ratio):
        A = (GAMMA+1)/(2*GAMMA)
        B = FB*D_ratio*(vd - v_s) - v_s - P_0 / (rhog0in*v_s)
        C = (GAMMA-1)/(2*GAMMA) * ( FB*D_ratio*(v_s**2 - vd**2) + v_s**2)  + P_0 / rhog0in
        disc = B**2 - 4*A*C
        
        if np.any(disc < 0.0):
            raise SolverError('Error in gas_velocity: no solution for gas velocity')
        
        w = (-B - np.sqrt(disc)) / (2*A)
        
        return w
    ###################################################

    # The shock velocity must be greater than the combined fluid velocity
    if mach <= (1. + D_ratio)**-0.5:
        raise Exception('Mach number must be greater than (1+D)^-1/2')

    # Three types of drag implemented so far    
    drag_types = ['power_law', 'third_order', 'mixed']
    
    if drag_params['drag_type'] not in drag_types:
        raise Exception('drag_type not found; must be: \'power_law\', \'third_order\', or \'mixed\'')
    ###

    ################## SOLVE THE ODES! ###################
    try:
        t_eval = arange(0, shock_length, shock_length/shock_step)
        if mach > 1.:
            result = solve_ivp(derivs, [0, shock_length], [v_s, rhog0, rhod0],
                               t_eval=t_eval,
                               method='BDF', atol=1e-14)
            #result = solver.solve(arange(0.0,  shock_length, shock_length/shock_step), [1.])
        else:
            result = solve_ivp(derivs, [0, shock_length], [(v_s-1e-4), rhog0, rhod0],
                               t_eval = arange(0, shock_length, shock_length/shock_step),
                               method="Radau")
            #result = solver.solve(arange(0.0,  shock_length, shock_length/shock_step), [1.-1.e-2])
                
        xi = result.t
        vd = result.y[0]
        
        #####################################################################
    except Exception as e:
        print (' Solver failed:', e)
        raise Exception('Solver failed. Great error message!')
    
    vg = gas_velocity(vd, mach, D_ratio)
    
    #### for steady solution only:
    v_post = 0
    
    dx = t*v_post
    scaled_x = xi + offset - dx
    
    rho_g = rhog0*vg0 / (vg)
    rho_d = rhod0*v_s / (vd)
    
    P = (GAMMA-1)/GAMMA * ( E - 0.5*(FB*md*vd**2 + mg*vg**2)) / vg
    
    solution={
        'xi': scaled_x,  #shift to match up with our frame
        'wd': vd - v_post,
        'wg': vg- v_post,
        'rhog': rho_g,
        'rhod': rho_d,
        'P': P
    }
    
    return solution

