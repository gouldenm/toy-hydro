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
    
    rhod0 = rhog0*D_ratio
    rhog0 = rhog0 * (GAMMA+1) / (GAMMA-1)
    
    v_s =  mach * c_s 
    vg0 = v_s * (GAMMA-1) / (GAMMA+1)
    
    
    def derivs(z, y):
        vd = y[0]
        try:
            vg = gas_velocity(vd, mach, D_ratio)
        except SolverError:
            return -1
        rhog = rhog0*vg0/(vg*v_s)
        rhod = rhod0*v_s/(vd*v_s)
        
        
        
        dwdz = -abs(vd -vg) *(Kin*rhog/(vd))
        return(dwdz, 0, 0)
            
    ###################################################
    
    
    ###################################################
    def gas_velocity(vd, mach, D_ratio):
        v_s = c_s*mach
        A = 1 - 0.5*(GAMMA-1)/GAMMA
        B = D_ratio*(vd - v_s) - v_s - (P_0/ (rhog0*v_s))
        C = (GAMMA-1)/GAMMA * ( v_s**2 - 0.5*vd**2) + (P_0 / (rhog0))
        
        disc = B**2 - 4*A*C
        
        if np.any(disc < 0.0):
            raise SolverError('Error in gas_velocity: no solution for gas velocity')
        
        w = 0.5*(B - np.sqrt(disc))
        
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
        if mach > 1.:
            result = solve_ivp(derivs, [0, shock_length], [1.0, rhog0, rhod0],
                               t_eval = arange(0, shock_length, shock_length/shock_step),
                               method='Radau', atol=1e-14)
            #result = solver.solve(arange(0.0,  shock_length, shock_length/shock_step), [1.])
        else:
            result = solve_ivp(derivs, [0, shock_length], [(1.0-1e-4), rhog0, rhod0],
                               t_eval = arange(0, shock_length, shock_length/shock_step),
                               method="Radau")
            #result = solver.solve(arange(0.0,  shock_length, shock_length/shock_step), [1.-1.e-2])
                
        xi = result.t
        wd = result.y[0]

        
        #####################################################################
    except Exception as e:
        print (' Solver failed:', e)
        raise Exception('Solver failed. Great error message!')
    
    v_post =v_s/((1+FB*D_ratio)*mach**2)
    wg = gas_velocity(wd, mach, D_ratio)
    
    dx = t*v_post
    scaled_x = xi + offset - dx
    
    rho_g = rhog0*vg0 / (wg*v_s)
    rho_d = rhod0*v_s / (wd*v_s)
    
    P = rho_g * c_s**2
    
    solution={
        'xi': scaled_x,  #shift to match up with our frame
        'wd': wd*v_s - v_post,
        'wg': wg*v_s - v_post,
        'rhog': rho_g,
        'rhod': rho_d,
        'P': P
    }
    
    return solution

