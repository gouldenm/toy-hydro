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
def shock(mach, D_ratio, drag_params, shock_length, shock_step, t=0, c_s=1.0, Kin=1.0, rhog0=1.0, FB=0):
    # ### ### ### Redefine K ### ### ### ### 
    v_s =  mach * c_s 
    rhod0 = rhog0*D_ratio
    
    vg0 = v_s/mach**2
    rhog0 = mach**2 * rhog0 #changes across boundary
    
    def derivs(z, y):
        vd = y[0]
        
        try:
            vg = gas_velocity(vd, mach, D_ratio)
        except SolverError:
            return -1
        
        rhog = rhog0*vg0/vg
        rhod = rhod0*v_s/vd
        
        #new_c_s =  np.sqrt((rhog0*v_s**2 + rhod0*v_s**2 + rhog0*c_s**2 - rhog*(w*v_s)**2 - rhod*(wd*v_s)**2)/rhog)
        #print(new_c_s)
        
        dvdz = -abs(vd -vg) *(Kin*rhog/(vd))
        
        #drhod_dz = -dwdz*rhod0 / v_s**2
        #drhog_dz = FB* drhod_dz * v_s**2 / (c_s**2 - v_s**2)
        
        #drhod_dz = - dwdz * rhod / (wd*v_s**2)
        #drhog_dz = FB * drhod_dz * (wd*v_s)**2 / (c_s**2 - (w*v_s)**2)
        return(dvdz)#, drhog_dz, drhod_dz)
            
    ###################################################
    
    
    ###################################################
    def gas_velocity(vd, mach, D_ratio):
        
        beta = D_ratio*FB*(vd-v_s) - v_s - c_s**2/v_s
        disc = beta**2 - 4.0*c_s**2. #discriminant of quadratic
        
        if np.any(disc < 0.0):
            raise SolverError('Error in gas_velocity: no solution for gas velocity')
        
        vg = 0.5*(beta - np.sqrt(disc))
        
        return vg
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
            result = solve_ivp(derivs, [0, shock_length], [v_s, 0],
                               t_eval = arange(0, shock_length, shock_length/shock_step))#,
                               #method='BDF')
            #result = solver.solve(arange(0.0,  shock_length, shock_length/shock_step), [1.])
        else:
            result = solve_ivp(derivs, [0, shock_length], [(1.0-1e-4)*v_s],
                               t_eval = arange(0, shock_length, shock_length/shock_step))
            #result = solver.solve(arange(0.0,  shock_length, shock_length/shock_step), [1.-1.e-2])
                
        xi = result.t
        vd = result.y[0]

        
        #####################################################################
    except Exception as e:
        print (' Solver failed:', e)
        raise Exception('Solver failed. Great error message!')
        
    v_post =v_s/((1+FB*D_ratio)*mach**2)
    vg = gas_velocity(vd, mach, D_ratio)
    
    rho_g = rhog0*vg0 / vg
    rho_d = rhod0*v_s / vd
    
    dx = t*v_post
    scaled_x = xi + 10 - dx
    
    print(rho_g[-1], rho_d[-1])
    solution={
        'xi': scaled_x,  #shift to match up with our frame
        'wd': vd - v_post,
        'wg': vg - v_post,
        'rhog': rho_g,
        'rhod': rho_d
    }
    
    return solution





#shock(mach, D_ratio, drag_params, shock_length, shock_step)

"""
sol1 = shock(1.1, 0.01, {'drag_type':'power_law', 'drag_const':1.0}, 15., 1000.)
sol2 = shock(1.1, 0.1, {'drag_type':'power_law', 'drag_const':1.0}, 15., 1000.)
sol3 = shock(1.1, 1.0, {'drag_type':'power_law', 'drag_const':1.0}, 15., 1000.)

plt.figure()
plt.plot(sol1['xi'], sol1['wd'])
plt.plot(sol1['xi'], sol1['wg'])

plt.figure()
plt.plot(sol2['xi'], sol2['wd'])
plt.plot(sol2['xi'], sol2['wg'])

plt.figure()
plt.plot(sol3['xi'], sol3['wd'])
plt.plot(sol3['xi'], sol3['wg'])
"""


"""
sol0p1 = shock(0.96, 0.5, {'drag_type':'power_law', 'drag_const':1.0}, 15., 1000., t=4.0)
sol1p0 = shock(0.96, 1.0, {'drag_type':'power_law', 'drag_const':1.0}, 15., 1000.)
sol2p0 = shock(0.96, 2.0, {'drag_type':'power_law', 'drag_const':1.0}, 15., 1000.)

plt.figure()
plt.plot(sol0p1['xi'], sol0p1['wd'])
plt.plot(sol0p1['xi'], sol0p1['wg'])

plt.figure()
plt.plot(sol1p0['xi'], sol1p0['wd'])
plt.plot(sol1p0['xi'], sol1p0['wg'])

plt.figure()
plt.plot(sol2p0['xi'], sol2p0['wd'])
plt.plot(sol2p0['xi'], sol2p0['wg'])
plt.show()
###################################################
"""
