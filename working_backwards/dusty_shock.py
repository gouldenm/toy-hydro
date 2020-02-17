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
from scikits_extract import ode
from numpy import arange, array


###################################################
class SolverError(Exception):
    pass
###################################################
    
    
    
    
    
    
    

###################################################
def derivs(z, y, dydz, params):
    '''
    RHS function (rhseqn) for the ode function in scikits.odes

    Parameters
    ----------
    z : float
        position (never used here)
    y : array
        y[0] = wd
    dydz : array
        derivative of wd at the position z (length 1)
    params : dict
        dictionary of constants required in various places

    Returns
    -------
        None; this function just operates on the dydz vector
    '''

    try:
        w = gas_velocity(y[0], params['mach'], params['D_ratio'])
    except SolverError:
        return -1

    if params['drag_type'] == 'power_law':
        n = params['drag_const']
        dydz[0] = -abs(y[0]-w)**(n+1.)
    elif params['drag_type'] == 'third_order':
        k = params['drag_const']
        dydz[0] = -(1. + k*abs(y[0]-w)**2.)*(y[0]-w)
    elif params['drag_type'] == 'mixed':
        k = params['drag_const']
        dydz[0] = -(1. + k*abs(y[0]-w)**2.)**0.5*(y[0]-w)
        
###################################################
    
    















###################################################
def gas_velocity(wd, mach, D_ratio):
    '''
    Returns dimensionless gas velocity (vg/vs)
     
    Parameters
    ----------
    wd : float
        dimensionless dust velocity vd/vs
    mach : float
        mach number vs/cs (cs is preshock sound speed)
    D_ratio : float
        initial dust-to-gas mass density ratio

    Returns
    -------
    w : float
        normalised gas velocity vg/vs

    '''
    beta = 1.0 + mach**(-2.) + D_ratio*(1. - wd)
    disc = beta**2 - 4.0/mach**2. #discriminant of quadratic
    
    if disc < 0.0:
        raise SolverError('Error in gas_velocity: no solution for gas velocity')

    w = 0.5*(beta - sqrt(disc))
    
    return w
###################################################







###################################################
def shock(mach, D_ratio, drag_params, shock_length, shock_step):
    '''
    Solve. The main function of PyDustyShock. This solves
    the first order ODE describing a two-fluid dust-gas shock.
    
    Paramaters
    ----------
    mach : float
        sonic mach number vs/cs
    D_ratio : float
        preshock dust-to-gas mass density ratio
    drag_params : dict
        dictionary of things relevant to the drag, has keys
            'drag_type': str
                sets the form of the drag, 'power_law', 'third_order',
                'mixed' are the allowed entries
            'drag_const': float
                different for each drag type; let drag_const=k, then
                derivative of dust velocity is
                 
                 -abs(wd-wg)^(k+1.) for 'power_law'
                 -(1 + k|wd-wg|^2)(wd-wg) for 'third_order'
                 -(1 + k|wd-wg|^2)^0.5 (wd-wg) for 'mixed'
                 
                 where w=v/vs, vs is the shock velocity
    shock_length : float
        integration length (in units of xi?). I haven't figured this out
        yet, but for a shock with mach=2, D_ratio=2, 
        drag_type='power_law' and drag_const=0, this number should be ~10.
    shock_step : float
        number of points to sample the shock solution at, e.g. maybe
        10xshock_length

    Returns
    -------
    solution : dict
        dictionary with entries
        'xi': numpy array
            dimensionless position
        'wd': numpy array
            normalised dust velocity v_dust/vs
        'wg': numpy array
            normalised gas velocity v_gas/vs
    '''
    ### Couple of validity checks ###

    # The shock velocity must be greater than the combined fluid velocity
    if mach <= (1. + D_ratio)**-0.5:
        raise Exception('Mach number must be greater than (1+D)^-1/2')

    # Three types of drag implemented so far    
    drag_types = ['power_law', 'third_order', 'mixed']
    
    if drag_params['drag_type'] not in drag_types:
        raise Exception('drag_type not found; must be: \'power_law\', \'third_order\', or \'mixed\'')
    ###



    params = {
        'mach' : mach,
        'D_ratio' : D_ratio,
        'drag_type': drag_params['drag_type'],
        'drag_const': drag_params['drag_const']
    }



    ################## SOLVE THE ODES! ###################
    try:         
        solver = ode('cvode', derivs, user_data=params, max_steps=50000)
        
        if mach > 1.:
            result = solver.solve(arange(0.0,  shock_length, shock_length/shock_step), [1.])
        else:
            result = solver.solve(arange(0.0,  shock_length, shock_length/shock_step), [1.-1.e-2])
                
        xi = array(result[1])
        
        # Create array of the solutions wd
        wd = [result[2][i][0] for i in xrange(len(xi))]


        
        #####################################################################
    except Exception as e:
        print (' Solver failed:', e)
        raise Exception('Solver failed. Great error message!')
        




    solution={
        'xi': xi,
        'wd': array(wd),
        'wg': array([gas_velocity(wd[i], mach, D_ratio) for i in xrange(len(result[2]))])
    }
    
    return solution

sol = shock(2.0, 2.0, {'drag_type':'power_law', 'drag_const':0.0}, 10., 100.)
###################################################

