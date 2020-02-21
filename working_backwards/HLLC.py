import numpy as np



class HLLC(object):
    '''HLLC Riemann Solver for adiabatic/isothermal gasses.

    The conserved variables, Q, are:
        Q[0,:] = rho
        Q[1:ndim+1,:] = rho*v
        Q[ndim+1. :] = 0.5*rho*|v|**2 + P/(gamma-1) 

    The parameter iso_cs determines whether the isothermal or adiabatic mode
    is activated. By default (iso_cs = None), an adiabatic solver is used.

    args:
        gamma  : float, adiabatic index, default = 5/3.
        iso_cs : float or None, isothermal sound speed.
    '''
    def __init__(self, gamma=5/3., iso_cs=None):

        self._gamma = gamma
        self._iso_cs = iso_cs


    def __call__(self, Ql, Qr, jdir=0):
        '''Compute the HLLC fluxes'''

        # Make sure we have the correct shape
        Ql, Qr = map(self._as_2D_array, [Ql, Qr])
        
        Sl, Sr = map(self._compute_state, [Ql, Qr])

        # Wave speeds
        Smin, Smax = self._HLL_speeds(Sl, Sr, jdir)

        Fl = self._flux(Sl, Ql, jdir)
        Fr = self._flux(Sr, Qr, jdir)

        if self._iso_cs is not None:
            Fm = (Smax*Fl - Smin*Fr + Smin*Smax*(Qr-Ql)) / (Smax - Smin)

            return (Fr * ( Smax <= 0) +
                    Fm * ((Smin <= 0) & (Smax > 0)) +
                    Fl * ( Smin >  0))
            
        else:
            Sstar = self._star_speed(Sl, Sr, Smin, Smax, jdir)
            Fml = Fl + self._compute_wave_jump(Sl, Ql, jdir, Smin, Sstar)
            Fmr = Fr + self._compute_wave_jump(Sr, Qr, jdir, Smax, Sstar)
            return (Fr  * ( Smax  <= 0) +
                    Fmr * ((Sstar <= 0) & (Smax > 0)) +
                    Fml * ((Sstar >  0) & (Smin < 0)) +
                    Fl  * ( Smin  >  0))

            

    def _as_2D_array(self, Q):
        Q = np.array(Q)
        if len(Q.shape) == 1:
            Q.reshape(1, len(Q))
        return Q
    
    def _compute_state(self, Q):
        '''Compute the important derived quantities'''
        
        m = Q[1:-1] 
        v = m / Q[0]
        if self._iso_cs is not None:
            P = Q[0] * self._iso_cs
            cs = self._iso_cs
        else:
            u = Q[-1] - 0.5*(m*v).sum(0)
            P = u * (self.gamma - 1)
            cs = (self.gamma * P / Q[0])**0.5

        return { 'd'  : Q[0],
                 'v'  : v,
                 'P'  : P,                 
                 'cs' : cs,
                 'E'  : Q[-1],
                 'H'  : Q[-1] + P,
                 }

    def _HLL_speeds(self, Sl, Sr, jdir):
        '''Fastest / slowest wave-speed estimates'''
        R = (Sr['d'] / Sl['d'])**0.5
        fl = 1. / (1. + R)
        fr = 1. - fl
        
        vl, vr = Sl['v'][jdir], Sr['v'][jdir]
        v_av = fl*vl + fr*vr

        if self._iso_cs is not None:
            cs_av = cs_l = cs_r = self._iso_cs
        else:
            cs_l, cs_r = Sl['cs'], Sr['cs']

            dv2 = ((Sl['v'] - Sr['v'])**2).sum(0)
            cs_av = np.sqrt(fl*cs_l*cs_l + fr*cs_r*cs_r +
                            0.5*fl*fr*(self.gamma - 1)*dv2)

        Smin = np.minimum(vl - cs_l, v_av - cs_av)
        Smax = np.maximum(vr + cs_r, v_av + cs_av)

        return Smin, Smax


    def _star_speed(self, Sl, Sr, Smin, Smax, jdir):
        '''Central wave-speed estimate'''
        vl, vr = Sl['v'][jdir], Sr['v'][jdir]

        dml = Sl['d'] * (vl - Smin)
        dmr = Sr['d'] * (vr - Smax)

        if self._iso_cs is not None:
            return (Smax*dml - Smin*dmr) / (dml - dmr)
        else:
            return (vl*dml - vr*dmr + Sl['P'] - Sr['P']) / (dml - dmr)


    def _flux(self, S, Q, jdir):
        '''Euler flux'''

        flux = np.empty_like(Q)

        flux[0]  = S['d'] * S['v'][jdir]
        flux[1:-1] = flux[0] * S['v']
        flux[jdir+1] += S['P']
        flux[-1] = S['H'] * S['v'][jdir]

        return flux
        

    def _compute_wave_jump(self, S, Q, jdir, v_wave, v_star):
        '''Add the flux-jump across the outer waves via RH condition'''

        vs  = S['v'][jdir]
        dms = S['d']*(vs - v_wave)

        Qs = np.empty_like(Q)
        Qs[ 0] = Q [0]*(v_wave - vs)/(v_wave - v_star)
        Qs[1:-1]   = Qs[0]*S['v']
        Qs[jdir+1] = Qs[0]*v_star
        Qs[-1] = Qs[0]*(Q[-1]/Q[0] + (v_star - vs)*(v_star - S['P'] / dms))
        
        return v_wave * (Qs -  Q)

    @property
    def gamma(self):
        return self._gamma
    @property
    def iso_cs(self):
        return self._iso_cs

    
class _test_Riemann_Problem(object):
    def __init__(self, RSolver):
        self._RSolver = RSolver
        self._gamma = RSolver.gamma

    def _set_state(self, rho, p, v):
        gamma = self._gamma
        return np.array([rho, rho*v, 0, 0.5*rho*v*v + p/(gamma-1)])

    def _flux(self, Q):
        '''Euler flux'''
        gamma = self._gamma 
        flux = np.empty_like(Q)

        P = self._pressure(Q)
        
        flux[0]    = Q[1]
        flux[1:-1] = Q[1] * Q[1:-1]/Q[0]
        flux[1]   += P
        flux[-1]   = (Q[-1] + P)*Q[1]/Q[0]
        
        return np.array(flux)

    def _pressure(self, Q):
        gamma = self._gamma
        return (Q[-1] -  0.5*(Q[1:-1]**2).sum(0)/Q[0]) * (gamma-1)

    def test_RP(self, l, r, wave='contact'):
        '''Test the fluxes from the Riemann Solver'''
        fRP = self._RSolver(l, r)

        if wave == 'contact':
            v = l[1]
        elif wave == 'forward-shock':
            Pl, Pr = [ self._pressure(Q) for Q in [l,r]]
            gm = self._gamma 
            if Pl < Pr:
                v = l[1]/l[0] - ((gm*Pl + 0.5*(gm+1)*(Pr-Pl))/l[0])**0.5
            else:
                v = r[1]/r[0] + ((gm*Pr + 0.5*(gm+1)*(Pl-Pr))/r[0])**0.5
            
        if v > 0:
            fHydro = self._flux(l)
        else:
            fHydro = self._flux(r)
            
        test =  np.allclose(fHydro, fRP)
        if not test:
            print( 'Failed: L={}, R={}'.format(l, r))
            print( 'v0={}'.format(v))
            print( '\tflux(RS)={}'.format(fRP))
            fl, fr = [ self._flux(s) for s in [l, r]]
            print( '\tflux(L )={}\n\tflux(R )={}'.format(fl, fr))

        return test


class _test_contact_resolution(_test_Riemann_Problem):

    def __init__(self, RSolver):
        _test_Riemann_Problem.__init__(self, RSolver)

    def __call__(self):
        
        passed = True
        
        # Supersonic right
        l = self._set_state(1.0, 0.1, 10.)
        r = self._set_state(0.1, 0.1, 10.)
        passed &= self.test_RP(l, r)
        
        # Supersonic left
        l = self._set_state(1.0, 0.1, -10.)
        r = self._set_state(0.1, 0.1, -10.)
        passed &= self.test_RP(l, r)
        
        # Subsonic Right
        l = self._set_state(1  , 0.1, 0.3) ;
        r = self._set_state(0.5, 0.1, 0.3) ;
        passed &= self.test_RP(l, r)
        
        # Subsonic Left
        l = self._set_state(1  , 0.1, -0.3) ;
        r = self._set_state(0.5, 0.1, -0.3) ;
        passed &= self.test_RP(l, r)
   
        # Stationary
        l = self._set_state(1  , 0.1, 0.) ;
        r = self._set_state(0.5, 0.1, 0.) ;
        passed &= self.test_RP(l, r)

        assert(passed)



class _test_shock_resolution(_test_Riemann_Problem):
    def __init__(self, RSolver):
        _test_Riemann_Problem.__init__(self, RSolver)

    def _set_RH_RHS(self, rho, p, v):
        '''Compute the post-shock quantities for a stationary shock'''
        j = rho * v
        Ek = 0.5 * j * v

        g00 = self._gamma
        gm1 = self._gamma - 1
        gp1 = self._gamma + 1

        rho1 = rho * gp1 / (g00*p/Ek + gm1)
        v1   = j / rho1
        p1   = (4*Ek - gm1*p) / gp1

        return self._set_state(rho1, p1, v1)

    def _shift_state(self, Q, dvx):
        vx_o = Q[1] / Q[0]
        vx_n = vx_o + dvx

        Q[1]   = Q[0]*vx_n
        Q[-1] += 0.5*Q[0]*(vx_n**2 - vx_o**2)

        return Q
        
    def __call__(self):
        passed = True

        # Stationary shock
        l = self._set_state (1.0, 1.0, 10.0)
        r = self._set_RH_RHS(1.0, 1.0, 10.0)
        passed &= self.test_RP(l, r, wave='forward-shock')

        # Subsonic, left
        l, r =  [self._shift_state(Q, -5.) for Q in [l, r]]
        passed &= self.test_RP(l, r, wave='forward-shock')

        # Supersonic, left
        l, r =  [self._shift_state(Q, -5.) for Q in [l, r]]
        passed &= self.test_RP(l, r, wave='forward-shock')

        # Supersonic, Right
        l, r =  [self._shift_state(Q, +15.) for Q in [l, r]]
        passed &= self.test_RP(l, r, wave='forward-shock')
        
        assert(passed)

        
if __name__ == "__main__":

    RS = HLLC()
    _test_shock_resolution(RS)()
    _test_contact_resolution(RS)()
    
