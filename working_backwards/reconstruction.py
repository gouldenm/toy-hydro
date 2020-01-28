# reconstruction.py
#
# Author: R. Booth
# Date: 12 - Jan - 2018
#
# Contains classes for computing slope-limited reconstructions
################################################################################
from __future__ import print_function
import numpy as np
from scipy.special import erf

from FV_interpolation import compute_centroids, construct_FV_edge_weights

class DonorCell(object):
    '''First-order upwind reconstruction

    args:
        xe : Cell edge locations
        m  : Index of radial scaling giving volume, V = x^{m+1} / m+1.
    '''
    STENCIL = 1
    ORDER = 1
    def __init__(self, xe, m):
        pass

    def reconstruct(self, Q, dt=0, v_edge=None):
        '''Reconstruct the left/right states'''
        return Q, Q
        
    def __call__(self, v_edge, Q, dt=0.):
        '''Compute the upwinded face value'''
        Qp = Qm = Q
        return np.where(v_edge[1:-1] > 0, Qp[:-1], Qm[1:])

    
class VanLeer(object):
    '''Second-order upwind reconstruction with Van Leer limiting.

    Uses the geometrically consistent formulation of Mignone (2014).

    args:
        xe : Cell edge locations
        m  : Index of radial scaling giving volume, V = x^{m+1} / m+1.
    '''
    STENCIL = 2
    ORDER = 2
    def __init__(self, xe, m):

        self._xe = xe
        self._xc = xc = compute_centroids(xe, m)
        self._cF = (xc[2:] - xc[1:-1]) / (xe[2:-1] - xc[1:-1])
        self._cB = (xc[2:] - xc[1:-1]) / (xc[2:  ] - xe[2:-1])

        dx = np.diff(xe)
        self._dxp = (self._xe[1:]  - self._xc) / dx
        self._dxm = (self._xe[:-1] - self._xc) / dx

    def reconstruct(self, Q, dt=0., v_edge=None):
        '''Reconstruct the left/right states'''

        # Compute the limited slopes
        dQ = Q[1:] - Q[:-1]
        QF, QB = dQ[1:], dQ[:-1]

        cF, cB = self._cF, self._cB

        num = QF*QB*(cF*QB + cB*QF)
        den = (QB*QB + (cF + cB - 2)*QB*QF + QF*QF) + 1e-300

        dQ_lim = np.where(QB*QF > 0, num/den, 0) 

        # Reconstruct the face states
        if dt:
            Qp = Q[1:-1] + dQ_lim * (self._dxp[1:-1] - v_edge[2:-1]*dt/2.)
            Qm = Q[1:-1] + dQ_lim * (self._dxm[1:-1] - v_edge[1:-2]*dt/2.)
        else:
            Qp = Q[1:-1] + dQ_lim *  self._dxp[1:-1]
            Qm = Q[1:-1] + dQ_lim *  self._dxm[1:-1]
                                     
        return Qm, Qp

    def __call__(self, v_edge, Q, dt=0.):
        '''Compute the upwinded face value (optionally time-centered)'''

        Qm, Qp = self.reconstruct(Q, dt, v_edge)
        
        return np.where(v_edge[2:-2] > 0, Qp[:-1], Qm[1:])


class Arepo2(object):
    """Second-order reconstruction as in AREPO."""
    STENCIL = 2
    ORDER = 2
    def __init__(self, xe, m):
        if m != 0:
            raise ValueError("Arepo2 assumes m = 0")

        self._xe = xe
        self._xc = xc = compute_centroids(xe, m)
        self._dx = xc[2:] - xc[:-2]

    def _mc(self, Q):
        Qm = Q[:-2]
        Q0 = Q[1:-1]
        Qp = Q[2:]

        def mm(l, r):
            return np.where(l*r > 0, np.where(np.abs(l) < np.abs(r), l, r), 0)

        dQr = Qp - Q0
        dQl = Q0 - Qm 

        mc =  mm(0.5*(dQl + dQr), 2*mm(dQl, dQr))
        return mc

    def reconstruct(self, Q):
        """Reconstruct the left/right states"""
        Qm = Q[:-2]
        Q0 = Q[1:-1]
        Qp = Q[2:]

        Qmax = np.maximum(np.maximum(Qp, Qm), Q0)
        Qmin = np.minimum(np.minimum(Qp, Qm), Q0)

        grad = (Qp - Qm) / self._dx

        dQ = grad*(self._xe[2:-1] - self._xc[1:-1])
        Qp = Q0 + dQ

        pos = Qp > Qmax ; neg = Qp < Qmin
        phir = np.where(pos, (Qmax - Q0)/dQ, np.where(neg, (Qmin - Q0)/dQ, 1))
        
        dQ = grad*(self._xe[1:-2] - self._xc[1:-1])
        Qm = Q0 + dQ

        pos = Qm > Qmax ; neg = Qm < Qmin
        phil = np.where(pos, (Qmax - Q0)/dQ, np.where(neg, (Qmin - Q0)/dQ, 1))

        alpha = np.maximum(0, np.minimum(1, np.minimum(phir, phil)))
        grad *= alpha

        Qm = Q0 + grad*(self._xe[1:-2] - self._xc[1:-1])
        Qp = Q0 + grad*(self._xe[2:-1] - self._xc[1:-1])

        return Qm, Qp

        
        
    def __call__(self, v_edge, Q):
        """Compute the upwinded face value"""

        Qm, Qp = self.reconstruct(Q)
        
        return np.where(v_edge[2:-2] > 0, Qp[:-1], Qm[1:])




class Weno3(object):
    '''Third-order upwind WENO reconstruction.

    Uses the geometrically consistent formulation of Mignone (2014).

    args:
        xe : Cell edge locations
        m  : Index of radial scaling giving volume, V = x^{m+1} / m+1.
    '''
    STENCIL = 2
    ORDER = 3
    def __init__(self, xe, m, coeff=1.0):

        self._xe = xe
        self._xc = xc = compute_centroids(xe, m)

        dx  = np.diff(xe)
        self._dxp = (xe[1:]  - xc) / dx
        self._dxm = (xe[:-1] - xc) / dx

        self._cdx2 = (coeff * (xe[1:] - xe[:-1]))[1:-1]**2

        # Compute the linear WENO weights
        wp, wm = construct_FV_edge_weights(xe, m, 1, 1)

        self._dp0 = wp[1:-1,0,2] * np.diff(xc[1:]) / (xe[2:-1] - xc[1:-1])
        self._dm0 = wm[1:-1,0,2] * np.diff(xc[1:]) / (xe[1:-2] - xc[1:-1])
        self._dp1 = 1 - self._dp0
        self._dm1 = 1 - self._dm0

    def reconstruct(self, Q):
        '''Reconstruct the left/right states'''

        # Compute the face values using the forwards/backwards slopes
        dQ = Q[1:] - Q[:-1]
        QF, QB = dQ[1:], dQ[:-1]

        # Compute the weights:
        dQFB = (QF-QB)**2
        b0 = QF*QF
        b1 = QB*QB

        Q2 = Q*Q
        QR = self._cdx2*np.maximum(Q2[1:-1], np.maximum(Q2[2:], Q2[:-2]))

        wp0 = self._dp0*(1 + dQFB/(b0 + QR + 1e-300))
        wp1 = self._dp1*(1 + dQFB/(b1 + QR + 1e-300))

        wp0, wp1 = wp0 / (wp0 + wp1), wp1 / (wp0 + wp1)

        wm0 = self._dm0*(1 + dQFB/(b0 + QR + 1e-300))
        wm1 = self._dm1*(1 + dQFB/(b1 + QR + 1e-300))

        wm0, wm1 = wm0 / (wm0 + wm1), wm1 / (wm0 + wm1)

        # Now compute the edge states
        QFp = Q[1:-1] + QF * self._dxp[1:-1]
        QFm = Q[1:-1] + QF * self._dxm[1:-1]
        
        QBp = Q[1:-1] + QB * self._dxp[1:-1]
        QBm = Q[1:-1] + QB * self._dxm[1:-1]
        
        Qp = wp0*QFp + wp1*QBp
        Qm = wm0*QFm + wm1*QBm

        return Qm, Qp
        
    def __call__(self, v_edge, Q):
        '''Compute the upwinded face value'''

        Qm, Qp = self.reconstruct(Q)
        
        return np.where(v_edge[2:-2] > 0, Qp[:-1], Qm[1:])
    
        
def _test_scheme(Npts, IC, reconstruction, tout, a, Ca = 0.7, order=2):
    """Test schemes using an Explicit TVD RK integration"""
    # Setup up the grid
    stencil = reconstruction.STENCIL
    
    shape = Npts + 1 + 2*stencil
    dx = 1. / Npts
    xe = np.linspace(-dx*stencil, 1 + dx*stencil, shape)

    # Setup the velocity function
    v = a * np.ones_like(xe)

    # Reconstruction function:
    R = reconstruction(xe, 0)
    
    def boundary(Q):
        Q[ :stencil] = Q[Npts:Npts+stencil]
        Q[-stencil:] = Q[stencil:2*stencil]
    

    def update_stage(Q, v, dt):
        Qs = np.empty(Npts + 2*stencil)
        Qs[stencil:-stencil] = Q

        boundary(Qs)
        Qs = R(v, Qs)

        return Q - dt * np.diff(Qs*v[stencil:-stencil]) / dx


    # Set the initial conditions
    Q = IC(xe[stencil:-stencil])

    t = 0
    dtmax = Ca * dx / abs(a)
    while t < tout:
        dt = min(dtmax, tout-t)
        if order==3:
            Qs =          update_stage(Q , v, dt)
            Qs = (3*Q +   update_stage(Qs, v, dt))/4
            Q  = (  Q + 2*update_stage(Qs, v, dt))/3
        elif order == 2:
            Qs =      update_stage(Q , v, dt)
            Q  = (Q + update_stage(Qs, v, dt))/2.
        else:
            Q  = update_stage(Q , v, dt)
        t = min(tout, t+dt)

    xe = xe[stencil:-stencil]
    xc = 0.5*(xe[1:] + xe[:-1])
    return xc, Q
                


def _run_tests(IC, N=64, tol=1e9, reverse=False):
    a = 1.

    if reverse:
        a *= -1


    #### IC
    xc, Q0 = _test_scheme(10**3, IC, DonorCell, 0.,a)
    plt.plot(xc, Q0, 'k-')

    #### Donor
    xc, Q  = _test_scheme(N, IC, DonorCell, 1.,a)

    plt.plot(xc, Q, 'o', label='Donor Cell')

    # Check the symmetry:
    xc, Q1 = _test_scheme(N, IC, DonorCell, 1., -a)
    assert( (abs(1 - Q/Q1[::-1]).mean() < tol))


    #### Van Leer
    xc, Q = _test_scheme(N, IC, VanLeer, 1.,a)
    plt.plot(xc, Q, '+', label='Van Leer')
    plt.legend()
    
    # Check the symmetry:
    xc, Q1 = _test_scheme(N, IC, VanLeer, 1., -a)

    assert( abs(1 - Q/Q1[::-1]).mean() < tol)

    ### Arepo:
    xc, Q = _test_scheme(N, IC, Arepo2, 1.,a)
    plt.plot(xc, Q, '+', label='Arepo2')
    plt.legend()
    
    # Check the symmetry:
    xc, Q1 = _test_scheme(N, IC, Arepo2, 1., -a)

    assert( abs(1 - Q/Q1[::-1]).mean() < tol)

    
    #### Weno3
    xc, Q = _test_scheme(N, IC, Weno3, 1.,a)
    plt.plot(xc, Q, 'x', label='WENO-3')
    plt.legend()
    
    # Check the symmetry:
    xc, Q1 = _test_scheme(N, IC, Weno3, 1., -a)
    assert( (abs(1 - Q/Q1[::-1]).mean() < tol))

def _test_convergence(IC, pmin=4, pmax=10):

    a = 1.0 
    N = 2**np.arange(pmin, pmax+1)
    for scheme in [ VanLeer, Arepo2]:#[DonorCell, VanLeer, Arepo2, Weno3]:
        errs = []
        c=None
        label=scheme.__name__
        for Ni in N:
            _, Q0 = _test_scheme(Ni, IC, scheme, 0., a, order=scheme.STENCIL)
            x, Q = _test_scheme(Ni, IC, scheme, 1., a, order=scheme.ORDER)
            
            c = plt.plot(x,Q, c=c,label=label)[0].get_color()
            label=None
            
            errs.append(np.sqrt(np.mean((Q - Q0)**2)))

        #plt.loglog(N, errs, label=scheme.__name__)

    plt.xlabel('N')
    plt.ylabel('L2 error')
    plt.legend()

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    # Test schemes using a square wave
    def square(x):
        xu = x[1:]
        xl = x[:-1]
        dx = xu - xl

        r = np.zeros_like(xu)
        r[(xl >= 0.375) & (xu <= 0.625)] = 1.0

        args=(xu >= 0.375) & (xl <= 0.375)
        r[args] = (xu[args] - 0.375) / dx[args]
        args=(xu >= 0.625) & (xl <= 0.625)
        r[args] = (0.625 - xl[args]) / dx[args]
        
        return r

    def triangle(x):
        xu = x[1:]
        xl = x[:-1]
        dx = xu - xl

        r = np.zeros_like(xu)

        args = (xl >= 0.25) & (xu <= 0.5)
        r[args] = 2*(xu + xl)[args] - 1
        args = (xu >= 0.25) & (xl <= 0.25)
        r[args] = ((2*(xu**2 - 0.25**2) - (xu-0.25)) / (xu-xl))[args]

        args = (xl <= 0.5) & (xu >= 0.5)
        t1 = 2*(0.5**2 - xl**2) - (0.5-xl)
        t2 = 3*(xu-0.5) - 2*(xu**2 - 0.5**2)
        r[args] = ((t1 + t2)/ (xu - xl))[args]
        
        args = (xl >= 0.5) & (xu <= 0.75)
        r[args] = 3 - 2*(xu + xl)[args]
        args = (xu >= 0.75) & (xl <= 0.75)
        r[args] = ((3*(0.75 - xl) - 2*(0.75**2 - xl**2)) / (xu-xl))[args]
        
        return r

    def gauss(xe):
        x = 8*(xe-0.5)
        return (erf(x[1:]) - erf(x[:-1])) / np.diff(xe)

    def sin(xe):
        x = 2*np.pi*xe
        return -np.diff(np.cos(x)) / np.diff(x)
              
    """
    plt.figure()
    _run_tests(square)

    plt.figure()
    _run_tests(triangle)

    plt.figure()
    _run_tests(gauss)
    """
    plt.figure()
    _test_convergence(sin)

    plt.show()
    
