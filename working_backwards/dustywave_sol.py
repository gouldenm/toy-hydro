
import numpy as np
from scipy.integrate import ode

class _DustyWaveSolution(object):
    def __init__(self, t, k,
                 rho_g, rho_d, P0, drho_g, drho_d, v_g, v_d, dP, v_f):
        self._t = t
        self._k = k
        self._rho_g = rho_g
        self._rho_d = rho_d
        self._drho_g = drho_g
        self._drho_d = drho_d
        self._v_g = v_g
        self._v_d = v_d
        self._P0 = P0
        self._dP = dP
        self._v_f = v_f


    def v_gas(self, x):
        return (self._v_f + self._v_g * np.exp(1j*self._k*x)).real
    def v_dust(self, x):
        return (self._v_f + self._v_d * np.exp(1j*self._k*x)).real

    def rho_gas(self, x):
        return (self._rho_g + self._drho_g * np.exp(1j*self._k*x)).real
    def rho_dust(self, x):
        return (self._rho_d + self._drho_d * np.exp(1j*self._k*x)).real

    def P(self, x):
        return (self._P0 + self._dP * np.exp(1j*self._k*x)).real

    @property
    def time(self):
        return self._t

class DustyWaveSolver(object):

    def __init__(self, rho_g=1., rho_d=1., cs=1., K=1., delta=1e-3, vf=0,
                 GAMMA=1.0, wavelength=1., feedback=True):
        
        self.GAMMA = GAMMA

        self.vf = vf
        self.rho_g = rho_g
        self.rho_d = rho_d
        self.cs = cs
        self.K = K
        self.delta = delta
        self.wavelength=wavelength
        self.feedback=feedback

    def _solve_system(self, times):
        '''Solve the dusty-wave problem up to specified times'''
        k = 2*np.pi / self.wavelength
        cs2 = self.cs**2
        gammaP0 = cs2 * self.rho_g 
        gamma = self.GAMMA

        ts_inv = self.K * self.rho_g
        if self.feedback:
            Kg = (self.rho_d/self.rho_g) * ts_inv
            Kd = ts_inv
        else:
            Kg = 0
            Kd = ts_inv

        def f(t, y,_=None):
            drho_g, v_g = y[0:2]
            drho_d, v_d = y[2:4]
            dP = y[4]
            
            dydt = np.empty([5], dtype='c8')
            dydt[0] = - 1j*k*self.vf*drho_g - 1j*k*self.rho_g * v_g 
            dydt[1] = - 1j*k*self.vf * v_g - Kg*(v_g - v_d) - 1j*k*dP/self.rho_g
            dydt[2] = - 1j*k*self.rho_d * v_d - 1j*k*self.vf*drho_d
            dydt[3] = - 1j*k*self.vf * v_d + Kd*(v_g - v_d)
            dydt[4] = - 1j*k*gammaP0 * v_g - 1j*k*self.vf*dP
            return dydt

        _jac = np.zeros([5,5], dtype='c8')
        _jac[0,:] = [-1j*k*self.vf, -1j*k*self.rho_g, 0, 0, 0]
        _jac[1,:] = [0, -Kg - 1j*k*self.vf, 0,  Kg, -1j*k/self.rho_g]
        _jac[2,:] = [0, 0, -1j*k*self.vf, -1j*k*self.rho_d, 0]
        _jac[3,:] = [0,  Kd, 0, -Kd - 1j*k*self.vf,      0         ]
        _jac[4,:] = [0, - 1j*k*gammaP0, 0, 0, -1j*k*self.vf]
        def jac(t, y,_=None):
            return _jac

        # Do the right going part of the wave (we can get the left going wave
        # for free by taking the real part).
        e = -1j * self.delta

        IC = np.array([self.rho_g*e, self.cs*e, 
                       self.rho_d*e, self.cs*e,
                       gammaP0*e],
                      dtype='c8')

        # Diagonalize the equations
        l, U = np.linalg.eig(_jac)
        u0 = np.linalg.solve(U, IC)

        sol = []
        for ti in times:
            # Use the analytical solution and project back to 
            # the primitive variables
            sol.append(np.dot(U, u0 * np.exp(l*ti)))

        return np.array(sol)

    def __call__(self, time):
        '''Solve the dustwave problem at a given time or list of times'''
        try:
            iter(time)
        except TypeError:
            time = [time,]
        time = sorted(time)
        sol = self._solve_system(list(time))

        def _create_solution(t, drho_g, v_g, drho_d, v_d, dP):
            return _DustyWaveSolution(t, 2*np.pi/self.wavelength,
                                      self.rho_g, self.rho_d, 
                                      self.rho_g*self.cs**2/self.GAMMA,
                                      drho_g, drho_d, v_g, v_d, dP, self.vf)

        sol = [_create_solution(t, *s) for t, s in zip(time,sol)]
        if len(sol) == 1:
            return sol[0]
        else:
            return sol


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    
    dw = DustyWaveSolver(vf=0, GAMMA=5/3., K=10)
    
    N = 101
    times = np.linspace(0, 10, N)
    x     = np.linspace(0, 1, 1001)
    
    sol = dw(times)

    f, ax = plt.subplots(2,1)
    l1, = ax[0].plot(x, sol[0].rho_gas(x)-1,  'k-')     
    l2, = ax[0].plot(x, sol[0].rho_dust(x)-1, 'k:')
    l3, = ax[1].plot(x, sol[0].v_gas(x),  'k-', label='gas') 
    l4, = ax[1].plot(x, sol[0].v_dust(x), 'k:', label='dust')
    plt.legend(frameon=False)


    ax[0].set_ylabel(r'$\delta \rho$')
    #ax[0].set_ylim(-0.002,0.002)

    ax[1].set_xlabel(r'$x$')
    ax[1].set_ylabel(r'$v$')
    #ax[1].set_ylim(-0.001,0.001)

    def animate(i):
        l1.set_ydata(sol[i].rho_gas(x)-1)
        l2.set_ydata(sol[i].rho_dust(x)-1)
        l3.set_ydata(sol[i].v_gas(x))
        l4.set_ydata(sol[i].v_dust(x))
        return l1,l2,l3,l4

    def init():
        return l1,l2,l3,l4

    ani = animation.FuncAnimation(f, animate, np.arange(0, N), 
                                  interval=100, repeat_delay=200, blit=True)


    plt.show()
    
