from __future__ import print_function
import numpy as np
from dustywave_sol2 import *
from dust_settling_sol import *
from dusty_shock_adiabatic import *
#from explicit_euler_solver import *
from exponential_euler_solver_mid import *
NHYDRO=5


def boundary_periodic(Q, shape):
    stencil=2
    Npts = shape - 2*stencil
    Qb = np.empty([shape, NHYDRO])
    Qb[stencil:-stencil] = Q
    Qb[ :stencil] = Qb[Npts:Npts+stencil]
    Qb[-stencil:] = Qb[stencil:2*stencil]
    return(Qb)


def boundary_reflecting(Q, shape):
    stencil=2
    Npts = shape - 2*stencil
    Qb = np.empty([shape, NHYDRO])
    Qb[stencil:-stencil] = Q
    Qb[0] = Qb[3]
    Qb[1] = Qb[2]
    Qb[-1] = Qb[-4]
    Qb[-2] = Qb[-3]
    
    #flip signs for velocities
    Qb[0,1] = - Qb[3,1]
    Qb[1,1] = - Qb[2,1]
    Qb[-1,1] = - Qb[-4,1]
    Qb[-2,1] = - Qb[-3,1]
            
    return(Qb)

def boundary_flow(Q, shape):
    stencil=2
    Npts = shape - 2*stencil
    Qb = np.empty([shape, NHYDRO])
    Qb[stencil:-stencil] = Q
    Qb[0] = Qb[2]
    Qb[1] = Qb[2]
    Qb[-2] = Qb[-3]
    Qb[-1] = Qb[-3]
    return(Qb)

def boundary_inflowL_and_reflectR(Q, shape):
    stencil=2
    Npts = shape - 2*stencil
    Qb = np.empty([shape, NHYDRO])
    Qb[stencil:-stencil] = Q
    #   inflow both on left
    Qb[0] = Qb[2]
    Qb[1] = Qb[2]
    
    #   inflow dust on right
    Qb[-2] = Qb[-3]
    Qb[-1] = Qb[-3]
    #   reflect gas on right
    Qb[-1,:3] = Qb[-4,:3]
    Qb[-2,:3] = Qb[-3,:3]
    Qb[-1,1] = -Qb[-4,1]
    Qb[-2,1] = -Qb[-3,1]
    return (Qb)




## ### ### TEST PROBLEMS ### ###

def _test_convergence(IC, pmin=4, pmax=9, figs_evol=None, fig_err=None, t_final=3.0, FB=1, 
                      GAMMA=5./3., K=0.1):
    N = 2**np.arange(pmin, pmax+1)
    errs_gas = []
    errs_dust = []
    c=None
    for Ni in N:
        print (Ni)
        _, W0 = solve_euler(Ni, IC, boundary_periodic, 0, Ca = 0.4, mesh_type = "fixed", 
                            FB=FB, K=K, GAMMA=GAMMA)
        x, W = solve_euler(Ni, IC, boundary_periodic, t_final, Ca = 0.4, mesh_type = "fixed", 
                           FB=FB, K=K, GAMMA=GAMMA)
        true = IC(x, K, t=t_final)
        if figs_evol is not None:
            c = figs_evol[0].plot(x, W[:,0], c=c, 
                                  label="gas")[0].get_color()
            figs_evol[0].plot(x, W[:,3], 
                                  label="dust "+ str(Ni))
            figs_evol[1].plot(x, W[:,1], c=c)
            figs_evol[1].plot(x, W[:,4], 
                                  label="dust "+ str(Ni))
            figs_evol[2].plot(x, W[:,2], c=c)
            
            figs_evol[0].set_ylabel('Density')
            figs_evol[1].set_ylabel('Velocity')
            figs_evol[2].set_ylabel('Pressure')
            figs_evol[2].set_xlabel('x')
            
            label=None
        
        err = W - true
        errs_gas.append(np.sqrt(np.mean((err[:,1])**2)))
        errs_dust.append(np.sqrt(np.mean((err[:,4])**2)))
    if fig_err is not None:
        fig_err.loglog(N, errs_gas, c=c, label=" gas", ls = "-")
        fig_err.loglog(N, errs_dust, c=c, label=" dust", ls="--")
        plt.title("K=" + str(K))

    if fig_err is not None:
        fig_err.set_xlabel('N')
        fig_err.set_ylabel('L2 velocity error')
        fig_err.plot(N, 1e-4/N**2, label='1/N^2', c='k')
        fig_err.legend()
    if figs_evol is not None:
        figs_evol[0].legend(loc='best',frameon=False)
        figs_evol[0].plot(x, true[:,0], c="black", linestyle="-", 
                                  label="gas")[0].get_color()
        figs_evol[0].plot(x, true[:,3], c="black", 
                                  label="dust", linestyle = "--")
        figs_evol[1].plot(x, true[:,1], c="black", linestyle="-", 
                                  label="gas")[0].get_color()
        figs_evol[1].plot(x, true[:,4], c="black", 
                                  label="dust", linestyle = "--")


def init_wave(xc, K, cs0=1.0, rho0=1.0, v0=1.0, drho=1e-6, t=0, dust_gas_ratio = 1.0, 
              GAMMA=5./3., FB=1, mach=0, gravity=0):
    if t == 0:
        kx = 2*np.pi*xc
    
        W = np.full([len(xc), NHYDRO], np.nan)
        #gas
        W[:,0] = rho0 + drho*np.sin(kx)
        W[:,1] = v0 + drho*cs0*np.sin(kx)
        W[:,2] = (rho0*cs0**2/GAMMA) * (W[:,0]/rho0)**GAMMA
    
        #dust
        W[:,3] = W[:,0]
        W[:,4] = W[:,1]
    
    else:
        sol = DustyWaveSolver(K=K, delta=drho, feedback=FB)(t)
        
        x = xc - v0*t
        W = np.full([len(x), NHYDRO], np.nan)
        
        W[:,0] = sol.rho_gas(x)
        W[:,1] = v0 + sol.v_gas(x)
        W[:,2] = (rho0*cs0**2/GAMMA) * (W[:,0]/rho0)**GAMMA
        
        W[:,3] = sol.rho_dust(x)
        W[:,4] = v0 + sol.v_dust(x)
    return W


def init_sod(xc, K, dust_gas_ratio = 1.0, gravity=0.0, GAMMA=5./3., FB=1.0, mach=0):
    Pl = 1.0
    Pr = 0.1
    rhol = 1.0
    rhor = 0.125
    vl = 0
    vr = 0
    
    #Change once you've messed with boundaries
    idl = (xc < 0.5) 
    
    W = np.full([len(xc), NHYDRO], np.nan)
    
    W[idl, 0] = rhol
    W[idl, 1] = vl
    W[idl, 2] = Pl
    
    W[~idl, 0] = rhor
    W[~idl, 1] = vr
    W[~idl, 2] = Pr
    
    W[:,3] = W[:,0]
    W[:,4] = W[:,1]
    return(W)


def _test_sod(Nx=256, t_final=0.1, gravity=0.0, FB=1.0):
    IC = init_sod
    
    f, subs = plt.subplots(5, 1)
    plt.suptitle("Dustyshock test")
    
    xL, WL = solve_euler(Nx, IC, boundary_flow, t_final, Ca=0.4, 
                         mesh_type = "Lagrangian", fixed_v = 0.0, FB=FB)
    xF, WF = solve_euler(Nx, IC, boundary_flow, t_final, Ca=0.4, 
                         mesh_type= "Fixed", fixed_v=0.0, FB=FB)
    xI, WI = solve_euler(Nx, IC, boundary_flow, 0.0, Ca=0.4, 
                         mesh_type = "Fixed", fixed_v=0.0, FB=FB)
    
    for i in range(0,5):
        subs[i].plot(xL, WL[:,i], c="b", label="Lagrangian")
        subs[i].plot(xF, WF[:,i], c="g", ls="--", label="Fixed")
        subs[i].plot(xI, WI[:,i], c="k", label="IC")
    
    subs[0].set_ylabel('Density')
    subs[1].set_ylabel('Velocity')
    subs[2].set_ylabel('Pressure')
    subs[3].set_ylabel('Dust Density')
    subs[4].set_ylabel('Dust Velocity')
    
    subs[4].set_xlabel('x')

    subs[0].legend(loc='best')
    
    """subs[0].set_xlim(0, 0.5)
    subs[1].set_xlim(0, 0.5)
    subs[2].set_xlim(0, 0.5)
    subs[3].set_xlim(0, 0.5)
    subs[4].set_xlim(0, 0.5)"""

def init_dustybox(xc, K, dust_gas_ratio = 1.0, gravity=0.0, GAMMA=5./3., FB=1.0, mach=0):
    W = np.full([len(xc), NHYDRO], np.nan)
    P = 1.0
    rho_g = 1.0
    rho_d = rho_g*dust_gas_ratio 
    v_d = 1.0
    v_g = 0.0
    
    W[:,0] = rho_g
    W[:,1] = v_g
    W[:,2] = P
    W[:,3] = rho_d
    W[:,4] = v_d
    
    return(W)


def _test_dustybox_time(Nx=256, t_final= 1.0, FB=1.0):
    IC = init_dustybox
    
    plt.figure()
    for ratio in [0.01, 0.1, 1., 10., 100.]:
        plt.title("Dustybox velocity over time")
        ts = []
        vLs = []
        vFs = []
        for t in np.linspace(0, t_final, 10):
            ts.append(t)
            xL, WL = solve_euler(Nx, IC, boundary_periodic, t, Ca=0.4,
                                 mesh_type = "Lagrangian",
                                 dust_gas_ratio = ratio, FB=FB)
            """xF, WF = solve_euler(Nx, IC, t, Ca=0.4,
                                 mesh_type = "fixed", b_type = "periodic",
                                 dust_gas_ratio = ratio)"""
            vL = np.mean(WL[:,4])
            #vF = np.mean(WF[:,4])
            vLs.append(vL)
            #vFs.append(vF)
        plt.plot(ts, vLs, label="Lagrangian ratio=" + str(ratio))
        #plt.plot(ts, vFs, ls="--")
        plt.xlabel("t")
        plt.ylabel("Dust velocity")
        plt.xlim(-1e-5, 2)
        plt.ylim(-1e-5, 1)
        plt.legend()


def analytical_dustybox_feedback(t, K, dust_gas_ratio=1.0):
    P = 1.0
    rho_g = 1.0
    rho_d = rho_g*dust_gas_ratio
    rho = rho_g + rho_d
    ed = rho_d / rho
    eg = rho_g / rho 
    exp = np.exp(-K*rho*t)
    
    v_d = 1.0
    v_g = 0.0
    p_d = v_d*rho_d
    p_g = v_g*rho_g
    
    p_dust_out = p_g*(ed-ed*exp) + p_d*(ed+eg*exp)
    v_dust_out = p_dust_out / rho_d
    return(v_dust_out)


def _test_dustybox_convergence(pmin = 4, pmax=10, t_final=1.0, FB=1, K=0.01):
    IC = init_dustybox
    N = 2**np.arange(pmin, pmax+1)
    
    plt.figure()
    for ratio in [0.01, 0.1, 1., 10.]:
        true = analytical_dustybox_feedback(t_final, K, ratio)
        plt.title("Dustybox velocity error convergence")
        vLs = []
        vFs = []
        for Ni in N:
            xL, WL = solve_euler(Ni, IC, boundary_periodic, t_final, Ca=0.4,
                                 mesh_type = "fixed",
                                 dust_gas_ratio = ratio, FB=1, K=K)
            vL = true - np.mean(WL[:,4])
            if vL == 0:
                vL = 1e-20
            vLs.append(vL)
        plt.plot(N, vLs, label="Lagrangian ratio=" + str(ratio))
        plt.xlabel("N")
        plt.ylabel("Dust velocity error")
    plt.plot(N, 1./N, label="1/N", ls="--")
    plt.plot(N, 1./N**2, label="1/N^2", ls="--")
    plt.yscale("log")
    plt.xscale("log")
    plt.legend()


def init_const_gravity(xc, K, dust_gas_ratio=1.0, gravity=-1.0, GAMMA=5./3., FB=0, mach=0):
    W = np.full([len(xc), NHYDRO], np.nan)
    H = (gravity*GAMMA)
    P_0 = 1.0/GAMMA
    rho_0 = P_0*GAMMA
    
    rho_g = rho_0*np.exp(xc*H)
    P = P_0*np.exp(xc*H)
    v_g=0
    v_d=0
    #insert rho dust in as a constant, since gas is in equilibrium already.
    rho_d = rho_0*np.exp(xc*H)[-1]*dust_gas_ratio
    
    W[:,0] = rho_g
    W[:,1] = v_g
    W[:,2] = P
    W[:,3] = rho_d
    W[:,4] = v_d
    
    return(W)


def _test_const_gravity(t_final=1.0, Nx=256, gravity=-1.0, dust_gas_ratio=1.0, Ca=0.4, 
                        GAMMA=5./3., FB=0, K=10.0):
    print(FB)
    IC = init_const_gravity
    f, subs = plt.subplots(5, 1)
    
    xI, WI = solve_euler(Nx, IC, boundary_reflecting, 0, Ca=Ca,
                         mesh_type="fixed", K=K,
                         dust_gas_ratio= dust_gas_ratio, gravity = gravity, FB=0)
    
    for t in [2.0, 3.0, 5.0, 10.0]:
        x, W = solve_euler(Nx, IC, boundary_reflecting, t, Ca=Ca, 
                           mesh_type = "fixed", K=K,
                           dust_gas_ratio = dust_gas_ratio, gravity=gravity, FB=0)
        for i in range(0,5):
            subs[i].plot(x, W[:,i], label=str(t))
        
        #   Compute terminal velocity of dust
        H = (gravity*GAMMA)
        x_true, v_true = dust_settling_sol(H, v_0=W[-1,4], K=K)
        #subs[4].plot(xI, v_terminal, c = "k", ls="--", label="terminal velocity")
        subs[4].plot(x_true, v_true, c="k", ls="--", label="analytical solution")
    
    #   Plot IC 
    #for i in range(0,5):
    #    subs[i].plot(xI, WI[:,i], c="k", label="IC")
        
    
    subs[0].set_ylabel('Density')
    subs[1].set_ylabel('Velocity')
    subs[2].set_ylabel('Pressure')
    subs[3].set_ylabel('Dust Density')
    subs[4].set_ylabel('Dust Velocity')
    
    subs[4].set_xlabel('x')

    subs[0].legend(loc='best')




def init_dusty_shock_Jtype(xc, K, dust_gas_ratio = 1.0, gravity=0.0, GAMMA=1.0001, FB=1.0, mach=1.1):
    extent = xc[-1] - xc[0]
    Nx = len(xc)
    halfNx = int(Nx/2)
    true = shock(mach, dust_gas_ratio, {'drag_type':'power_law', 'drag_const':1.0}, extent/2., halfNx, GAMMA, 1.0,
                     t=0, FB=FB, Kin=K, offset=extent)
    M = mach
    
    P = 1.0
    rho = 1.0
    rho_d = dust_gas_ratio*rho
    
    c_s = np.sqrt(GAMMA*P/rho)
    v_s = c_s*M
    
    A = (1+FB*dust_gas_ratio)* (GAMMA+1) / (2*GAMMA)
    B = - ( v_s*(1+FB*dust_gas_ratio) + P / (rho*v_s))
    C = (GAMMA-1)/(2*GAMMA) *(v_s**2 * (1+FB*dust_gas_ratio) ) + P/rho
    
    v_post = (-B - np.sqrt(B**2 - 4*A*C)) / (2*A)
    
    
    dv = v_s - v_post
    
    W = np.full([len(xc), NHYDRO], np.nan)
    W[:, 0] = rho
    W[:, 1] = dv
    W[:, 2] = P
    W[:, 3] = rho_d
    W[:, 4] = dv
    
    """
    W[halfNx:, 0] = true["rhog"]
    W[halfNx:, 1] = true["wg"]
    W[halfNx:, 2] = true["P"]
    W[halfNx:, 3] = true["rhod"]
    W[halfNx:, 4] = true["wd"]"""
    return(W)


def _test_dusty_shocks_mach(t_final=5.0, Nx=200, Ca=0.2, FB = 1.0, K=1000., D = 1.0, GAMMA=7./5., extent=30):
    machs=  [10]#, 20, 40]#, 5, 10, 20, 50]#, 5, 6, 7, 8, 10.]
    times = [ t_final*10./m for m in machs ]
    #times = [10 for _ in machs]
    for i in range(0, len(machs)):
        mach = machs[i]
        print(mach)
        t_final = times[i]
        x, W = solve_euler(Nx, init_dusty_shock_Jtype, boundary_inflowL_and_reflectR, t_final, Ca=Ca,
                           mesh_type = "Lagrangian",
                           dust_gas_ratio = D, GAMMA=GAMMA, xend=extent, 
                           FB=FB, K=K, mach=mach)
        
        f, subs = plt.subplots(3, 1, sharex=True)
        subs[0].plot(x, W[:,1], c="r", label="Gas")
        subs[0].plot(x, W[:,4], c="k", label="Dust")
        
        
        true = shock(mach, D, {'drag_type':'power_law', 'drag_const':1.0}, 10., 1000., GAMMA, 1.0,
                     t=t_final, FB=FB, Kin=K, offset=extent - 0.04)
        
        subs[0].plot(true["xi"], true["wd"], c="gray", ls="--", label="True Dust")
        subs[0].plot(true["xi"], true["wg"], c="pink", ls="--", label="True Gas" )
        
        subs[0].set_ylabel("v")
        subs[0].legend(loc="best")
        f.suptitle("J-type shock, t=" + str(t_final) + ", M=" + str(mach))
       
        subs[1].plot(x, W[:,0], c="r", label="Gas")
        subs[1].plot(x, W[:,3], c="k", label="Dust")
       
        subs[1].plot(true["xi"], true["rhog"], c="pink", ls="--", label="True Gas")
        subs[1].plot(true["xi"], true["rhod"], c="gray", ls="--", label="True Dust")
        subs[1].set_ylabel("rho")
        
        subs[2].plot(true['xi'], true['P'], c="pink", ls="--", label="True Pressure")
        subs[2].plot(x, W[:,2], c="red", label="Pressure")
        subs[2].set_ylabel("P")
        subs[2].set_xlabel("pos")
        #plt.legend(loc="best")


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    for K in [0.1, 1.0, 10.0, 100.0]:
        _test_convergence(init_wave, figs_evol=plt.subplots(3, 1)[1], fig_err=plt.subplots(1)[1],
                          t_final = 1.0, FB=1, GAMMA=1.5, K=K)
    
    #_test_sod(t_final=0.2, Nx=569)
    
    #_test_dustybox_time(Nx=256, t_final= 2.0)
    
    #_test_dustybox_convergence(t_final=0.5)
    
    #_test_const_gravity()
    
    #for t in [2.5]:
    #    _test_dusty_shocks_mach(t_final=t, D=0.5, K=3., Nx=500, FB=1, GAMMA=7./5, extent=40)
    plt.show()



