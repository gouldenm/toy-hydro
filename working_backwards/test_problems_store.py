


def _test_convergence(IC, pmin=4, pmax=10, figs_evol=None, fig_err=None, t_final=3.0, FB=1):
    N = 2**np.arange(pmin, pmax+1, GAMMA=5./3.)
    scheme = Arepo2
    errs_gas = []
    errs_dust = []
    c=None
    label=scheme.__name__
    for Ni in N:
        print (scheme.__name__, Ni)
        _, W0 = solve_euler(Ni, IC, 0, Ca = 0.4, mesh_type = "Lagrangian",  fixed_v = 1.0, FB=FB)
        x, W = solve_euler(Ni, IC, t_final, Ca = 0.4, mesh_type = "Lagrangian", fixed_v = 1.0, FB=FB)
        true = IC(x, t=t_final)
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
        fig_err.loglog(N, errs_gas, c=c, label=scheme.__name__ + " gas", ls = "-")
        fig_err.loglog(N, errs_dust, c=c, label=scheme.__name__ + " dust", ls="--")

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


def init_wave(xc, cs0=1.0, rho0=1.0, v0=1.0, drho=1e-6, t=0, dust_gas_ratio = 1.0, GAMMA=5./3., FB=1):
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


def init_sod(xc, dust_gas_ratio = 1.0, gravity=0.0, GAMMA=5./3., FB=1.0):
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
    
    xL, WL = solve_euler(Nx, IC, t_final, Ca=0.4, 
                         mesh_type = "Lagrangian", fixed_v = 0.0, b_type="flow", FB=FB)
    xF, WF = solve_euler(Nx, IC, t_final, Ca=0.4, 
                         mesh_type= "Fixed", fixed_v=0.0, b_type="flow", FB=FB)
    xI, WI = solve_euler(Nx, IC, 0.0, Ca=0.4, 
                         mesh_type = "Fixed", fixed_v=0.0, b_type="flow", FB=FB)
    
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

def init_dustybox(xc, dust_gas_ratio = 1.0, gravity=0.0, GAMMA=5./3., FB=1.0):
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
            xL, WL = solve_euler(Nx, IC, t, Ca=0.4,
                                 mesh_type = "Lagrangian", b_type = "periodic",
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


def analytical_dustybox_feedback(t, dust_gas_ratio=1.0):
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
    print(v_dust_out)
    return(v_dust_out)


def _test_dustybox_convergence(pmin = 4, pmax=10, t_final=1.0, FB=1):
    IC = init_dustybox
    N = 2**np.arange(pmin, pmax+1)
    
    plt.figure()
    for ratio in [0.01, 0.1, 1., 10.]:
        true = analytical_dustybox_feedback(t_final, ratio)
        plt.title("Dustybox velocity error convergence")
        vLs = []
        vFs = []
        for Ni in N:
            xL, WL = solve_euler(Ni, IC, t_final, Ca=0.4,
                                 mesh_type = "Lagrangian", b_type = "periodic",
                                 dust_gas_ratio = ratio, FB=1)
            vL = true - np.mean(WL[:,4])
            if vL == 0:
                vL = 1e-20
            vLs.append(vL)
            print(vL)
        plt.plot(N, vLs, label="Lagrangian ratio=" + str(ratio))
        plt.xlabel("N")
        plt.ylabel("Dust velocity error")
        plt.legend()
    plt.yscale("log")
    plt.xscale("log")


def init_const_gravity(xc, dust_gas_ratio=1.0, gravity=-1.0, GAMMA=5./3., FB=0):
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


def _test_const_gravity(t_final=1.0, Nx=256, gravity=-1.0, dust_gas_ratio=1.0, Ca=0.4, GAMMA=5./3., FB=0):
    IC = init_const_gravity
    f, subs = plt.subplots(5, 1)
    
    xI, WI = solve_euler(Nx, IC, 0, Ca=Ca,
                         mesh_type="fixed", b_type = "reflecting",
                         dust_gas_ratio= dust_gas_ratio, gravity = gravity, FB=0)
    
    for t in [2.0, 3.0, 5.0, 10.0]:
        x, W = solve_euler(Nx, IC, t, Ca=Ca, 
                           mesh_type = "fixed", b_type = "reflecting",
                           dust_gas_ratio = dust_gas_ratio, gravity=gravity, FB=0)
        for i in range(0,5):
            subs[i].plot(x, W[:,i], label=str(t))
        
        #   Compute terminal velocity of dust
        H = (gravity*GAMMA)
        x_true, v_true = dust_settling_sol(H, v_0=W[-1,4])
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















def _test_dusty_shocks(t_final=1.0, Nx=500, Ca=0.2, FB = 1.0, K=10.):
    linestyles = [":", "--", "-"]
    D = [0.5, 1.0]
    offsets = [9.78, 9.85]
    """for i in range(0,2):
        plt.figure()
        x, W = solve_euler(Nx, init_dusty_shock_Ctype, t_final, Ca=Ca,
                           mesh_type = "fixed", b_type = "inflowL_and_reflectR", dust_reflect = True,
                           dust_gas_ratio = D[i], GAMMA=1.00001, xend=10.0, 
                           FB=FB, K=K, mach=0.96)
        
        plt.plot(x, W[:,1], c="r", label="Gas; D=" + str(D[i]))
        plt.plot(x, W[:,4], c="k", label="Dust; D=" + str(D[i]))
        
        
        true = shock(0.96, D[i], {'drag_type':'power_law', 'drag_const':1.0}, 10., 1000., 
                     t=t_final, FB=FB, Kin=K, offset= offsets[i])
        plt.plot(true["xi"], true["wd"], c="gray", ls="--", label="True Dust; D=" + str(D[i]))
        plt.plot(true["xi"], true["wg"], c="pink", ls="--", label="True Gas; D=" + str(D[i]))
        
        plt.xlabel("pos")
        plt.ylabel("v")
        plt.title("C-type shock, t=" + str(t_final) + ", D=" + str(D[i]))
        plt.legend(loc="best")
        
        plt.figure()
        plt.title("C-type shock density, t=" + str(t_final) + ", D=" + str(D[i]))
       
        plt.plot(x, W[:,0], c="r", label="Gas density" + str(D[i]))
        plt.plot(x, W[:,3], c="k", label="Dust density" + str(D[i]))
       
        plt.plot(true["xi"], true["rhog"], c="pink", ls="--", label="True Gas; D=" + str(D[i]))
        plt.plot(true["xi"], true["rhod"], c="gray", ls="--", label="True Dust; D=" + str(D[i]))
        plt.xlabel("pos")
        plt.ylabel("rho")
        plt.legend(loc="best")
    """
    D = [0.01, 0.1, 1.0]
    offsets = [10.0, 10., 9.98]
    for i in range(0,3):
        plt.figure()
        x, W = solve_euler(Nx, init_dusty_shock_Jtype, t_final, Ca=Ca,
                           mesh_type = "fixed", b_type = "inflowL_and_reflectR",
                           dust_gas_ratio = D[i], GAMMA=1.00001, xend=10.0, 
                           FB=FB, K=K, mach=1.5)
        
        plt.plot(x, W[:,1], c="r", label="Gas; D=" + str(D[i]))
        plt.plot(x, W[:,4], c="k", label="Dust; D=" + str(D[i]))
        
        
        true = shock(1.5, D[i], {'drag_type':'power_law', 'drag_const':1.0}, 20., 1000., 
                     t=t_final, FB=FB, Kin=K, offset=offsets[i])
        plt.plot(true["xi"], true["wd"], c="gray", ls="--", label="True Dust; D=" + str(D[i]))
        plt.plot(true["xi"], true["wg"], c="pink", ls="--", label="True Gas; D=" + str(D[i]))
        
        plt.xlabel("pos")
        plt.ylabel("v")
        plt.title("J-type shock, t=" + str(t_final))
        plt.legend(loc="best")
       
       
        plt.figure()
        plt.title("J-type shock density, t=" + str(t_final) + ", D=" + str(D[i]))
       
        plt.plot(x, W[:,0], c="r", label="Gas density" + str(D[i]))
        plt.plot(x, W[:,3], c="k", label="Dust density" + str(D[i]))
       
        plt.plot(true["xi"], true["rhog"], c="pink", ls="--", label="True Gas; D=" + str(D[i]))
        plt.plot(true["xi"], true["rhod"], c="gray", ls="--", label="True Dust; D=" + str(D[i]))
        plt.xlabel("pos")
        plt.ylabel("rho")
        plt.legend(loc="best")

