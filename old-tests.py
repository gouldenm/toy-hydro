
"""
grid = mesh(250, 0.2, 1.0,  fixed_v = 2.0)
grid.setup(boundary = "flow", IC = "LRsplit", vL = 2, vR = 2)
grid.solve()
plt.plot(grid.x-0.4, grid.W[:,0] , label="w="+str(grid.v[0]) )
plt.legend()
plt.pause(0.5)



grid = mesh(500, 0.2, 2.0,  fixed_v = -1)
grid.setup(boundary = "flow", IC = "LRsplit",cutoff=0.25)
grid.solve()
plt.plot(grid.x, grid.W[:,0] , label="w="+str(grid.v[0]) )
plt.legend()
plt.pause(0.5)


plt.xlabel("Position")
plt.ylabel("Pressure")
plt.pause(0.5)



t=0.2

plt.figure()

gridsound = mesh(1000,t, 1.0, mesh_type="Lagrangian")
gridsound.setup(vB=-1)
gridsound.solve()
plt.plot(gridsound.x, gridsound.W[:,0], label="Lagrangian")
plt.legend()
plt.pause(0.5)

gridsound = mesh(1000,t, 1.0, mesh_type="Fixed")
gridsound.setup(vB=-1)
gridsound.solve()
plt.plot(gridsound.x, gridsound.W[:,0], label="Eulerian")
plt.legend()
plt.pause(0.5)


#Relative motion = 0, should be identical
plt.figure()
gridsound = mesh(1000, t, 1.0, fixed_v = 0)
gridsound.setup(vB=0)
gridsound.solve()
plt.plot(gridsound.x, gridsound.W[:,0], label="vB=0, w="+str(gridsound.v[0]))
plt.legend()
plt.pause(0.5)

gridsound = mesh(500, t, 1.0, fixed_v = 1)
gridsound.setup(vB=1)
gridsound.solve()
plt.plot(gridsound.x, gridsound.W[:,0], label="vB=1, w="+str(gridsound.v[0]))
plt.legend()
plt.pause(0.5)
"""




plt.figure()

#Relative motion = sound speed, should match initial conditions
gridsound = mesh(500, 0.05, 1.0, fixed_v = 1.0, K=100.0)
gridsound.setup(drhod=0, l=1.0)
gridsound.solve()
plt.plot(gridsound.x, gridsound.W[:,0], label="Gas, w=c_s", color="k")
plt.plot(gridsound.x, gridsound.W[:,3], label="dust t=0.05")
plt.legend()
plt.pause(1.0)

gridsound.tend = 0.1
gridsound.solve()
plt.plot(gridsound.x, gridsound.W[:,3], label="dust t=0.1")
plt.legend()
plt.pause(1.0)

gridsound.tend = 0.3
gridsound.solve()
plt.plot(gridsound.x, gridsound.W[:,3], label="dust t=0.3")
plt.legend()
plt.pause(1.0)

gridsound.tend = 0.5
gridsound.solve()
plt.plot(gridsound.x, gridsound.W[:,3], label="dust t=0.5")
plt.legend()
plt.pause(1.0)

gridsound.tend = 0.75
gridsound.solve()
plt.plot(gridsound.x, gridsound.W[:,3], label="dust t=0.75s")
plt.legend()
plt.pause(0.5)

gridsound.tend = 1.0
gridsound.solve()
plt.plot(gridsound.x, gridsound.W[:,3], label="dust t=1.0s")
plt.legend()
plt.pause(0.5)

gridsound.tend = 1.5
gridsound.solve()
plt.plot(gridsound.x, gridsound.W[:,3], label="dust t=1.5")
plt.legend()
plt.pause(0.5)

gridsound.tend = 2.0
gridsound.solve()
plt.plot(gridsound.x, gridsound.W[:,3], label="dust t=2.0")
plt.legend()
plt.pause(0.5)

gridsound.tend = 5.0
gridsound.solve()
plt.plot(gridsound.x, gridsound.W[:,3], label="dust t=5.0")
plt.legend()
plt.pause(0.5)

gridsound.tend = 10.0
gridsound.solve()
plt.plot(gridsound.x, gridsound.W[:,3], label="dust t=10.0")
plt.legend()
plt.pause(0.5)

gridsound.tend = 15.0
gridsound.solve()
plt.plot(gridsound.x, gridsound.W[:,3], label="dust t=15.0")
plt.legend()
plt.pause(0.5)




"""
gridsound = mesh(500, t, 1.0, fixed_v = 0)
gridsound.setup(vB=-1)
gridsound.solve()
plt.plot(gridsound.x, gridsound.W[:,0], label="vB=-1, w="+str(gridsound.v[0]))
plt.legend()
plt.pause(0.5)

#Random faster speed for comparison
gridsound = mesh(500, t, 1.0, fixed_v = 2.0)
gridsound.setup()
gridsound.solve()
plt.plot(gridsound.x, gridsound.W[:,0], label="w="+str(gridsound.v[0]))
plt.legend()
plt.pause(0.5)



plt.figure()

gridsound = mesh(2000, 1.0, 1.0, fixed_v = 0.0)
gridsound.setup()
gridsound.solve()
plt.plot(gridsound.x, gridsound.W[:,2], label="w="+str(gridsound.v[0]))
plt.legend()
plt.pause(0.5)

gridsound = mesh(2000, 1.0, 1.0, fixed_v = 1.0)
gridsound.setup(vB=1.0)
gridsound.solve()
plt.plot(gridsound.x, gridsound.W[:,2], label="w="+str(gridsound.v[0]))
plt.legend()
plt.pause(0.5)

gridsound = mesh(2000, 1.0, 1.0, fixed_v = 1.0)
gridsound.setup()
gridsound.solve()
plt.plot(gridsound.x, gridsound.W[:,2], label="w="+str(gridsound.v[0]))
plt.legend()
plt.pause(0.5)

gridsound = mesh(2000, 1.0, 1.0, fixed_v = 2.0)
gridsound.setup()
gridsound.solve()
plt.plot(gridsound.x, gridsound.W[:,2], label="w="+str(gridsound.v[0]))
plt.legend()
plt.pause(0.5)

gridsound = mesh(2000, 1.0, 1.0, fixed_v = 2.0)
gridsound.setup()
gridsound.solve()
plt.plot(gridsound.x, gridsound.W[:,2], label="w="+str(gridsound.v[0]))
plt.legend()
plt.pause(0.5)
"""