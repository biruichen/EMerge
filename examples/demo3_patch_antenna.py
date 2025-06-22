import emerge as em
import numpy as np
from emerge.pyvista import PVDisplay
import matplotlib.pyplot as plt

mm = 0.001
margin = 5*mm
Nmodes = 1
f1 = 0.5e9
f2 = 3e9

Wpatch = 53*mm
Lpatch = 52*mm
wline = 3.2*mm
wstub = 7*mm
lstub = 15.5*mm
wsub = 100*mm
hsub = 100*mm
th = 1.524*mm

Hair = 40*mm

er = 3.38

f1 = 1.54e9
f2 = 1.6e9

with em.Simulation3D('MySimulation', PVDisplay, loglevel='DEBUG') as model:
    dielectric = em.geo.Box(wsub, hsub, th, position=(-wsub/2, -hsub/2, -th))

    air = em.geo.Box(wsub, hsub, Hair, position=(-wsub/2, -hsub/2, 0))
    
    rpatch = em.geo.XYPlate(Wpatch, Lpatch, position=(-Wpatch/2, -Lpatch/2, 0))
    
    cutout1 = em.geo.XYPlate(wstub, lstub, position=(-wline/2-wstub, -Lpatch/2, 0))
    cutout2 = em.geo.XYPlate(wstub, lstub, position=(wline/2, -Lpatch/2, 0))

    line = em.geo.XYPlate(wline, lstub, position=(-wline/2, -Lpatch/2, 0))

    port = em.geo.Plate(np.array([-wline/2, -Lpatch/2, -th]), np.array([wline, 0, 0]), np.array([0, 0, th]))

    rpatch = em.geo.remove(rpatch, cutout1)
    rpatch = em.geo.remove(rpatch, cutout2)
    rpatch = em.geo.add(rpatch, line)
    
    rpatch.material = em.COPPER # Only for viewing

    dielectric.material = em.Material(er, tand=0.0, color=(0.0, 0.5, 0.0), opacity=0.6)
    model.physics.resolution = 0.2
    
    model.physics.set_frequency_range(1.5e9, 1.7e9, 21)

    model.define_geometry([dielectric, air, rpatch, port])

    model.mesher.set_boundary_size(rpatch, 5*mm, 1.1)
    model.mesher.set_boundary_size(port, 0.5*mm, 1.1)

    model.generate_mesh()
    
    #model.view(selections=[port,], use_gmsh=False)

    port = em.bc.LumpedPort(port, 1, width=wline, height=th, direction=em.ZAX, active=True, Z0=50)

    boundary_selection = air.outside('bottom')
    
    abc = em.bc.AbsorbingBoundary(boundary_selection)
    
    pec = em.bc.PEC(rpatch)

    model.physics.assign(port, pec, abc)
    model.physics.solveroutine.direct_solver = em.solver.SolverSuperLU()
    data = model.physics.frequency_domain()

    xs, ys, zs = em.YAX.pair(em.ZAX).span(wsub, Hair, 31, (0, -wsub/2, -th))

    freqs, S11 = data.ax('freq').S(1,1)
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(freqs/1e9, 20*np.log10(np.abs(S11)))
    ax.set_xlabel('Frequency (GHz)')
    ax.set_ylabel('S-parameter (dB)')
    ax.grid(True)
    plt.show()
    
    topsurf = model.mesh.boundary_surface(boundary_selection.tags, (0,0,0))

    Ein, Hin = data.item(0).interpolate(*topsurf.exyz).EH
    
    theta = np.linspace(-np.pi, 1*np.pi, 201)
    phi = 0*theta
    E, H = em.physics.edm.stratton_chu(Ein, Hin, topsurf, theta, phi, data.item(0).k0)
    
    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.plot(theta, em.norm(E))
    plt.show()
    
    ### Create a polar farfield plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='polar')
    ax.set_theta_zero_location('N')

    ax.plot(theta, em.norm(E), label='E-field', color='blue')
    plt.show()
