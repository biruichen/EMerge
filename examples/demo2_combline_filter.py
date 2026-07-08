import emerge as em
import numpy as np
from emerge.plot.pyvista import PVDisplay
import matplotlib.pyplot as plt
""" DEMO: COMBLINE FILTER

In this demo we will look at the design of a combline filter in EMerge. The filter design was taken from
the book "Modern RF and Microwave Filter Design" by Protap Pramanick and Prakash Bhartia.
Some of the dimensions where not clear.

In this demo we will look at the Modeler class and how it can help us quickly create complicated geometries.

"""

# First we define some quantities for our simulation.
mm = 0.001
mil = 0.0254*mm

a = 240*mil
b = 248*mil
d1 = 10*mil
d2 = 10*mil
dc = 8.5*mil
lr1 = b-d1
lr2 = b-d2
W = 84*mil
S1 = 117*mil
S2 = 136*mil
C1 = b-dc
h = 74*mil
wi = 84*mil
Lbox = 5*W + 2*(S1+S2+wi)

x1 = wi+W/2
x2 = x1 + W + S1
x3 = x2 + W + S2
x4 = x3 + W + S2
x5 = x4 + W + S1

rout = 40.5*mil
rin = 12.5*mil
lfeed = 100*mil

# A usual we start our simulation file
with em.Simulation3D('Combline_DEMO', PVDisplay, loglevel='DEBUG') as m:

    # The filter consists of quarter lamba cylindrical pins inside an airbox.
    # First we create the airbox
    box = em.geo.Box(Lbox, a, b, position=(0,-a/2,0))


    # Next we create 5 cyllinders using the Modeler class.
    # The modeler class also implements a method chaining interface. In this example we stick to simpler features.
    # The modeler class allows us to create a parameter series using the modeler.series() method. We provid it with quantities.
    # We can do this for multiple at the same time (as you can also see with the position). The modeler class
    # will recognize the multiple quantities and simply create 5 different cyllinders, one for each parameter pair.
    stubs = m.modeler.cyllinder(W/2, m.modeler.series(C1, lr1, lr2, lr1, C1), position=(m.modeler.series(x1, x2, x3, x4, x5), 0, 0), NPoly=10)

    # Next we create the in and output feed cyllinders for the coaxial cable. We will use the Nsections feature in order to guarantee a better
    # adherence to the boundary.
    feed1out = em.geo.Cyllinder(rout, lfeed, em.CoordinateSystem(em.ZAX, em.YAX, em.XAX, np.array([-lfeed, 0, h])), Nsections=12)
    feed1in = em.geo.Cyllinder(rin, lfeed+wi+W/2, em.CoordinateSystem(em.ZAX, em.YAX, em.XAX, np.array([-lfeed, 0, h])), Nsections=8)
    feed2out = em.geo.Cyllinder(rout, lfeed, em.CoordinateSystem(em.ZAX, em.YAX, em.XAX, np.array([Lbox, 0, h])), Nsections=12)
    feed2in = em.geo.Cyllinder(rin, lfeed+wi+W/2, em.CoordinateSystem(em.ZAX, em.YAX, em.XAX, np.array([Lbox-wi-W/2, 0, h])), Nsections=8)
    
    # Next we subtract the stubs and the center conductor from the box and feedline.
    for ro in stubs:
        box = em.geo.subtract(box, ro)
    box = em.geo.subtract(box, feed1in, remove_tool=False)
    box = em.geo.subtract(box, feed2in, remove_tool=False)
    feed1out = em.geo.subtract(feed1out, feed1in, remove_tool=True)
    feed2out = em.geo.subtract(feed2out, feed2in, remove_tool=True)
    
    # Finally we may define our geometry
    m.define_geometry(box, feed1out, feed2out)

    m.view()

    # We define our frequency range and a fine sampling.
    m.physics.set_frequency_range(6e9, 8e9, 4)
    m.physics.resolution = 0.04

    # To improve simulation quality we refine the faces at the top of the cylinders.
    for stub in stubs:
        m.mesher.set_boundary_size(box.face('back', tool=stub), 0.0002)

    # Finally we may create our mesh.
    m.generate_mesh()

    m.view()

    # We define our modal ports, assign the boundary condition and execute a modal analysis to solve for the
    # coaxial field mode.
    port1 = em.bc.ModalPort(m.select.face.near(-lfeed, 0, h), 1)
    port2 = em.bc.ModalPort(m.select.face.near(Lbox+lfeed, 0, h), 2)

    m.physics.assign(port1, port2)

    m.physics.modal_analysis(port1, 1, TEM=True)
    m.physics.modal_analysis(port2, 1, TEM=True)
    
    # At last we can compute the frequency domain study
    data = m.physics.frequency_domain()

    # We plot our S-parameters using the Pyescher module (that I created).
    f, S11 = data.ax('freq').S(1,1)
    f, S21 = data.ax('freq').S(2,1)

    plt.plot(f/1e9, 20*np.log10(np.abs(S11)))
    plt.plot(f/1e9, 20*np.log10(np.abs(S21)))
    plt.legend(['S11','S21'])
    plt.xlabel('Frequency (GHz)')
    plt.ylabel('Sparam (dB)')
    plt.grid(True)
    plt.show()
    
    # We can also plot the fild inside. First we create a grid of sample point coordinates
    xs = np.linspace(0, Lbox, 41)
    ys = np.linspace(-a/2, a/2, 11)
    zs = np.linspace(0, b, 15)

    X, Y, Z = np.meshgrid(xs, ys, zs, indexing='ij')
    X = X.flatten()
    Y = Y.flatten()
    Z = Z.flatten()

    # The E-field can be interpolated by selecting a desired solution and then interpolating it.

    Ex, Ey, Ez = data.item(3).interpolate(X,Y,Z,data.item(3).freq).E

    # We can add the objects we want and fields using the shown methods.
    m.display.add_object(box, opacity=0.1, show_edges=True)
    m.display.add_quiver(X,Y,Z, Ex.real, Ey.real, Ez.real)
    m.display.add_object(feed1out, opacity=0.1)
    m.display.add_portmode(port1, port1.get_mode().k0, 21)
    m.display.add_portmode(port2, port2.get_mode().k0, 21)
    m.display.show()