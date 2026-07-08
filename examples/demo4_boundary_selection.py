import emerge as em
from emerge.plotting.pyvista import PVDisplay
import matplotlib.pyplot as plt
import numpy as np

"""
In this demonstration we are going to show how one can make selections using the 
available tools in EMerge. 

To understand the need for the given selection methods it is important to understand
what the difficulty is.

Any CAD implementation will create objects/shapes that can be subtracted or added
from one another. The direct result of this is that any face that logically exists
or is definable (like the front) on an input geometry (of say a box) is not guaranteed 
to exist after manipulations of the shapes. Additionally, the face tags that GMSH keeps
track off will change after you make changes to the geometries.

To still facilitate face selection, EMerge implements a heuristic face
selection algorithm that defines faces of objects as  by two sets of data:
an origin and the face normal. These pairs of origins/normals will be transformed
with all transformations on objects such that at a later stage, these
orinal references can be accessed. Then when a selection is requested, the boundaries
of objects will be computed together with their origin and normal. If an origin is close
to an original face surface and the normals are aligned, it is treated as part of the selection.

On top of that, EMerge has a selection interface that allows the user to select
faces and objects based on coordinates, layers etc.

In this demonstration we will create a complicated waveguide structure ending
in an air box to demonstrate the various selection methods.
"""

# First lets define some dimensions
mm = 0.001
wga = 22.86*mm
wgb = 10.16*mm
L = 50*mm

with em.Simulation3D('Test Mode', PVDisplay) as m:

    # first lets define a WR90 waveguide
    wg_box = em.geo.Box(L, wga, wgb, position=(-L, -wga/2, -wgb/2))
    # Then define a capacitive iris cutout
    cutout = em.geo.Box(2*mm, wga, wgb/2, position=(-L/2, -wga/2, -wgb/2))

    # remove the cutout from the box
    wg_box = em.geo.remove(wg_box, cutout)

    # define an air-box to radiat in.
    airbox = em.geo.Box(L/2, L, L, position=(0,-L/2, -L/2))

    # Now define the geometry
    m.define_geometry(wg_box, airbox)

    # Lets define a frequency range for our simulation. This is needed
    # If we want to mesh our model.
    m.physics.set_frequency_range(8e9, 10e9, 11)

    # Now lets mesh our geometry
    m.generate_mesh()

    ## We can now select faces and show them using the .view() interface

    # The box is defined in XYZ space. The sides left/right correspond to the
    # X-axis, The sides top/down to the Z-axis and front/back to the Y-axis.
    feed_port = wg_box.face('left')

    # We can also select the outside and exclude a given face.
    radiation_boundary = airbox.outside('left')
    
    # Lets view our result
    m.view(selections=[feed_port, radiation_boundary])

    # As you can see, the appropriate faces have been selected.
    # You can also see that the bottom side of the resultant box
    # which has been split in two can still be selected because of the selection system.

    m.view(selections=[wg_box.face('bottom'),])
    
    # You can even access faces of the original tool objects using the optional argument tool.

    m.view(selections=[wg_box.face('left', tool=cutout),])

    # Another way to select the radiation boundary on the right is by using the 
    # selction interface.
    
    # The interface works by a language like method-chaining philosophy.
    # The .select attribute is a Selector class. The property .face returns
    # The same selector class but with the 'face selection mode' turned on.
    # Now we can call the 'inlayer' method which selects all faces of which the
    # Center of mass is inside the layer ranging from the provided starting
    # coordinate up to all coordinates that extend to the vector (L,0,0).
    #
    #            |                    |
    #            |       ____\        |
    #   (origin) + ---- vector ------>|
    #            |                    |
    #            |                    |
    #            < inside is selected >
    #

    radiation_boundary_2 = m.select.face.inlayer(1*mm, 0,0, (L,0,0))
    m.view(selections=[radiation_boundary_2,])

    # Now lets define our simulation futher and do some farfield-computation!

    port = em.bc.ModalPort(feed_port, 1)
    rad = em.bc.AbsorbingBoundary(radiation_boundary)

    m.physics.assign(port, rad)

    m.physics.modal_analysis(port, 1)

    # Run the simulation
    data = m.physics.frequency_domain()


    # First the S11 plot
    f, S11 = data.ax('freq').S(1,1)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(f/1e9, 20*np.log10(np.abs(S11)))
    plt.show()

    # First we need to create a boundary mesh
    rad_surf = m.mesh.boundary_surface(radiation_boundary.tags, (0,0,0))
    # Then we need to compute the E-field on the edges.  We will pick the first frequency.
    Ein, Hin = data.item(0).interpolate(*rad_surf.exyz).EH

    # Then we define some angles for our plot
    theta = np.linspace(-np.pi/2, 1.5*np.pi, 201)
    phi = 0*theta
    k0 =  data.item(0).k0
    E, H = em.physics.edm.stratton_chu(Ein, Hin, rad_surf, theta, phi, k0)

    # Finally we create the plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='polar')
    ax.set_theta_zero_location('N')
    ax.plot(theta, em.norm(E), label='E-field', color='blue')
    plt.show()