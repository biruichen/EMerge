import emerge as em

"""DEMO 7: Periodic Cells

Since version 0.2 of EMerge, there is a good support for periodic environemnts.
The setup requires some manual steps. In this demonstration we will look at seting up a rectangular waveguide
array in a flat hexagonal periodic Tiling.
+-----------------+-----------------+-----------------+
|  +-----------+  |  +-----------+  |  +-----------+  |
|  |           |  |  |           |  |  |           |  |
|  +-----------+  |  +-----------+  |  +-----------+  |
+--------+--------+--------+--------+--------+--------+
         |  +-----------+  |  +-----------+  |
         |  |           |  |  |           |  |
         |  +-----------+  |  +-----------+  |
+--------+--------+--------+--------+--------+--------+
|  +-----------+  |  +-----------+  |  +-----------+  |
|  |           |  |  |           |  |  |           |  |
|  +-----------+  |  +-----------+  |  +-----------+  |
+--------+--------+--------+--------+--------+--------+
"""
mm = 0.001
a = 106*mm
b = 30*mm
H = 70*mm
wga = 70*mm
wgb = 18*mm
fl = 25*mm

# We start again by defining our simulation model
model = em.Simulation3D('Periodic', loglevel='DEBUG')

# Next we will create a PeriodicCell class (in our case a hexagonal cell). This class
# is simply meant to simplify our lives and improve the simulation setup flow.
# To define our hexagonal cell we have to specify the three points that make up our hexagonal grid. For a normal
# hexagon the points would be the following

#             _____
#            /     \
#           /       \
#     ,----(         )----.
#    /      \       /      \
#   /       (1)____/        \
#   \        /     \        /
#    \      /       \      /
#     )---(2)        )----(
#    /      \       /      \
#   /        \_____/        \
#   \       (3)    \        /
#    \      /       \      /
#     `----(         )----'
#           \       /
#            \_____/

# In the case of our rectangular waveguide array we will use the following;
#(1)-------+--------+--------+--------+--------+--------+
# |  +-----------+  |  +-----------+  |  +-----------+  |
# |  |           |  |  |           |  |  |           |  |
# |  +-----------+  |  +-----------+  |  +-----------+  |
#(2)------(3)-------+--------+--------+--------+--------+

periodic_cell = em.HexCell((-a/2, b/2, 0), (-a/2, -b/2, 0), (0, -b/2, 0))

# We can easily use our periodic cell to construct volumes with the appropriate faces. We simply call the volume method
# to construct a cell region from z=0 to z=H

model['box'] = periodic_cell.volume(0, H)

# We also create a waveguide foor the feed
model['wg'] = em.geo.Box(wga,wgb,fl, (-wga/2, -wgb/2,-fl) )

# Next we define our geometry as usual
# Beause we stored our geometry in our model object using the get and set-item notation. We don't have to pass the items anymore.
model.define_geometry()

model.physics.set_frequency_range(2.8e9,3.3e9,5)
model.physics.set_resolution(0.1)

# To make sure that we can run a periodic simulation we must tell the mesher that
# it has to copy the meshing on each face that is duplcated. We can simply pass our periodic
# cell to the mesher using the set_periodic_cell() method.
model.mesher.set_periodic_cell(periodic_cell)

# Then we create our mesh and view the result
model.generate_mesh()
model.view()

# Now lets define our boundary conditions
# First the waveguide port
wg = model['wg']
wgbc = em.bc.RectangularWaveguide(wg.face('bottom'), 1)

# And then the absorbing boundary at the top
abc = em.bc.AbsorbingBoundary(model['box'].face('back'))

# We can simply create the necessary periodic boundary conditions using the bcs() property of our cell.
periodic_bcs = periodic_cell.bcs

# Finally we assign the boundary conditions. 
model.physics.assign(wgbc, abc, *periodic_bcs)

# We can use the set_scanangle method to set the appropriate phases for the boundary. The scan angle is defined as following
# kx = sin(θ)·cos(ϕ)
# ky = sin(θ)·sin(ϕ)
# kx = cos(θ)ϕ
# The arguments of the function are θ,ϕ in degrees.
periodic_cell.set_scanangle(30,45)

# And at last we run our simulation and view the results.
data = model.physics.frequency_domain()

model.display.add_object(wg)
model.display.add_object(model['box'])
model.display.add_surf(*data.item(0).cutplane(3*mm, y=0).scalar('Ey','real'))
model.display.add_surf(*data.item(0).cutplane(3*mm, x=0).scalar('Ey','real'))
model.display.show()
