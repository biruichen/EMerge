import emerge as em

model = em.Simulation('Sphere')
model.check_version("0.6.10") # Checks version compatibility.

poly = em.geo.XYPolygon([0, 0.05, 0.05, 0], [0, 0, 0.1, 0.1])
vol = poly.revolve(em.XZPLANE.cs(), (0,0,0), (1,0,0))

model.commit_geometry()


model.mw.set_frequency(3e9)

model.generate_mesh()
model.view(selections=[vol.face('side1'),])