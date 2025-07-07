import emerge as em
from emerge.pyvista import PVDisplay

"""This demo is still in progress. 

For now it just shows you how to work with the revolve system.
"""
model = em.Simulation3D('Revolve test', PVDisplay)

poly = em.geo.XYPolygon.rect(0.5, 1, (0.5, 0))

model.view()

vol = poly.revolve(em.cs.CoordinateSystem(em.XAX, em.ZAX, em.YAX), (0,0,0), (0,0,1))

model.define_geometry(vol)

model.mw.set_frequency(0.2e9)

model.generate_mesh()

model.view()