import emerge as em
from emerge.pyvista import PVDisplay

"""This demo is still in progress. 

For now it just shows you how to work with the revolve system.
"""
with em.Simulation3D('Revolve test', PVDisplay) as m:

    poly = em.geo.XYPolygon.rect(0.5, 1, (0.5, 0))

    m.view()

    vol = poly.revolve(em.cs.CoordinateSystem(em.XAX, em.ZAX, em.YAX), (0,0,0), (0,0,1))
    
    m.define_geometry(vol)

    m.physics.set_frequency(0.2e9)

    m.generate_mesh()

    m.view()