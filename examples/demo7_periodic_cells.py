import emerge as em
from emerge.pyvista import PVDisplay

mm = 0.001
a = 106*mm
b = 30*mm
H = 70*mm
wga = 70*mm
wgb = 18*mm
fl = 25*mm

write = False


m = em.Simulation3D('Periodic', PVDisplay, loglevel='DEBUG')

rect = em.HexCell((-a/2, b/2, 0), (-a/2, -b/2, 0), (0, -b/2, 0))

m['box'] = rect.volume(0, H)
m['wg'] = em.geo.Box(wga,wgb,fl, (-wga/2, -wgb/2,-fl) )

m.define_geometry()

m.physics.set_frequency_range(2.8e9,3.3e9,5)
m.physics.set_resolution(0.1)
m.mesher.set_periodic_cell(rect)

m.generate_mesh()
m.view()

wg = m['wg']

wgbc = em.bc.RectangularWaveguide(wg.face('bottom'), 1)
abc = em.bc.AbsorbingBoundary(m['box'].face('back'))

m.physics.assign(wgbc, abc, *rect.bcs())

rect.set_scanangle(60,45)

data = m.physics.frequency_domain(parallel=True, njobs=3)

m.display.add_object(wg)
m.display.add_object(m['box'])
m.display.add_surf(*data.item(0).cutplane(3*mm, y=0).scalar('Ey','real'))
m.display.add_surf(*data.item(0).cutplane(3*mm, x=0).scalar('Ey','real'))
m.display.show()
