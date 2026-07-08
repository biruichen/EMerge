import emerge as em
from emerge.pyvista import PVDisplay
import numpy as np

mm = 0.001
a = 106*mm
b = 30*mm
H = 70*mm
wga = 70*mm
wgb = 18*mm
fl = 25*mm

write = False


with em.Simulation3D('Periodic', PVDisplay, loglevel='DEBUG') as m:
    m['box'] = em.geo.Box(a,b,H,(-a/2,-b/2,0))
    m['wg'] = em.geo.Box(wga,wgb,fl, (-wga/2, -wgb/2,-fl) )
    
    m.define_geometry()

    m.physics.set_frequency_range(2.8e9,3.3e9,5)

    fl = m['box'].face('left')
    fr = m['box'].face('right')

    m.mesher.set_periodic(fl, fr, (a,0,0))
    m.mesher.set_periodic(m['box'].face('front'), m['box'].face('back'), (0,b,0))
    
    m.generate_mesh()
    m.view(use_gmsh=True)

    box = m['box']
    wg = m['wg']
    period1 = em.bc.Periodic(box.face('left'), box.face('right'), (a,0,0))
    period2 = em.bc.Periodic(box.face('front'), box.face('back'), (0,b,0))

    period1.ux = np.sin(np.pi/4)
    period1.uz = np.cos(np.pi/4)
    period2.ux = np.sin(np.pi/4)
    period2.uz = np.cos(np.pi/4)

    wgbc = em.bc.RectangularWaveguide(wg.face('bottom'), 1)
    abc = em.bc.AbsorbingBoundary(box.face('top'))

    m.view(selections=[box.face('top'), wg.face('bottom'), box.face('front'), box.face('back')])
    m.physics.assign(period1, period2, wgbc, abc)

    data = m.physics.frequency_domain()

    m.display.add_object(wg)
    m.display.add_object(box)
    m.display.add_surf(*data.item(0).cutplane(3*mm, y=0).surf('Ey','real'))
    m.display.add_surf(*data.item(0).cutplane(3*mm, x=0).surf('Ey','real'))
    m.display.show()
