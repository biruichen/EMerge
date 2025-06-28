import emerge as em
from emerge.pyvista import PVDisplay
import numpy as np

# Constants
cm = 0.01
mm = 0.001
mil = 0.0254
um = 0.000001
PI = np.pi

# Variable definitions
w1 = 38.4*mm
l1 = 40.3*mm
h1 = 2.4*mm
fp = 19.3*mm
w2 = 27*mm
l2 = 26.7*mm
h2 = 10.2*mm

htot = h1+h2
W = 59*mm
W2 = 51*mm
WF = 43*mm

th = 0.5*mm
thf = 2*mm
WA = 70*mm
HA = 40*mm
RF35 = em.Material(3.5, tand=0.0018, color=(0.2,1.0,0.3))
RESIN = em.Material(2.9, color=(0.5,0.5,0.5))
lfeed = 5*mm

with em.Simulation3D("File1", PVDisplay, loglevel='DEBUG', save_file=True) as m:
    # Casiing
    Cout = em.geo.Box(W+2*thf, W+2*thf, h1+h2, (-W/2-thf,-W/2-thf,0))
    Cin = em.geo.Box(W, W, h1+h2, (-W/2,-W/2,0))
    m['case'] = em.geo.subtract(Cout,Cin).set_material(em.lib.COPPER)
    # Frame1
    f1out = em.geo.Box(W,W,h1,(-W/2,-W/2,0))
    f1in = em.geo.Box(WF,WF,h1,(-WF/2,-WF/2,0))
    
    m['f1'] = em.geo.subtract(f1out,f1in)
    # Frame2
    f2out = em.geo.Box(W,W,h2,(-W/2,-W/2,h1))
    f2in = em.geo.Box(WF,WF,h2,(-WF/2,-WF/2,h1))

    m['f2'] = em.geo.subtract(f2out,f2in)

    m['f1'].material = RESIN
    m['f2'].materila = RESIN

    # BOTTOM PCB
    m['pcb1'] = em.geo.Box(W2,W2,th,(-W2/2, -W2/2, h1-th))
    m['pcb2'] = em.geo.Box(W2,W2,th,(-W2/2, -W2/2, h1+h2-th))

    m['pcb1'].material = RF35
    m['pcb2'].material = RF35

    m['f1'] = em.geo.subtract(m['f1'], m['pcb1'], remove_tool=False)
    m['f2'] = em.geo.subtract(m['f2'], m['pcb2'], remove_tool=False)

    # Patches
    m['p1'] = em.geo.XYPlate(w1, l1, (-w1/2,-l1/2,h1)).set_material(em.lib.COPPER)
    m['p2'] = em.geo.XYPlate(w2, l2, (-w2/2,-l2/2,h1+h2)).set_material(em.lib.COPPER)
    
    # AIR
    m['air'] = em.geo.Box(WA, WA, HA, (-WA/2,-WA/2, 0)).set_material(em.lib.AIR).set_priority(1)

    m['via'] = em.geo.Cyllinder(0.5*mm, h1+lfeed, em.GCS.displace(0, -fp, -lfeed), Nsections=10).set_material(em.lib.COPPER).set_priority(15)
    m['out'] = em.geo.Cyllinder(em.coax_rout(0.5*mm, 1, 50), lfeed, em.GCS.displace(0, -fp, -lfeed), Nsections=10).set_material(em.lib.AIR)
    
    m.define_geometry()
    #m.view()
    
    m.physics.set_frequency_range(2.4e9,3.4e9,11)
    m.physics.set_resolution(0.16)
    m.mesher.set_domain_size(m['out'], 0.5*mm)
    m.mesher.set_domain_size(m['via'], 0.5*mm)
    m.mesher.set_boundary_size(m['p1'], 2*mm, growth_rate=1.3)
    m.mesher.set_boundary_size(m['p2'], 2*mm, growth_rate=1.3)
    m.generate_mesh()

    m.view(use_gmsh=True)#selections=[m['out'].face('front'),m['air'].outside('bottom')])

    # Set boundary conditions

    port1 = em.bc.ModalPort(m['out'].face('front'), 1)
    pec2 = em.bc.PEC(m['p1'])
    pec3 = em.bc.PEC(m['p2'])
    abc = em.bc.AbsorbingBoundary(m['air'].outside('bottom'))

    # Assign boundary conditions
    m.physics.assign(port1, pec2, pec3, abc)

    m.physics.modal_analysis(port1, 1, True, TEM=True)
    # Run simulation steps
    data = m.physics.frequency_domain()

    f, S11 = data.ax('freq').S(1,1)

    from emerge.plot import plot_sp

    plot_sp(f/1e9, S11)

    for item in m.obj.keys():
        m.display.add_object(m[item])
    m.display.add_surf(*data.item(5).cutplane(1*mm, x=0).scalar('Ey','real'))
    m.display.show()