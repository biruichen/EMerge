import emerge as em
from emerge.plot import plot_sp
import numpy as np

mm = 0.001
mil = 0.0254*mm
W = 20
D = 20
w0 = 37
l0 = 100
l1 = 314.22
l2 = 301.658
l3 = 300.589

w1, w2, w3, w4, w5, w6 = 18.8, 43.484, 44.331, 44.331, 43.484, 18.8
g1, g2, g3, g4, g5, g6 = 9.63, 24.84, 41.499, 41.499, 24.84, 9.63

th = 20
e = 0
Wtot = 2*l1 + 5*l2 + 7*e + 2*l0
WP = 200
Dtot = 750

extra = 100

model = em.Simulation3D('Demo3', loglevel='DEBUG')


mat = em.Material(3.55, color=(0.1,1.0,0.2), opacity=0.1)
pcb = em.geo.PCBLayouter(th,unit=mil)

pcb.new(0,140,w0,(1,0)).store('p1').straight(l0).turn(0).straight(l1*0.8)\
    .straight(l1,w1, dy=abs(w1-w0)/2).jump(gap=g1, side='left', reverse=l1-e).straight(l1,w1)\
    .straight(l2,w2, dy=abs(w2-w1)/2).jump(gap=g2, side='left', reverse=l2-e).straight(l2,w2)\
    .straight(l3,w3).jump(gap=g3, side='left', reverse=l2-e).straight(l2,w3)\
    .straight(l3,w4).jump(gap=g4, side='left', reverse=l2-e).straight(l2,w4)\
    .straight(l2,w5).jump(gap=g5, side='left', reverse=l2-e).straight(l2,w5)\
    .straight(l1,w6, dy=abs(w2-w1)/2).jump(gap=g6, side='left', reverse=l1-e).straight(l1,w6)\
    .turn(0).straight(l1*0.8, w0).straight(l0,w0, dy=abs(w1-w0)/2).store('p2')

stripline = pcb.compile_paths(merge=True)

pcb.determine_bounds(topmargin=150, bottommargin=150)

diel = pcb.gen_pcb()
air = pcb.gen_air(4*th)

diel.material = mat

p1 = pcb.modal_port(pcb.load('p1'), width_multiplier=5, height=4*th)
p2 = pcb.modal_port(pcb.load('p2'), width_multiplier=5, height=4*th)

model.physics.set_resolution(0.2)

model.physics.set_frequency_range(5.2e9,6.2e9,23)

model.define_geometry(stripline, diel, p1, p2, air)

model.mesher.set_boundary_size(stripline, 0.5*mm)

model.generate_mesh()

model.view(use_gmsh=True)
port1 = em.bc.ModalPort(p1, 1, True)
port2 = em.bc.ModalPort(p2, 2, False)
pec = em.bc.PEC(stripline)

model.physics.assign(port1, port2, pec)

model.physics.modal_analysis(port1, 1, direct=True, TEM=True, freq=10e9)
model.physics.modal_analysis(port2, 1, direct=True, TEM=True, freq=10e9)

d = model.display
d.add_object(diel, color='green', opacity=0.5)
d.add_object(stripline, color='red')
d.add_object(p1, color='blue', opacity=0.3)
d.add_portmode(port1, 21)
d.add_portmode(port2, 21)
d.show()

data = model.physics.frequency_domain(parallel=True, njobs=4, frequency_groups=8)

f = np.linspace(5.2e9, 6.2e9, 1001)

S11 = data.model_S(1,1)(f)
S21 = data.model_S(2,1)(f)

plot_sp(f/1e9, [S11, S21], labels=['S11','S21'])
