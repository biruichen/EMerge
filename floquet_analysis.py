import emerge as em
from emerge.plot import plot_sp
import numpy as np

mm = 0.001
W = 20*mm
H = 50*mm

m = em.Simulation3D('Floquet')

cell = em.RectCell(W, W)
sphere = em.geo.Sphere(W*0.2, position=(0,0,H/2)).set_material(em.lib.COPPER)

air = cell.volume(0, H)
m.view()
p1, P1 = cell.floquet_port(1, H)
p2, P2 = cell.floquet_port(2, 0)

m.define_geometry(air, p1, p2, sphere)

m.physics.set_frequency_range(7e9,9e9,3)
m.physics.set_resolution(0.2)
m.mesher.set_periodic_cell(cell)
m.mesher.set_domain_size(sphere, 2*mm)
m.generate_mesh()
#m.view()

thetas = np.arange(0,85,20)
periodic_bcs = cell.bcs
m.physics.assign(P1, P2, *periodic_bcs)

for theta in m.parameter_sweep(False, theta=thetas):
    cell.set_scanangle(theta,0)

    sol = m.physics.frequency_domain(parallel=True, njobs=3)


th, freq, S11 = sol.ax('theta', 'freq').S(1,1)
th, freq, S21 = sol.ax('theta', 'freq').S(2,1)

import matplotlib.pyplot as plt

fig, ax = plt.subplots()
labels = []
for i, f in enumerate(m.physics.frequencies):
    plt.plot(th, 20*np.log10(np.abs(S11[:,i])))
    labels.append(f'Freq = {f/1e9:.1f}GHz')
ax.grid(True)
ax.legend(labels)
ax.set_xlabel('Angle (deg)')
ax.set_ylabel('Reflection Coeff (dB)')
plt.show()

m.display.set.add_light = False
m.display.add_object(air)
m.display.add_surf(*sol.item(1).cutplane(2*mm, y=0).scalar('Ey','complex'))
m.display.add_surf(*sol.item(1).cutplane(2*mm, x=0).scalar('Ey','complex'))
m.display.add_portmode(P1, k0=sol.item(1).k0)
m.display.add_portmode(P2, k0=sol.item(1).k0)
m.display.show()
