import emerge as em
from emerge.pyvista import PVDisplay
import numpy as np
from emerge.plot import *
# Constants
cm = 0.01
mm = 0.001
mil = 0.0254
um = 0.000001
PI = np.pi

# Variable definitions
w1 = 40.3*mm#28.4*mm
l1 = 40.3*mm
h1 = 2.4*mm
fp = 19.3*mm
w2 = 27*mm
l2 = 26.7*mm
h2 = 10.2*mm

htot = h1+h2
W = 59*mm
W2 = 51*mm
WF = 48*mm

th = 0.5*mm
thf = 2*mm
WA = 70*mm
HA = 40*mm
RF35 = em.Material(3.5, tand=0.0018, color=(0.2,1.0,0.3))
RESIN = em.Material(2.9, color=(0.5,0.5,0.5))
lfeed = 5*mm
with em.Simulation3D("File1", PVDisplay, loglevel='DEBUG', load_file=True) as m:
    # Casiing
    data = m.physics.freq_data

    f, S11 = data.ax('freq').S(1,1)

    #smith(f,S11*np.exp(1j*2*5*mm*2*np.pi*f/299792458))

    for item in m.obj.keys():
        m.display.add_object(m[item], opacity=0.1)
    m.display.animate().add_surf(*data.item(5).cutplane(1*mm, x=0).scalar('Ey','complex'), scale='symlog', cmap="CET_D1A")
    m.display.show()