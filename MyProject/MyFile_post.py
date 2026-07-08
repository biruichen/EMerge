import emerge as em
from emerge.pyvista import PVDisplay
import emerge.plot as plt
import numpy as np

# Constants
cm = 0.01
mm = 0.001
mil = 0.0254
um = 0.000001
PI = np.pi

with em.Simulation3D("MyFile", PVDisplay, load_file=True) as m:
    
    fdata = m.physics.freq_data

    f, S11 = fdata.ax('freq').S(1,1)
    f, S21 = fdata.ax('freq').S(2,1)
    plt.plot_sp(f/1e9, [S11, S21])

    m.display.add_object(m['box'])
    m.display.add_surf(*fdata.item(1).cutplane(1*mm, z=5*mm).scalar('Ez','real'))
    m.display.show()
