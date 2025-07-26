import emerge as em
from emerge.plot import plot_sp, plot_ff
import numpy as np

""" STANDARD GAIN HORN ANTENNA

This demo sets up and simulates a rectangular horn antenna in an
absorbing domain with PML layers. We compute return loss (S11) over a
90–110 GHz band and plot the far-field radiation pattern. """

# --- Units ---------------------------------------------------------------
mm = 0.001               # meters per millimeter

# --- Horn and feed dimensions -------------------------------------------
A = 27.4 * mm            # horn input aperture width
B = 10 * mm              # horn input aperture height
C = 7 * mm               # horn output aperture height
D = 2.054 * mm           # feed waveguide width
E = 1.04 * mm            # feed waveguide height
F = 21 * mm              # horn length
G = 1.016 * mm           # PML half-width (x-direction)
H = 2.032 * mm           # PML half-height (y-direction)

# --- Feed and simulation setup ------------------------------------------
Lfeed = 5 * mm           # length of feed waveguide
th = 1.5 * mm            # PML thickness
dx = 2 * mm              # distance from horn exit to PML start

# Create simulation object
m = em.Simulation3D('HornAntenna', loglevel='DEBUG')

# --- Coordinate system for horn geometry -------------------------------
hornCS = em.CS(em.YAX, em.ZAX, em.XAX)

# Feed waveguide as rectangular box (metal)
feed = em.geo.Box(
    Lfeed,   # length along X
    D/2,     # half-width along Y (centered)
    E/2,     # half-height along Z
    position=(-Lfeed, 0, 0)
)

# --- Horn geometry ------------------------------------------------------
# Inner horn taper from (D,E) at throat to (B,C) at mouth over length F
horn_in = em.geo.Horn(
    (D, E), (B, C), F, hornCS
)
# Outer horn (including metal thickness) helps define PML subtraction
horn_out = em.geo.Horn(
    (D+2*th, E+2*th), (B+2*th, C+2*th), F, hornCS
)

# --- Bounding objects and PML -------------------------------------------
# Define large intersection box to trim horn geometry
ibox = em.geo.Box(30*mm, 30*mm, 30*mm)
horn_in = em.geo.intersect(horn_in, ibox, remove_tool=False)
horn_out = em.geo.intersect(horn_out, ibox)

# Create airbox with PML layers on +X, +Y, +Z faces
rat = 1.6  # PML extension ratio
air, *pmls = em.geo.pmlbox(
    4*mm,          # air padding before PML
    rat*B/2,       # half-height in Y
    rat*C/2,       # half-width in Z
    (F - dx, 0, 0),# PML origin offset along X
    thickness=1.5*mm,
    N_mesh_layers=4,
    top=True, right=True, back=True
)
# Subtract horn volume from airbox so PML does not cover metal
air2 = em.geo.subtract(air, horn_out)

# --- Solver parameters --------------------------------------------------
m.mw.set_frequency_range(90e9, 110e9, 3)  # 90–110 GHz sweep
m.mw.set_resolution(0.3)                # mesh resolution fraction

# --- Assemble geometry and mesh -----------------------------------------
m.define_geometry(air2, horn_in, feed, pmls)
m.generate_mesh()

# --- Boundary conditions ------------------------------------------------
p1 = m.mw.bc.ModalPort(feed.face('left'), 1)     # excite TE10 in feed
PMC = m.mw.bc.PMC(m.select.face.inplane(0, 0, 0, 0, 1, 0))  # perfect magnetic on symmetry
radiation_boundary = air.outside('front', 'left', 'bottom')  # open faces

# View mesh and BC selections
m.view(selections=[p1.selection, PMC.selection, radiation_boundary])

# --- Run frequency-domain solver ----------------------------------------
data = m.mw.frequency_domain(False)

# --- Plot return loss ---------------------------------------------------
scal = data.scalar.grid
plot_sp(scal.freq, scal.S(1,1))  # S11 vs frequency

# --- Far-field radiation pattern ----------------------------------------
# Compute E and H on 2D cut for phi=0 plane over -90° to 90°
ang, E, H = data.field[0].farfield_2d(
    (1, 0, 0), (0, 1, 0), radiation_boundary,
    (-90, 90), syms=['Ez','Hy']
)
# Normalize to free-space impedance and convert to dB
Eiso = np.sqrt(2 * np.pi / 376.14)
plot_ff(ang * 180/np.pi, 20 * np.log10(em.norm(E) / Eiso))
