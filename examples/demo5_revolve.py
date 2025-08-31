import emerge as em
from emerge.plot import plot_sp, plot_ff
import numpy as np

""" CONICAL HORN ANTENNA

This demo uses the revolve feature to build a conical horn antenna. The conical
horn antenna is created by revolving a 2D XYPolygon profile. We compute and plot 
the far-field radiation pattern at an operating frequency of 10GHz.

The dimensions come from this video:
https://www.youtube.com/watch?v=DuXdLuYBGQk

Demo by Edvin Berling
"""

# --- Units ---------------------------------------------------------------
cm = 0.01   # meters per centimeter

# --- Horn and feed dimensions -------------------------------------------
aperture_radius  = 10.334/2 * cm    # horn aperture radius
aperture_length  = 7.809 * cm       # horn length

waveguide_radius = 2.779/2 * cm     # feed waveguide radius
waveguide_length = 2.872 * cm       # feed waveguide length

# --- Create simulation object -------------------------------------------
model = em.Simulation('ConicalHornAntenna')
model.check_version("0.6.11")   # Checks version compatibility

# --- Feed geometry -------------------------------------------------------
feed = em.geo.Cylinder(
    waveguide_radius, 
    waveguide_length, 
    cs=em.YZPLANE.cs()
)

# --- Horn geometry (revolved polygon) -----------------------------------
# Define polygon profile: (x = length, y = radius)
horn_poly = em.geo.XYPolygon(
    [waveguide_length, aperture_length+waveguide_length, aperture_length+waveguide_length, waveguide_length],
    [0, 0, aperture_radius, waveguide_radius]
)
# Revolve polygon around X-axis to create 3D horn
horn_vol = horn_poly.revolve(em.XZPLANE.cs(), (0,0,0), (1,0,0))

# --- Surrounding air --------------------------------------------
air = em.geo.Box(5*cm, 20*cm, 20*cm, (aperture_length + waveguide_length,-10*cm,-10*cm)).background()

# --- Finalize geometry --------------------------------------------------
model.commit_geometry()

# --- Solver setup -------------------------------------------------------
model.mw.set_frequency(10e9)                  # 10GHz frequency
model.mw.set_resolution(0.3)                  # mesh resolution fraction
model.generate_mesh()
model.view(selections=[air.boundary(exclude=('left',)),], plot_mesh=False)

# --- Boundary conditions ------------------------------------------------
port1 = model.mw.bc.ModalPort(feed.face('front'), 1)  # excite port at waveguide
radiation_boundary = air.boundary(exclude=('left',)) # open faces
abc = model.mw.bc.AbsorbingBoundary(radiation_boundary)
# --- Run frequency-domain solver ----------------------------------------
data = model.mw.run_sweep()

# --- Far-field radiation pattern (2D cut) -------------------------------
ff_data = data.field[0].farfield_2d(
    (1, 0, 0), (0, 1, 0), radiation_boundary,
    (-90, 90)
)
plot_ff(ff_data.ang * 180/np.pi, ff_data.normE/em.lib.EISO, dB=True, ylabel='Gain [dBi]')

# --- Visualization ------------------------------------------------------
model.display.add_object(horn_vol, opacity=0.5)
model.display.add_object(feed, opacity=0.5)
model.display.add_surf(
    *data.field[0].farfield_3d(radiation_boundary).surfplot(
        'normE', 'abs', True, True, -10, 5*cm, (waveguide_length+aperture_length,0,0)
    ),
    cmap='viridis', symmetrize=False
)
model.display.show()