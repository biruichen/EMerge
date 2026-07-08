from _emerge.material import Material, AIR, COPPER

### MATERIALS 
VACUUM = Material(color=(0.5,0.5,0.5), opacity=0.05)
FR4 = Material(er=4.4, tand=0.001, color=(0.1,1.0,0.2), opacity=0.9)

ALUMINUM  = Material(cond=3.5e7,         color=(0.7, 0.7, 0.7))
GOLD      = Material(cond=4.1e7,         color=(1.0, 0.84, 0.0))
SILVER    = Material(cond=6.3e7,         color=(0.75, 0.75, 0.75))
TIN       = Material(cond=9.17e6,        color=(0.9, 0.8, 0.6))
NICKEL    = Material(cond=1.43e7,        color=(0.47, 0.43, 0.38))
IRON      = Material(cond=1.0e7,  ur=5000, color=(0.4, 0.4, 0.4))
STEEL     = Material(cond=1.45e6, ur=100,  color=(0.5, 0.5, 0.5))

SILICON   = Material(er=11.68,    cond=0.1, color=(0.2, 0.2, 0.2))
SIO2      = Material(er=3.9,                       color=(0.9, 0.9, 0.9), opacity=0.5)
GAAS      = Material(er=12.9,    cond=0.0, color=(0.3, 0.3, 0.8), opacity=0.5)

PTFE      = Material(er=2.1,     tand=0.0002,       color=(0.8, 0.8, 0.8), opacity=0.7)
POLYIMIDE = Material(er=3.4,     tand=0.02,         color=(1.0, 0.5, 0.0), opacity=0.8)
CERAMIC   = Material(er=6.0,     tand=0.001,        color=(0.8, 0.7, 0.7), opacity=0.8)

WATER     = Material(er=80.1,    cond=0.0,  color=(0.0, 0.5, 1.0), opacity=0.3)
FERRITE   = Material(er=12.0,    ur=2000, tand=0.02, color=(0.6, 0.3, 0.3), opacity=0.9)

# Specialty RF Substrates
ROGERS_4350B = Material(er=3.66, tand=0.0037,       color=(0.2, 0.8, 0.5), opacity=0.9)
ROGERS_5880  = Material(er=2.2,  tand=0.0009,       color=(0.2, 0.6, 0.8), opacity=0.9)
ROGERS_RO3003   = Material(er=3.0,  tand=0.0013,      color=(0.3, 0.7, 0.7), opacity=0.9)
ROGERS_RO3010   = Material(er=10.2, tand=0.0023,      color=(0.2, 0.5, 0.2), opacity=0.9)
ROGERS_RO4003C  = Material(er=3.55, tand=0.0027,      color=(0.4, 0.6, 0.8), opacity=0.9)
ROGERS_DUROID6002 = Material(er=2.94, tand=0.0012,    color=(0.6, 0.4, 0.6), opacity=0.9)
ROGERS_RT5880  = Material(er=2.2,  tand=0.0009,       color=(0.2, 0.6, 0.8), opacity=0.9)
TACONIC_RF35   = Material(er=3.5,  tand=0.0018,       color=(0.8, 0.5, 0.5), opacity=0.9)
TACONIC_TLC30  = Material(er=3.0,  tand=0.0020,       color=(0.9, 0.6, 0.4), opacity=0.9)
ISOLA_I_TERA_MT = Material(er=3.45, tand=0.0030,      color=(0.5, 0.5, 0.8), opacity=0.9)
ISOLA_NELCO_4000_13 = Material(er=3.77, tand=0.008,  color=(0.7, 0.7, 0.5), opacity=0.9)
VENTEC_VERDELAY_400HR = Material(er=4.0,  tand=0.02,  color=(0.6, 0.8, 0.6), opacity=0.9)

# Legacy FR Materials
FR1            = Material(er=4.8,  tand=0.025,        color=(0.9, 0.9, 0.7), opacity=0.9)
FR2            = Material(er=4.8,  tand=0.02,        color=(0.9, 0.8, 0.6), opacity=0.9)
# Magnetic Materials
MU_METAL  = Material(cond=1.0e6, ur=200000, color=(0.4, 0.4, 0.5), opacity=0.9)
