from _emerge.material import Material, AIR, COPPER

### MATERIALS 
VACUUM = Material(color="#2d8cd5", opacity=0.05)
FR4 = Material(er=4.4, tand=0.001, color="#3c9747")
# Conductors
ALUMINUM  = Material(cond=3.5e7,         color="#aaaaaa")
GOLD      = Material(cond=4.1e7,         color="#A07130")
SILVER    = Material(cond=6.3e7,         color="#bfbfbf")
TIN       = Material(cond=9.17e6,        color="#d6d6d6")
NICKEL    = Material(cond=1.43e7,        color="#78706c")
IRON      = Material(cond=1.0e7,  ur=5000, color="#666666")
STEEL     = Material(cond=1.45e6, ur=100,  color="#808080")

# Semiconductors & Dielectrics
SILICON   = Material(er=11.68,    cond=0.1, color="#333333")
SIO2      = Material(er=3.9,                color="#e6e6e6")
GAAS      = Material(er=12.9,    cond=0.0, color="#404071")

# Dielectrics
PTFE      = Material(er=2.1,     tand=0.0002,       color="#cccccc")
POLYIMIDE = Material(er=3.4,     tand=0.02,         color="#b8b8b8")
CERAMIC   = Material(er=6.0,     tand=0.001,        color="#efead1")

# Liquids
WATER     = Material(er=80.1,    cond=0.0,  color="#0080ff", opacity=0.3)
FERRITE   = Material(er=12.0,    ur=2000, tand=0.02, color="#994d4d")

# Specialty RF Substrates
ROGERS_4350B = Material(er=3.66, tand=0.0037,       color="#3c9747")
ROGERS_5880  = Material(er=2.2,  tand=0.0009,       color="#3c9747")
ROGERS_RO3003   = Material(er=3.0,  tand=0.0013,      color="#3c9747")
ROGERS_RO3010   = Material(er=10.2, tand=0.0023,      color="#3c9747")
ROGERS_RO4003C  = Material(er=3.55, tand=0.0027,      color="#3c9747")
ROGERS_DUROID6002 = Material(er=2.94, tand=0.0012,    color="#3c9747")
ROGERS_RT5880  = Material(er=2.2,  tand=0.0009,       color="#3c9747")
TACONIC_RF35   = Material(er=3.5,  tand=0.0018,       color="#3c9747")
TACONIC_TLC30  = Material(er=3.0,  tand=0.0020,       color="#3c9747")
ISOLA_I_TERA_MT = Material(er=3.45, tand=0.0030,      color="#3c9747")
ISOLA_NELCO_4000_13 = Material(er=3.77, tand=0.008,  color="#3c9747")
VENTEC_VERDELAY_400HR = Material(er=4.0,  tand=0.02,  color="#3c9747")

# Legacy FR Materials
FR1            = Material(er=4.8,  tand=0.025,        color="#3c9747")
FR2            = Material(er=4.8,  tand=0.02,        color="#3c9747")

# Magnetic Materials
MU_METAL  = Material(cond=1.0e6, ur=200000, color="#666680")