# EMerge is an open source Python based FEM EM simulation module.
# Copyright (C) 2025  Robert Fennis.

# This program is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public License
# as published by the Free Software Foundation; either version 2
# of the License, or (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program; if not, see
# <https://www.gnu.org/licenses/>.

import numpy as np
from dataclasses import dataclass

@dataclass
class Material:
    """The Material class generalizes a material in the EMerge FEM environment.

    If a scalar value is provided for the relative permittivity or the relative permeability
    it will be used as multiplication entries for the material property diadic as identity matrix.

    Additionally, a function may be provided that computes a coordinate dependent material property
    for _fer. For example: Material(_fer = lambda x,y,z: ...). 
    The x,y and z coordinates are provided as a (N,) np.ndarray. The return array must be of shape (3,3,N)!
    
    """
    er: float = 1
    ur: float = 1
    tand: float = 0
    sigma: float = 0
    _neff: float = None
    _fer: callable = None
    _fur: callable = None
    color: tuple[int,int,int] = (0.9,0.9,1)
    opacity: float = 1.0

    @property
    def ermat(self) -> np.ndarray:
        if isinstance(self.er, (float, complex, int, np.float64, np.complex128)):
            return self.er*(1-1j*self.tand)*np.eye(3)
        else:
            return self.er*(1-1j*self.tand)
    
    @property
    def urmat(self) -> np.ndarray:
        if isinstance(self.ur, (float, complex, int, np.float64, np.complex128)):
            return self.ur*np.eye(3)
        else:
            return self.ur
    
    @property
    def neff(self) -> complex:
        if self._neff is not None:
            return self._neff
        er = self.ermat[0,0]
        ur = self.urmat[0,0]

        return np.abs(np.sqrt(er*(1-1j*self.tand)*ur))
    
    @property
    def fer2d(self) -> callable:
        if self._fer is None:
            return lambda x,y: self.er*(1-1j*self.tand)*np.ones_like(x)
        else:
            return self._fer
        
    @property
    def fur2d(self) -> callable:
        if self._fur is None:

            return lambda x,y: self.ur*np.ones_like(x)
        else:
            return self._fur
    @property
    def fer3d(self) -> callable:
        if self._fer is None:
            return lambda x,y,z: self.er*(1-1j*self.tand)*np.ones_like(x)
        else:
            return self._fer
    
    @property
    def fur3d(self) -> callable:
        if self._fur is None:
            return lambda x,y,z: self.ur*np.ones_like(x)
        else:
            return self._fur
    @property
    def fer3d_mat(self) -> callable:
        if self._fer is None:
            
            return lambda x,y,z: np.repeat(self.ermat[:, :, np.newaxis], x.shape[0], axis=2)
        else:
            return self._fer
    
    @property
    def fur3d_mat(self) -> callable:
        if self._fur is None:
            return lambda x,y,z: np.repeat(self.urmat[:, :, np.newaxis], x.shape[0], axis=2)
        else:
            return self._fur
    
AIR = Material(color=(0.8,0.9,1.0), opacity=0.2)
VACUUM = Material(color=(0.5,0.5,0.5), opacity=0.05)
COPPER = Material(color=(0.6, 0.2, 0.1))
FR4 = Material(er=4.4, tand=0.001, color=(0.1,1.0,0.2), opacity=0.9)

ALUMINUM  = Material(sigma=3.5e7,         color=(0.7, 0.7, 0.7))
GOLD      = Material(sigma=4.1e7,         color=(1.0, 0.84, 0.0))
SILVER    = Material(sigma=6.3e7,         color=(0.75, 0.75, 0.75))
TIN       = Material(sigma=9.17e6,        color=(0.9, 0.8, 0.6))
NICKEL    = Material(sigma=1.43e7,        color=(0.47, 0.43, 0.38))

IRON      = Material(sigma=1.0e7,  ur=5000, color=(0.4, 0.4, 0.4))
STEEL     = Material(sigma=1.45e6, ur=100,  color=(0.5, 0.5, 0.5))

SILICON   = Material(er=11.68,    sigma=0.1, color=(0.2, 0.2, 0.2))
SIO2      = Material(er=3.9,                       color=(0.9, 0.9, 0.9), opacity=0.5)
GAAS      = Material(er=12.9,    sigma=0.0, color=(0.3, 0.3, 0.8), opacity=0.5)

PTFE      = Material(er=2.1,     tand=0.0002,       color=(0.8, 0.8, 0.8), opacity=0.7)
POLYIMIDE = Material(er=3.4,     tand=0.02,         color=(1.0, 0.5, 0.0), opacity=0.8)
CERAMIC   = Material(er=6.0,     tand=0.001,        color=(0.8, 0.7, 0.7), opacity=0.8)

WATER     = Material(er=80.1,    sigma=0.0,  color=(0.0, 0.5, 1.0), opacity=0.3)
FERRITE   = Material(er=12.0,    ur=2000, tand=0.02, color=(0.6, 0.3, 0.3), opacity=0.9)

# Specialty RF Substrates
ROGERS_4350B = Material(er=3.66, tand=0.0037,       color=(0.2, 0.8, 0.5), opacity=0.9)
ROGERS_5880  = Material(er=2.2,  tand=0.0009,       color=(0.2, 0.6, 0.8), opacity=0.9)
# Additional Dielectric Materials for FEM Simulations

ROGERS_RO3003   = Material(er=3.0,  tand=0.0013,      color=(0.3, 0.7, 0.7), opacity=0.9)
ROGERS_RO3010   = Material(er=10.2, tand=0.0023,      color=(0.2, 0.5, 0.2), opacity=0.9)
ROGERS_RO4003C  = Material(er=3.55, tand=0.0027,      color=(0.4, 0.6, 0.8), opacity=0.9)
ROGERS_DUROID6002 = Material(er=2.94, tand=0.0012,    color=(0.6, 0.4, 0.6), opacity=0.9)
ROGERS_RT5880  = Material(er=2.2,  tand=0.0009,       color=(0.2, 0.6, 0.8), opacity=0.9)  # alias for 5880

TACONIC_RF35   = Material(er=3.5,  tand=0.0018,       color=(0.8, 0.5, 0.5), opacity=0.9)
TACONIC_TLC30  = Material(er=3.0,  tand=0.0020,       color=(0.9, 0.6, 0.4), opacity=0.9)

ISOLA_I_TERA_MT = Material(er=3.45, tand=0.0030,      color=(0.5, 0.5, 0.8), opacity=0.9)
ISOLA_NELCO_4000_13 = Material(er=3.77, tand=0.008,  color=(0.7, 0.7, 0.5), opacity=0.9)

VENTEC_VERDELAY_400HR = Material(er=4.0,  tand=0.02,  color=(0.6, 0.8, 0.6), opacity=0.9)

# Legacy FR Materials
FR1            = Material(er=4.8,  tand=0.025,        color=(0.9, 0.9, 0.7), opacity=0.9)
FR2            = Material(er=4.8,  tand=0.02,        color=(0.9, 0.8, 0.6), opacity=0.9)
# Magnetic Materials
MU_METAL  = Material(sigma=1.0e6, ur=200000, color=(0.4, 0.4, 0.5), opacity=0.9)
