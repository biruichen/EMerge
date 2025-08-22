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
from typing import Callable

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
    cond: float = 0
    _neff: float | None = None
    _fer: Callable | None= None
    _fur: Callable | None = None
    color: str = "#BEBEBE"
    _color_rgb: tuple[float,float,float] = (0.5, 0.5, 0.5)
    opacity: float = 1.0

    def __post_init__(self):
        hex_str = self.color.lstrip('#')
        self._color_rgb = tuple(int(hex_str[i:i+2], 16)/255.0 for i in (0, 2, 4))

    @property
    def sigma(self) -> float:
        return self.cond
        
    @property
    def color_rgb(self) -> tuple[float,float,float]:
        return self._color_rgb
    
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
    def fer2d(self) -> Callable:
        if self._fer is None:
            return lambda x,y: self.er*(1-1j*self.tand)*np.ones_like(x)
        else:
            return self._fer
        
    @property
    def fur2d(self) -> Callable:
        if self._fur is None:

            return lambda x,y: self.ur*np.ones_like(x)
        else:
            return self._fur
    @property
    def fer3d(self) -> Callable:
        if self._fer is None:
            return lambda x,y,z: self.er*(1-1j*self.tand)*np.ones_like(x)
        else:
            return self._fer
    
    @property
    def fur3d(self) -> Callable:
        if self._fur is None:
            return lambda x,y,z: self.ur*np.ones_like(x)
        else:
            return self._fur
    @property
    def fer3d_mat(self) -> Callable:
        if self._fer is None:
            
            return lambda x,y,z: np.repeat(self.ermat[:, :, np.newaxis], x.shape[0], axis=2)
        else:
            return self._fer
    
    @property
    def fur3d_mat(self) -> Callable:
        if self._fur is None:
            return lambda x,y,z: np.repeat(self.urmat[:, :, np.newaxis], x.shape[0], axis=2)
        else:
            return self._fur
        
AIR = Material(color="#4496f3", opacity=0.05)
COPPER = Material(cond=5.8e7, color="#62290c")