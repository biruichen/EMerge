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

def  norm(Field: np.ndarray) -> np.ndarray:
    """ Computes the complex norm of a field (3,N)

    Args:
        Field (np.ndarray): The input field, shape (3,N)

    Returns:
        np.ndarray: The complex norm in shape (N,)
    """
    return np.sqrt(np.abs(Field[0,:])**2 + np.abs(Field[1,:])**2 + np.abs(Field[2,:])**2)

def coax_rout(rin: float,
              eps_r: float = 1,
              Z0: float = 50) -> float:
    """Computes the outer radius given a dielectric constant, inner radius and characteristic impedance

    Args:
        rin (float): The inner radius
        eps_r (float, optional): The dielectric permittivity. Defaults to 1.
        Z0 (float, optional): The impedance. Defaults to 50.

    Returns:
        float: The outer radius
    """
    return rin*10**(Z0*np.sqrt(eps_r)/138)

def coax_rin(rout: float,
              eps_r: float = 1,
              Z0: float = 50) -> float:
    """Computes the inner radius given a dielectric constant, outer radius and characteristic impedance

    Args:
        rin (float): The outer radius
        eps_r (float, optional): The dielectric permittivity. Defaults to 1.
        Z0 (float, optional): The impedance. Defaults to 50.

    Returns:
        float: The inner radius
    """
    return rout/10**(Z0*np.sqrt(eps_r)/138)

