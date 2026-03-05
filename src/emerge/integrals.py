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


# Last Cleanup: 2026-03-04
import numpy as np
from emerge._emerge.physics.microwave.microwave_data import EHField
from emsutil.const import EPS0

def Qv(field: EHField) -> float:
    """ Drop in callable for volumetric loss integrations.
    
    Example:
    ---------
    >>> loss = data.field.find(freq=...).intvol(selection, Qv).real
    """
    return -1/2*2*np.pi*field.freq*EPS0*field.normE**2 * field.er.imag