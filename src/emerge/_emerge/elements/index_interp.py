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
from numba import njit, f8, c16, i8, types # type: ignore
import numpy as np


@njit(i8[:](f8[:,:], i8[:,:], f8[:,:], i8[:]), cache=True, nogil=True)
def index_interp(coords: np.ndarray,
                tets: np.ndarray, 
                nodes: np.ndarray,
                tetids: np.ndarray):
    ''' Nedelec 2 tetrahedral interpolation of the analytic curl'''
    # Solution has shape (nEdges, nsols)
    nNodes = coords.shape[1]
    
    prop = np.full((nNodes, ), -1, dtype=np.int64)
    
    for i_iter in range(tetids.shape[0]):
        itet = tetids[i_iter]
        
        iv1, iv2, iv3, iv4 = tets[:, itet]

        v1 = nodes[:,iv1]
        v2 = nodes[:,iv2]
        v3 = nodes[:,iv3]
        v4 = nodes[:,iv4]

        bv1 = v2 - v1
        bv2 = v3 - v1
        bv3 = v4 - v1

        blocal = np.zeros((3,3))
        blocal[:,0] = bv1
        blocal[:,1] = bv2
        blocal[:,2] = bv3
        basis = np.linalg.pinv(blocal)

        coords_offset = coords - v1[:,np.newaxis]
        coords_local = (basis @ (coords_offset))


        inside = ((coords_local[0,:] + coords_local[1,:] + coords_local[2,:]) <= 1.00000001) & (coords_local[0,:] >= -1e-6) & (coords_local[1,:] >= -1e-6) & (coords_local[2,:] >= -1e-6)

        if inside.sum() == 0:
            continue
        
        prop[inside] = itet

    return prop
