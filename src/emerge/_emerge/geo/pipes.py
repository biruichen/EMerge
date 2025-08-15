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

from __future__ import annotations
import gmsh
import numpy as np
from typing import Literal, Callable
from ..geometry import GeoEdge, GeoSurface, GeoVolume



class Curve(GeoEdge):
    
    
    def __init__(self, xpts: np.ndarray, ypts: np.ndarray, zpts: np.ndarray, 
                 degree: int = 3,
                 weights: list[float] | None = None,
                 knots: list[float] | None = None,
                 ctype: Literal['Spline','BSpline','Bezier'] = 'Bezier'):
        self.xpts: np.ndarray = xpts
        self.ypts: np.ndarray = ypts
        self.zpts: np.ndarray = zpts
        
        points = [gmsh.model.occ.add_point(x,y,z) for x,y,z in zip(xpts, ypts, zpts)]
        
        if ctype.lower()=='spline':
            tags = gmsh.model.occ.addSpline(points)
            
        elif ctype.lower()=='bspline':
            if weights is None:
                weights = []
            if knots is None:
                knots = []
            tags = gmsh.model.occ.addBSpline(points, degree=degree, weights=weights, knots=knots)
        else:
            tags = gmsh.model.occ.addBezier(points)
        
        tags = gmsh.model.occ.addWire([tags,])
        gmsh.model.occ.remove([(0,tag) for tag in points])
        super().__init__(tags)
    



class Helix(GeoVolume):
    
    
        