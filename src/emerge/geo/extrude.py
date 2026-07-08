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
import numpy as np
from ..cs import CoordinateSystem, GCS
from ..geometry import GeoVolume, GeoPolygon
from .shapes import Alignment
import gmsh
from typing import Generator, Callable
from ..selection import FaceSelection
from typing import Literal
from functools import reduce

def _discretize_curve(xfunc, yfunc, t0, t1, tol=1e-3, max_depth=20):
    """
    Adaptively sample the parametric curve (xfunc(t), yfunc(t)) from t0 to t1.

    Parameters
    ----------
    xfunc, yfunc : callables
        Functions of a single scalar t returning x(t) and y(t).
    t0, t1 : float
        Parameter interval to sample over.
    tol : float, optional
        Maximum allowed deviation (in Euclidean space) between the true curve
        midpoint and the chord midpoint before subdividing.
    max_depth : int, optional
        Maximum recursion depth to avoid pathological cases.

    Returns
    -------
    pts : (N,2) ndarray
        Array of sampled (x,y) points along the curve, ordered from t0→t1.
    """

    pts = []

    def recurse(ta, pa, tb, pb, depth):
        """
        Ensure the segment [pa,pb] approximates the curve on [ta,tb] within tol.
        """
        if depth > max_depth:
            # give up subdividing further
            pts.append(pb)
            return

        tm = 0.5*(ta + tb)
        pm = np.array([xfunc(tm), yfunc(tm)])

        # midpoint of the straight chord
        chord_mid = 0.5*(pa + pb)
        err = np.linalg.norm(pm - chord_mid)

        if err > tol:
            # subdivide
            recurse(ta, pa, tm, pm, depth+1)
            recurse(tm, pm, tb, pb, depth+1)
        else:
            # accept the straight segment
            pts.append(pb)

    # seed with the start point
    p0 = np.array([xfunc(t0), yfunc(t0)])
    p1 = np.array([xfunc(t1), yfunc(t1)])
    pts.append(p0)
    recurse(t0, p0, t1, p1, depth=0)

    return np.vstack(pts)

class GeoPrism(GeoVolume):
    """The GepPrism class generalizes the GeoVolume for extruded convex polygons.
    Besides having a volumetric definitions, the class offers a .front_face 
    and .back_face property that selects the respective faces.

    Args:
        GeoVolume (_type_): _description_
    """
    def __init__(self,
                 volume_tag: int,
                 front_tag: int,
                 side_tags: list[int],):
        super().__init__(volume_tag)
        self.front_tag: int = front_tag
        self.back_tag: int = None

        gmsh.model.occ.synchronize()
        o1 = gmsh.model.occ.get_center_of_mass(2, self.front_tag)
        n1 = gmsh.model.get_normal(self.front_tag, (0,0))
        self._add_face_pointer('back', o1, n1)

        tags = gmsh.model.get_boundary(self.dimtags, oriented=False)
        
        for dim, tag in tags:
            if (dim,tag) in side_tags:
                continue
            o2 = gmsh.model.occ.get_center_of_mass(2, tag)
            n2 = gmsh.model.get_normal(tag, (0,0))
            self._add_face_pointer('front', o2, n2)
            self.back_tag = tag
            break

        self.side_tags: list[int] = [dt[1] for dt in tags if dt[1]!=self.front_tag and dt[1]!=self.back_tag]

        for tag in self.side_tags:
            o2 = gmsh.model.occ.get_center_of_mass(2, tag)
            n2 = gmsh.model.get_normal(tag, (0,0))
            self._add_face_pointer(f'side{tag}', o2, n2)
            self.back_tag = tag

    def outside(self, *exclude: Literal['front','back']) -> FaceSelection:
        """Select all outside faces except for the once specified by outside

        Returns:
            FaceSelection: The resultant face selection
        """
        tagslist = [self._face_tags(name) for name in  self._face_pointers.keys() if name not in exclude]
        
        tags = list(reduce(lambda a,b: a+b, tagslist))
        return FaceSelection(tags)       

class XYPolygon:

    def __init__(self, 
                 xs: np.ndarray,
                 ys: np.ndarray):
        """Constructs an XY-plane placed polygon.

        Args:
            xs (np.ndarray): The X-points
            ys (np.ndarray): The Y-points
        """

        self.x: np.ndarray = xs
        self.y: np.ndarray = ys

        if np.sqrt((self.x[-1]-self.x[0])**2 + (self.y[-1]-self.y[0])**2) < 1e-6:
            self.x = self.x[:-1]
            self.y = self.y[:-1]

        self.N: int = self.x.shape[0]

        self.fillets: list[tuple[float, int]] = []

    def extend(self, xpts: list[float], ypts: list[float]) -> XYPolygon:
        """Adds a series for x and y coordinates to the existing polygon.

        Args:
            xpts (list[float]): The list of x-coordinates
            ypts (list[float]): The list of y-coordinates

        Returns:
            XYPolygon: The same XYpolygon object
        """
        self.x = np.hstack([self.x, np.array(xpts)])
        self.y = np.hstack([self.y, np.array(ypts)])
        return self
    
    def iterate(self) -> Generator[tuple[float, float],None, None]:
        """ Iterates over the x,y coordinates as a tuple."""
        for i in range(self.N):
            yield (self.x[i], self.y[i])

    def fillet(self, radius: float, *indices: int) -> None:
        """Add a fillet rounding with a given radius to the provided nodes.

        Example:
         >>> my_polygon.fillet(0.05, 2, 3, 4, 6)

        Args:
            radius (float): The radius
            *indices (int): The indices for which to apply the fillet.
        """
        for i in indices:
            self.fillets.append((radius, i))

    def _finalize(self, cs: CoordinateSystem = None) -> GeoPolygon:
        """Turns the XYPolygon object into a GeoPolygon that is embedded in 3D space.

        The polygon will be placed in the XY-plane of the provided coordinate center.

        Args:
            cs (CoordinateSystem, optional): The coordinate system in which to put the polygon. Defaults to None.

        Returns:
            GeoPolygon: The resultant 3D GeoPolygon object.
        """
        ptags = []
        xg, yg, zg = cs.in_global_cs(self.x, self.y, 0*self.x)

        for x,y,z in zip(xg, yg, zg):
            ptags.append(gmsh.model.occ.add_point(x,y,z))
        
        lines = []
        for i1, p1 in enumerate(ptags):
            p2 = ptags[(i1+1) % len(ptags)]
            lines.append(gmsh.model.occ.add_line(p1, p2))
        
        add = 0
        for radius, index in self.fillets:
            t1 = lines[index + add]
            t2 = lines[(index+add-1) % len(lines)]
            tag = gmsh.model.occ.fillet2_d(t1, t2, radius)
            lines.insert(index, tag)
            add += 1

        wiretag = gmsh.model.occ.add_wire(lines)
        surftag = gmsh.model.occ.add_plane_surface([wiretag,])
        poly = GeoPolygon([surftag,])
        poly.points = ptags
        poly.lines = lines
        return poly
    
    def extrude(self, length: float, cs: CoordinateSystem = None) -> GeoPrism:
        """Extrues the polygon along the Z-axis.
        The z-coordinates go from z1 to z2 (in meters). Then the extrusion
        is either provided by a maximum dz distance (in meters) or a number
        of sections N.

        Args:
            length (length): The length of the extrusion.

        Returns:
            GeoVolume: The resultant Volumetric object.
        """
        if cs is None:
            cs = GCS
        poly_fin = self._finalize(cs)
        zax = length*cs.zax.np
        
        volume = gmsh.model.occ.extrude(poly_fin.dimtags, zax[0], zax[1], zax[2])
        tags = [t for d,t in volume if d==3]
        surftags = [t for d,t in volume if d==2]
        return GeoPrism(tags, surftags[0], surftags)
    
    def revolve(self, cs: CoordinateSystem, origin: tuple[float, float, float], axis: tuple[float, float,float], angle: float = 360.0) -> GeoPrism:
        """Applies a revolution to the XYPolygon along the coordinate system Z-axis

        Args:
            cs (CoordinateSystem, optional): _description_. Defaults to None.
            angle (float, optional): _description_. Defaults to 360.0.

        Returns:
            Prism: The resultant 
        """
        if cs is None:
            cs = GCS
        poly_fin = self._finalize(cs)
        
        x,y,z = origin
        ax, ay, az = axis
        
        volume = gmsh.model.occ.revolve(poly_fin.dimtags, x,y,z, ax, ay, az, angle*np.pi/180)
        tags = [t for d,t in volume if d==3]
        surftags = [t for d,t in volume if d==2]
        return GeoPrism(tags, surftags[0], surftags)

    @staticmethod
    def circle(radius: float, 
               dsmax: float = None,
               tolerance: float = None,
               Nsections: int = None):
        """This method generates a segmented circle.

        The number of points along the circumpherence can be specified in 3 ways. By a maximum
        circumpherential length (dsmax), by a radial tolerance (tolerance) or by a number of 
        sections (Nsections).

        Args:
            radius (float): The circle radius
            dsmax (float, optional): The maximum circumpherential angle. Defaults to None.
            tolerance (float, optional): The maximum radial error. Defaults to None.
            Nsections (int, optional): The number of sections. Defaults to None.

        Returns:
            XYPolygon: The XYPolygon object.
        """
        if Nsections is not None:
            N = Nsections+1
        elif dsmax is not None:
            N = int(np.ceil((2*np.pi*radius)/dsmax))
        elif tolerance is not None:
            N = int(np.ceil(2*np.pi/np.arccos(1-tolerance)))

        angs = np.linspace(0,2*np.pi,N)

        xs = radius*np.cos(angs[:-1])
        ys = radius*np.sin(angs[:-1])
        return XYPolygon(xs, ys)

    @staticmethod
    def rect(width: float,
             height: float,
             origin: tuple[float, float],
             alignment: Alignment = Alignment.CORNER):
        if alignment is Alignment.CORNER:
            x0, y0 = origin
        else:
            x0 = origin[0]-width/2
            y0 = origin[1]-height/2
        xs = np.array([x0, x0, x0 + width, x0+width])
        ys = np.array([y0, y0+height, y0+height, y0])
        return XYPolygon(xs, ys)
    
    @staticmethod
    def parametric(xfunc: Callable,
                   yfunc: Callable,
                   tolerance: float,
                   tmin: float = 0,
                   tmax: float = 1):
        pts = _discretize_curve(xfunc, yfunc, tmin, tmax, tolerance)
        xs = pts[:,0]
        ys = pts[:,1]
        return XYPolygon(xs, ys)