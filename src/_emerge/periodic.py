from .selection import FaceSelection, SELECTOR_OBJ
from .cs import GCS, CoordinateSystem, Axis, _parse_axis
import numpy as np
from typing import Generator
from .bc import Periodic
from .geo.extrude import XYPolygon, GeoPrism


def _rotnorm(v: np.ndarray) -> np.ndarray:
    """Rotate 3D vector field v 90° counterclockwise around z axis.

    v shape = (3, Ny, Nx)
    """
    ax = np.array([-v[1], v[0], v[2]])
    ax = ax/np.linalg.norm(ax)
    return ax

class PeriodicCell:

    def __init__(self, 
                 origins: list[tuple[float, float, float]],
                 vectors: list[tuple[float, float, float] | Axis]):
        self.origins: list[tuple[float, float, float]] = origins
        self.vectors: list[Axis] = [_parse_axis(vec) for vec in vectors]
        self._bcs: list[Periodic] = []

    def volume(self, z1: float, z2: float) -> GeoPrism:
        """Genereates a volume with the cell geometry ranging from z1 tot z2

        Args:
            z1 (float): The start height
            z2 (float): The end height

        Returns:
            GeoPrism: The resultant prism
        """
        raise NotImplementedError('This method is not implemented for this subclass.')
    
    def cell_data(self) -> Generator[tuple[FaceSelection,FaceSelection,tuple[float, float, float]], None, None]:
        """An iterator that yields the two faces of the hex cell plus a cell periodicity vector

        Yields:
            Generator[np.ndarray, np.ndarray, np.ndarray]: The face and periodicity data
        """
        raise NotImplementedError('This method is not implemented for this subclass.')

    def bcs(self, exclude_faces: list[FaceSelection] = None) -> list[Periodic]:
        """Returns a list of Periodic boundary conditions for the given PeriodicCell

        Args:
            exclude_faces (list[FaceSelection], optional): A possible list of faces to exclude from the bcs. Defaults to None.

        Returns:
            list[Periodic]: The list of Periodic boundary conditions
        """
        bcs = []
        for f1, f2, a in self.cell_data():
            if exclude_faces is not None:
                f1 = f1 - exclude_faces
                f2 = f2 - exclude_faces
            bcs.append(Periodic(f1, f2, a))
        self._bcs = bcs
        return bcs
    
    def set_scanangle(self, theta: float, phi: float, degree: bool = True) -> None:
        """Sets the scanangle for the periodic condition. (0,0) is defined along the Z-axis

        Args:
            theta (float): The theta angle
            phi (float): The phi angle
            degree (bool): If the angle is in degrees. Defaults to True
        """
        if degree:
            theta = theta*np.pi/180
            phi = phi*np.pi/180

        ux = np.sin(theta)*np.cos(phi)
        uy = np.sin(theta)*np.sin(phi)
        uz = np.cos(theta)
        for bc in self._bcs:
            bc.ux = ux
            bc.uy = uy
            bc.uz = uz

class RectCell(PeriodicCell):
    """This class represents the unit cell environment of a regular rectangular tiling.

    Args:
        PeriodicCell (_type_): _description_
    """
    def __init__(self, 
                 width: float,
                 height: float,):
        """The RectCell class represents a regular rectangular tiling in the XY plane where
        the width is along the X-axis (centered at x=0) and the height along the Y-axis (centered at y=0)

        Args:
            width (float): The Cell width
            height (float): The Cell height
        """
        v1 = (width, 0, 0)
        o1 = (-width/2, 0, 0)
        v2 = (0, height, 0)
        o2 = (0, -height/2, 0)
        super().__init__([o1, o2], [v1, v2])
        self.width: float = width
        self.height: float = height
        self.fleft = (o1, v1)
        self.fbot = (o2, v2)
        self.ftop = ((0, height/2, 0), v2)
        self.fright = ((width/2, 0, 0), v1)

    def cell_data(self):
        f1 = SELECTOR_OBJ.inplane(*self.fleft[0], *self.fleft[1])
        f2 = SELECTOR_OBJ.inplane(*self.fright[0], *self.fright[1])
        vec = (self.fright[0][0]-self.fleft[0][0], self.fright[0][1]-self.fleft[0][1], self.fright[0][2]-self.fleft[0][2])
        yield f1, f2, vec

        f1 = SELECTOR_OBJ.inplane(*self.fbot[0], *self.fbot[1])
        f2 = SELECTOR_OBJ.inplane(*self.ftop[0], *self.ftop[1])
        vec = (self.ftop[0][0]-self.fbot[0][0], self.ftop[0][1]-self.fbot[0][1], self.ftop[0][2]-self.fbot[0][2])
        yield f1, f2, vec

    def volume(self, 
               z1: float,
               z2: float) -> GeoPrism:
        xs = np.array([-self.width/2, self.width/2, self.width/2, -self.width/2])
        ys = np.array([-self.height/2, -self.height/2, self.height/2, self.height/2])
        poly = XYPolygon(xs, ys)
        length = z2-z1
        return poly.extrude(length, cs=GCS.displace(0,0,z1))

class HexCell(PeriodicCell):

    def __init__(self,
                 p1: tuple[float, float, float],
                 p2: tuple[float, float, float],
                 p3: tuple[float, float, float]):
        """Generates a Hexagonal periodic tiling by providing 4 coordinates. The layout of the tiling is as following
        Assuming a hexagon with a single vertext at the top and bottom and two vertices on the left and right faces ⬢

        Args:
            p1 (tuple[float, float, float]): left face top vertex
            p2 (tuple[float, float, float]): left face bottom vertex
            p3 (tuple[float, float, float]): bottom vertex
        """
        p1, p2, p3 = [np.array(p) for p in [p1, p2, p3]]
        p4 = -p1
        self.p1: np.ndarray = p1
        self.p2: np.ndarray = p2
        self.p3: np.ndarray = p3
        o1 = (p1+p2)/2
        o2 = (p2+p3)/2
        o3 = (p3+p4)/2
        self.o1 = o1
        self.o2 = o2
        self.o3 = o3
        n1 = _rotnorm(p2-p1)
        n2 = _rotnorm(p3-p2)
        n3 = _rotnorm(p4-p3)
        
        super().__init__([o1, o2, o3], [n1,n2,n3])

        self.f11 = (o1, n1)
        self.f21 = (o2, n2)
        self.f31 = (o3, n3)
        self.f12 = (-o1, n1)
        self.f22 = (-o2, n2)
        self.f32 = (-o3, n3)

    def cell_data(self) -> Generator[FaceSelection, FaceSelection, np.ndarray]:
        nrm = np.linalg.norm

        o = self.o1[:-1]
        n = self.f11[1][:-1]
        w = nrm(self.p2-self.p1)/2
        f1 = SELECTOR_OBJ.inplane(*self.f11[0], *self.f11[1]).exclude(lambda x, y, z: (nrm(np.array([x,y])-o)>w) or (abs((np.array([x,y])-o) @ n ) > 1e-6))
        f2 = SELECTOR_OBJ.inplane(*self.f12[0], *self.f12[1]).exclude(lambda x, y, z: (nrm(np.array([x,y])+o)>w) or (abs((np.array([x,y])+o) @ n ) > 1e-6))
        vec = - (self.p1 + self.p2)
        yield f1, f2, vec

        o = self.o2[:-1]
        n = self.f21[1][:-1]
        w = nrm(self.p3-self.p2)/2
        f1 = SELECTOR_OBJ.inplane(*self.f21[0], *self.f21[1]).exclude(lambda x, y, z: (nrm(np.array([x,y])-o)>w) or (abs((np.array([x,y])-o) @ n ) > 1e-6))
        f2 = SELECTOR_OBJ.inplane(*self.f22[0], *self.f22[1]).exclude(lambda x, y, z: (nrm(np.array([x,y])+o)>w) or (abs((np.array([x,y])+o) @ n ) > 1e-6))
        vec = - (self.p2 + self.p3)
        yield f1, f2, vec
        
        o = self.o3[:-1]
        n = self.f31[1][:-1]
        w = nrm(-self.p1-self.p3)/2
        f1 = SELECTOR_OBJ.inplane(*self.f31[0], *self.f31[1]).exclude(lambda x, y, z: (nrm(np.array([x,y])-o)>w) or (abs((np.array([x,y])-o) @ n ) > 1e-6))
        f2 = SELECTOR_OBJ.inplane(*self.f32[0], *self.f32[1]).exclude(lambda x, y, z: (nrm(np.array([x,y])+o)>w) or (abs((np.array([x,y])+o) @ n ) > 1e-6))
        vec = - (self.p3 - self.p1)
        yield f1, f2, vec

    def volume(self, 
               z1: float,
               z2: float) -> GeoPrism:
        xs, ys, zs = zip(self.p1, self.p2, self.p3)
        xs = np.array(xs)
        ys = np.array(ys)
        xs = np.concatenate([xs, -xs])
        ys = np.concatenate([ys, -ys])
        poly = XYPolygon(xs, ys)
        length = z2-z1
        return poly.extrude(length, cs=GCS.displace(0,0,z1))
