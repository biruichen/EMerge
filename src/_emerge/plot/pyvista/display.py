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

from ...mesh3d import Mesh3D
from ...geometry import GeoObject
from ...selection import FaceSelection, DomainSelection, EdgeSelection, Selection
from ...bc import PortBC
import numpy as np
import pyvista as pv
from typing import Iterable, Literal
from ..display import BaseDisplay

from matplotlib.colors import ListedColormap
### Color scale

# Define the colors we want to use
col1 = np.array([57, 179, 227, 255])/255
col2 = np.array([22, 36, 125, 255])/255
col3 = np.array([33, 33, 33, 255])/255
col4 = np.array([173, 76, 7, 255])/255
col5 = np.array([250, 75, 148, 255])/255

def gen_cmap(mesh, N: int = 256):
    # build a linear grid of data‐values (not strictly needed for pure colormap)
    vmin, vmax = mesh['values'].min(), mesh['values'].max()
    mapping = np.linspace(vmin, vmax, N)
    
    # prepare output
    newcolors = np.empty((N, 4))
    
    # normalized positions of control points: start, middle, end
    control_pos = np.array([0.0, 0.25, 0.5, 0.75, 1]) * (vmax - vmin) + vmin
    # stack control colors
    controls = np.vstack([col1, col2, col3, col4, col5])
    
    # interp each RGBA channel independently
    for chan in range(4):
        newcolors[:, chan] = np.interp(mapping, control_pos, controls[:, chan])
    
    return ListedColormap(newcolors)



def setdefault(options: dict, **kwargs) -> dict:
    """Shorthand for overwriting non-existent keyword arguments with defaults

    Args:
        options (dict): The kwargs dict

    Returns:
        dict: the kwargs dict
    """
    for key in kwargs.keys():
        if key not in options:
            options[key] = kwargs[key]
    return options

def _logscale(dx, dy, dz):
    """
    Logarithmically scales vector magnitudes so that the largest remains unchanged
    and others are scaled down logarithmically.
    
    Parameters:
        dx, dy, dz (np.ndarray): Components of vectors.
        
    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: Scaled dx, dy, dz arrays.
    """
    dx = np.asarray(dx)
    dy = np.asarray(dy)
    dz = np.asarray(dz)

    # Compute original magnitudes
    mags = np.sqrt(dx**2 + dy**2 + dz**2)
    mags_nonzero = np.where(mags == 0, 1e-10, mags)  # avoid log(0)

    # Logarithmic scaling (scaled to max = original max)
    log_mags = np.log10(mags_nonzero)
    log_min = np.min(log_mags)
    log_max = np.max(log_mags)

    if log_max == log_min:
        # All vectors have the same length
        return dx, dy, dz

    # Normalize log magnitudes to [0, 1]
    log_scaled = (log_mags - log_min) / (log_max - log_min)

    # Scale back to original max magnitude
    max_mag = np.max(mags)
    new_mags = log_scaled * max_mag

    # Compute unit vectors
    unit_dx = dx / mags_nonzero
    unit_dy = dy / mags_nonzero
    unit_dz = dz / mags_nonzero

    # Apply scaled magnitudes
    scaled_dx = unit_dx * new_mags
    scaled_dy = unit_dy * new_mags
    scaled_dz = unit_dz * new_mags

    return scaled_dx, scaled_dy, scaled_dz

def _min_distance(xs, ys, zs):
    """
    Compute the minimum Euclidean distance between any two points
    defined by the 1D arrays xs, ys, zs.
    
    Parameters:
        xs (np.ndarray): x-coordinates of the points
        ys (np.ndarray): y-coordinates of the points
        zs (np.ndarray): z-coordinates of the points
    
    Returns:
        float: The minimum Euclidean distance between any two points
    """
    # Stack the coordinates into a (N, 3) array
    points = np.stack((xs, ys, zs), axis=-1)

    # Compute pairwise squared distances using broadcasting
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists_squared = np.sum(diff ** 2, axis=-1)

    # Set diagonal to infinity to ignore zero distances to self
    np.fill_diagonal(dists_squared, np.inf)

    # Get the minimum distance
    min_dist = np.sqrt(np.min(dists_squared))
    return min_dist

def _norm(x, y, z):
    return np.sqrt(np.abs(x)**2 + np.abs(y)**2 + np.abs(z)**2)

def _select(obj: GeoObject | Selection) -> Selection:
    if isinstance(obj, GeoObject):
        return obj.select
    return obj

def _merge(lst: list[GeoObject | Selection]) -> Selection:
    selections = [_select(item) for item in lst]
    dim = selections[0].dim
    all_tags = []
    for item in lst:
        all_tags.extend(_select(item).tags)
    
    if dim==1:
        return EdgeSelection(all_tags)
    elif dim==2:
        return FaceSelection(all_tags)
    elif dim==3:
        return DomainSelection(all_tags)

class PVDisplay(BaseDisplay):

    def __init__(self, mesh: Mesh3D, plotter: pv.Plotter = None):
        self._mesh: Mesh3D = mesh
        if plotter is None:
            plotter = pv.Plotter()
        self._plot: pv.Plotter = plotter

    def show(self):
        self._plot.add_axes()
        self._plot.show()
        self._plot = pv.Plotter()
    
    ## CUSTOM METHODS
    def mesh_volume(self, volume: DomainSelection) -> pv.UnstructuredGrid:
        tets = self._mesh.get_tetrahedra(volume.tags)

        ntets = tets.shape[0]

        cells = np.zeros((ntets,5), dtype=np.int64)

        cells[:,1:] = self._mesh.tets[:,tets].T

        cells[:,0] = 4
        celltypes = np.full(ntets, fill_value=pv.CellType.TETRA, dtype=np.uint8)
        points = self._mesh.nodes.T

        return pv.UnstructuredGrid(cells, celltypes, points)
    
    def mesh_surface(self, surface: FaceSelection) -> pv.UnstructuredGrid:
        tris = self._mesh.get_triangles(surface.tags)

        ntris = tris.shape[0]

        cells = np.zeros((ntris,4), dtype=np.int64)

        cells[:,1:] = self._mesh.tris[:,tris].T

        cells[:,0] = 3
        celltypes = np.full(ntris, fill_value=pv.CellType.TRIANGLE, dtype=np.uint8)
        points = self._mesh.nodes.T

        return pv.UnstructuredGrid(cells, celltypes, points)
    
    def mesh(self, obj: GeoObject | Selection | Iterable) -> pv.UnstructuredGrid:
        if isinstance(obj, Iterable):
            obj = _merge(obj)
        else:
            obj = _select(obj)
        
        if isinstance(obj, DomainSelection):
            return self.mesh_volume(obj)
        elif isinstance(obj, FaceSelection):
            return self.mesh_surface(obj)

    ## OBLIGATORY METHODS
    def add_object(self, obj: GeoObject | Selection | Iterable, *args, **kwargs):

        kwargs = setdefault(kwargs, color=obj.color, opacity=obj.opacity)

        self._plot.add_mesh(self.mesh(obj), *args, **kwargs)

    def add_scatter(self, xs: np.ndarray, ys: np.ndarray, zs: np.ndarray):
        """Adds a scatter point cloud

        Args:
            xs (np.ndarray): The X-coordinate
            ys (np.ndarray): The Y-coordinate
            zs (np.ndarray): The Z-coordinate
        """
        cloud = pv.PolyData(np.array([xs,ys,zs]).T)
        self._plot.add_points(cloud)

    def add_portmode(self, port: PortBC, k0: float, Npoints: int = 10, dv=(0,0,0), XYZ=None,
                      field: Literal['E','H'] = 'E') -> pv.UnstructuredGrid:
        if XYZ:
            X,Y,Z = XYZ
        else:
            X,Y,Z = port.selection.sample(Npoints)
            for x,y,z in zip(X,Y,Z):
                self.add_portmode(port, k0, Npoints, dv, (x,y,z), field)
            return
        
        X = X+dv[0]
        Y = Y+dv[1]
        Z = Z+dv[2]
        xf = X.flatten()
        yf = Y.flatten()
        zf = Z.flatten()

        d = _min_distance(xf, yf, zf)

        F = port.port_mode_3d_global(xf,yf,zf, k0, which=field)

        Fx = F[0,:].reshape(X.shape).T
        Fy = F[1,:].reshape(X.shape).T
        Fz = F[2,:].reshape(X.shape).T

        if field=='H':
            F = np.imag(F.T)
            Fnorm = np.sqrt(Fx.imag**2 + Fy.imag**2 + Fz.imag**2)
        else:
            F = np.real(F.T)
            Fnorm = np.sqrt(Fx.real**2 + Fy.real**2 + Fz.real**2)

        grid = pv.StructuredGrid(X,Y,Z)
        self._plot.add_mesh(grid, scalars = Fnorm, opacity=0.8)

        Emag = F/np.max(Fnorm.flatten())*d*3
        self._plot.add_arrows(np.array([xf,yf,zf]).T, Emag)

    def add_surf(self, 
                 x: np.ndarray,
                 y: np.ndarray,
                 z: np.ndarray,
                 field: np.ndarray,
                 opacity: float = 1.0,
                 **kwargs,):
        """Add a surface plot to the display
        The X,Y,Z coordinates must be a 2D grid of data points. The field must be a real field with the same size.

        Args:
            x (np.ndarray): The x-coordinates
            y (np.ndarray): The y-coordinates
            z (np.ndarray): The z-coordinates
            field (np.ndarray): The field to display
            opacity (float, optional): The opacity. Defaults to 1.0.
        """
        
        grid = pv.StructuredGrid(x,y,z)
        grid['values'] = field.flatten(order='F')
        kwargs = setdefault(kwargs, cmap=gen_cmap(grid))
        self._plot.add_mesh(grid, opacity=opacity, **kwargs)

    def add_quiver(self, x: np.ndarray, y: np.ndarray, z: np.ndarray,
              dx: np.ndarray, dy: np.ndarray, dz: np.ndarray,
              scale: float = 1,
              scalemode: Literal['lin','log'] = 'lin'):
        """Add a quiver plot to the display

        Args:
            x (np.ndarray): The X-coordinates
            y (np.ndarray): The Y-coordinates
            z (np.ndarray): The Z-coordinates
            dx (np.ndarray): The arrow X-magnitude
            dy (np.ndarray): The arrow Y-magnitude
            dz (np.ndarray): The arrow Z-magnitude
            scale (float, optional): The arrow scale. Defaults to 1.
            scalemode (Literal['lin','log'], optional): Wether to scale lin or log. Defaults to 'lin'.
        """
        x = x.flatten()
        y = y.flatten()
        z = z.flatten()
        dx = dx.flatten().real
        dy = dy.flatten().real
        dz = dz.flatten().real
        dmin = _min_distance(x,y,z)

        dmax = np.max(_norm(dx,dy,dz))
        
        Vec = scale * np.array([dx,dy,dz]).T / dmax * dmin 
        Coo = np.array([x,y,z]).T
        if scalemode=='log':
            dx, dy, dz = _logscale(Vec[:,0], Vec[:,1], Vec[:,2])
            Vec[:,0] = dx
            Vec[:,1] = dy
            Vec[:,2] = dz
        self._plot.add_arrows(Coo, Vec)

    def add_contour(self,
                     X: np.ndarray,
                     Y: np.ndarray,
                     Z: np.ndarray,
                     V: np.ndarray,
                     Nlevels: int = 5):
        Vf = V.flatten()
        vmin = np.min(Vf)
        vmax = np.max(Vf)
        grid = pv.StructuredGrid(X,Y,Z)
        grid['values'] = V.flatten(order='F')
        contour = grid.contour(isosurfaces=np.linspace(vmin, vmax, Nlevels))
        self._plot.add_mesh(contour, opacity=0.25)
