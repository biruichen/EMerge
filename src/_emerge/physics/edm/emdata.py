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
from ...dataset import SimData, DataSet
from ...elements.femdata import FEMBasis
from dataclasses import dataclass
import numpy as np
from typing import Literal
from loguru import logger
from .adaptive_freq import SparamModel
from ...cs import Axis, _parse_axis
from ...selection import FaceSelection
from ...geometry import GeoSurface

EMField = Literal[
    "er", "ur", "freq", "k0",
    "_Spdata", "_Spmapping", "_field", "_basis",
    "Nports", "Ex", "Ey", "Ez",
    "Hx", "Hy", "Hz",
    "mode", "beta",
]

def arc_on_plane(ref_dir, normal, angle_range_deg, num_points=100):
    """
    Generate theta/phi coordinates of an arc on a plane.

    Parameters
    ----------
    ref_dir : tuple (dx, dy, dz)
        Reference direction (angle zero) lying in the plane.
    normal : tuple (nx, ny, nz)
        Plane normal vector.
    angle_range_deg : tuple (deg_start, deg_end)
        Start and end angle of the arc in degrees.
    num_points : int
        Number of points along the arc.

    Returns
    -------
    theta : ndarray
        Array of theta angles (radians).
    phi : ndarray
        Array of phi angles (radians).
    """
    d = np.array(ref_dir, dtype=float)
    n = np.array(normal, dtype=float)

    # Normalize normal
    n = n / np.linalg.norm(n)

    # Project d into the plane
    d_proj = d - np.dot(d, n) * n
    if np.linalg.norm(d_proj) < 1e-12:
        raise ValueError("Reference direction is parallel to the normal vector.")

    e1 = d_proj / np.linalg.norm(d_proj)
    e2 = np.cross(n, e1)

    # Generate angles along the arc
    angles_deg = np.linspace(angle_range_deg[0], angle_range_deg[1], num_points)
    angles_rad = np.deg2rad(angles_deg)

    # Create unit vectors along the arc
    vectors = np.outer(np.cos(angles_rad), e1) + np.outer(np.sin(angles_rad), e2)

    # Convert to spherical angles
    ux, uy, uz = vectors[:,0], vectors[:,1], vectors[:,2]

    theta = np.arcsin(uz)         # theta = arcsin(z)
    phi = np.arctan2(uy, ux)      # phi = atan2(y, x)

    return theta, phi

def renormalise_s(S: np.ndarray,
                  Zn: np.ndarray,
                  Z0: complex | float = 50) -> np.ndarray:
    S   = np.asarray(S,  dtype=complex)
    Zn  = np.asarray(Zn, dtype=complex)
    N   = S.shape[1]
    if S.shape[1:3] != (N, N):
        raise ValueError("S must have shape (M, N, N) with same N on both axes")
    if Zn.shape != (N,):
        raise ValueError("Zn must be a length-N vector")

    # Constant matrices that do not depend on frequency
    Wref      = np.diag(np.sqrt(Zn))          # √Zn on the diagonal
    W0_inv_sc = 1 / np.sqrt(Z0)               # scalar because Z0 is common
    I_N       = np.eye(N, dtype=complex)

    M = S.shape[2]
    S0 = np.empty_like(S)

    for k in range(M):
        Sk = S[k, :, :]

        # Z  = Wref (I + S) (I – S)⁻¹ Wref
        Zk = Wref @ (I_N + Sk) @ np.linalg.inv(I_N - Sk) @ Wref

        # A  = W0⁻¹ Z W0⁻¹  → because W0 = √Z0·I → A = Z / Z0
        Ak = Zk * (W0_inv_sc ** 2)            # same as Zk / Z0

        # S0 = (A – I)(A + I)⁻¹
        S0[k, :, :] = (Ak - I_N) @ np.linalg.inv(Ak + I_N)

    return S0

@dataclass
class Sparam:
    """
    S-parameter matrix indexed by arbitrary port/mode labels (ints or floats).
    Internally stores a square numpy array; externally uses your mapping
    to translate (port1, port2) → (i, j).
    """
    def __init__(self, port_nrs: list[int | float]) -> None:
        # build label → index map
        self.map: dict[int | float, int] = {label: idx 
                                            for idx, label in enumerate(port_nrs)}
        n = len(port_nrs)
        # zero‐initialize the S‐parameter matrix
        self.arry: np.ndarray = np.zeros((n, n), dtype=np.complex128)

    def get(self, port1: int | float, port2: int | float) -> complex:
        """
        Return the S-parameter S(port1, port2).
        Raises KeyError if either port1 or port2 is not in the mapping.
        """
        try:
            i = self.map[port1]
            j = self.map[port2]
        except KeyError as e:
            raise KeyError(f"Port/mode {e.args[0]!r} not found in mapping") from None
        return self.arry[i, j]

    def set(self, port1: int | float, port2: int | float, value: complex) -> None:
        """
        Set the S-parameter S(port1, port2) = value.
        Raises KeyError if either port1 or port2 is not in the mapping.
        """
        try:
            i = self.map[port1]
            j = self.map[port2]
        except KeyError as e:
            raise KeyError(f"Port/mode {e.args[0]!r} not found in mapping") from None
        self.arry[i, j] = value

    # allow S(param1, param2) → complex, as before
    def __call__(self, port1: int | float, port2: int | float) -> complex:
        return self.get(port1, port2)

    # allow array‐style access: S[1, 1] → complex
    def __getitem__(self, key: tuple[int | float, int | float]) -> complex:
        port1, port2 = key
        return self.get(port1, port2)

    # allow array‐style setting: S[1, 2] = 0.3 + 0.1j
    def __setitem__(
        self,
        key: tuple[int | float, int | float],
        value: complex
    ) -> None:
        port1, port2 = key
        self.set(port1, port2, value)

@dataclass
class PortProperties:
    port_number: int | None = None
    k0: float | None= None
    beta: float | None = None
    Z0: float | None = None
    Pout: float | None = None
    mode_number: int = 1
    
class EMDataSet(DataSet):
    """The EMDataSet class stores solution data of FEM Time Harmonic simulations.

    
    """
    def __init__(self, **vars):
        self.er: np.ndarray = None
        self.ur: np.ndarray = None
        self.freq: float = None
        self.k0: float = None
        self.Sp: Sparam = None
        self._fields: dict[np.ndarray] = dict()
        self._mode_field: np.ndarray = None
        self.excitation: dict[int, complex] = dict()
        self.basis: FEMBasis = None
        self.Nports: int = None
        self.Ex: np.ndarray = None
        self.Ey: np.ndarray = None
        self.Ez: np.ndarray = None
        self.Hx: np.ndarray = None
        self.Hy: np.ndarray = None
        self.Hz: np.ndarray = None
        self.port_modes: list[PortProperties] = []
        self.mode: int = None
        self.beta: int = None

        super().__init__(**vars)

    @property
    def _field(self) -> np.ndarray:
        if self._mode_field is not None:
            return self._mode_field
        return sum([self.excitation[mode.port_number]*self._fields[mode.port_number] for mode in self.port_modes]) 
    
    def set_field_vector(self) -> None:
        self.excitation = {key: 0.0 for key in self._fields.keys()}
        self.excitation[self.port_modes[0].port_number] = 1.0 + 0j

    @property
    def EH(self) -> tuple[np.ndarray, np.ndarray]:
        ''' Return the electric and magnetic field as a tuple of numpy arrays '''
        return np.array([self.Ex, self.Ey, self.Ez]), np.array([self.Hx, self.Hy, self.Hz])
    
    @property
    def E(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        ''' Return the electric field as a tuple of numpy arrays '''
        return self.Ex, self.Ey, self.Ez
    
    @property
    def normE(self) -> np.ndarray:
        """The complex norm of the E-field
        """
        return np.sqrt(np.abs(self.Ex)**2 + np.abs(self.Ey)**2 + np.abs(self.Ez)**2)
    
    @property
    def normH(self) -> np.ndarray:
        """The complex norm of the H-field"""
        return np.sqrt(np.abs(self.Hx)**2 + np.abs(self.Hy)**2 + np.abs(self.Hz)**2)
    
    @property
    def Emat(self) -> np.ndarray:
        return np.array([self.Ex, self.Ey, self.Ez])
    
    @property
    def Hmat(self) -> np.ndarray:
        return np.array([self.Hx, self.Hy, self.Hz])

    @property
    def H(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        ''' Return the magnetic field as a tuple of numpy arrays '''
        return self.Hx, self.Hy, self.Hz
    
    def init_sp(self, portnumbers: list[int | float]) -> None:
        self.Sp = Sparam(portnumbers)

    def add_port_properties(self, 
                            port_number: int,
                            mode_number: int,
                            k0: float,
                            beta: float,
                            Z0: float,
                            Pout: float) -> None:
        self.port_modes.append(PortProperties(port_number=port_number,
                                              mode_number=mode_number,
                                              k0 = k0,
                                              beta=beta,
                                              Z0=Z0,
                                              Pout=Pout))
        
    def write_S(self, i1: int | float, i2: int | float, value: complex) -> None:
        self.Sp[i1,i2] = value

    def interpolate(self, xs: np.ndarray, ys: np.ndarray, zs: np.ndarray) -> EMDataSet:
        ''' Interpolate the dataset in the provided xs, ys, zs values'''
        shp = xs.shape
        xf = xs.flatten()
        yf = ys.flatten()
        zf = zs.flatten()
        Ex, Ey, Ez = self.basis.interpolate(self._field, xf, yf, zf)
        self.Ex = Ex.reshape(shp)
        self.Ey = Ey.reshape(shp)
        self.Ez = Ez.reshape(shp)

        
        constants = 1/ (-1j*2*np.pi*self.freq*(self.ur*4*np.pi*1e-7) )
        Hx, Hy, Hz = self.basis.interpolate_curl(self._field, xf, yf, zf, constants)
        self.Hx = Hx.reshape(shp)
        self.Hy = Hy.reshape(shp)
        self.Hz = Hz.reshape(shp)

        self._x = xs
        self._y = ys
        self._z = zs
        return self

    def cutplane(self, 
                     ds: float,
                     x: float=None,
                     y: float=None,
                     z: float=None) -> EMDataSet:
        xb, yb, zb = self.basis.bounds
        xs = np.linspace(xb[0], xb[1], int((xb[1]-xb[0])/ds))
        ys = np.linspace(yb[0], yb[1], int((yb[1]-yb[0])/ds))
        zs = np.linspace(zb[0], zb[1], int((zb[1]-zb[0])/ds))
        if x is not None:
            Y,Z = np.meshgrid(ys, zs)
            X = x*np.ones_like(Y)
        if y is not None:
            X,Z = np.meshgrid(xs, zs)
            Y = y*np.ones_like(X)
        if z is not None:
            X,Y = np.meshgrid(xs, ys)
            Z = z*np.ones_like(Y)
        self.interpolate(X,Y,Z)
        return self
    
    def grid(self, ds: float) -> EMDataSet:
        """Interpolate a uniform grid sampled at ds

        Args:
            ds (float): the sampling distance

        Returns:
            This object
        """
        xb, yb, zb = self.basis.bounds
        xs = np.linspace(xb[0], xb[1], int((xb[1]-xb[0])/ds))
        ys = np.linspace(yb[0], yb[1], int((yb[1]-yb[0])/ds))
        zs = np.linspace(zb[0], zb[1], int((zb[1]-zb[0])/ds))
        X, Y, Z = np.meshgrid(xs, ys, zs)
        self.interpolate(X,Y,Z)
        return self
    
    def S(self, i1: int, i2: int) -> complex:
        ''' Returns the S-parameter S(i1,i2)'''
        return self.Sp(i1, i2)

    def vector(self, field: Literal['E','H'], metric: Literal['real','imag','complex'] = 'real') -> tuple[np.ndarray, np.ndarray,np.ndarray,np.ndarray,np.ndarray,np.ndarray]:
        """Returns the X,Y,Z,Fx,Fy,Fz data to be directly cast into plot functions.

        The field can be selected by a string literal. The metric of the complex vector field by the metric.
        For animations, make sure to always use the complex metric.

        Args:
            field ('E','H'): The field to return
            metric ([]'real','imag','complex'], optional): the metric to impose on the field. Defaults to 'real'.

        Returns:
            tuple[np.ndarray,...]: The X,Y,Z,Fx,Fy,Fz arrays
        """
        if field=='E':
            Fx, Fy, Fz = self.Ex, self.Ey, self.Ez
        elif field=='H':
            Fx, Fy, Fz = self.Hx, self.Hy, self.Hz
        
        if metric=='real':
            Fx, Fy, Fz = Fx.real, Fy.real, Fz.real
        elif metric=='imag':
            Fx, Fy, Fz = Fx.imag, Fy.imag, Fz.imag
        
        return self._x, self._y, self._z, Fx, Fy, Fz
    
    def scalar(self, field: Literal['Ex','Ey','Ez','Hx','Hy','Hz','normE','normH'], metric: Literal['abs','real','imag','complex'] = 'real') -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Returns the data X, Y, Z, Field based on the interpolation

        For animations, make sure to select the complex metric.

        Args:
            field (str): The field to plot
            metric (str, optional): The metric to impose on the plot. Defaults to 'real'.

        Returns:
            (X,Y,Z,Field): The coordinates plus field scalar
        """
        field = getattr(self, field)
        if metric=='abs':
            field = np.abs(field)
        elif metric=='real':
            field = field.real
        elif metric=='imag':
            field = field.imag
        elif metric=='complex':
            field = field
        return self._x, self._y, self._z, field
    
    def farfield_2d(self,ref_direction: tuple[float,float,float] | Axis,
                         plane_normal: tuple[float,float,float] | Axis,
                         faces: FaceSelection | GeoSurface,
                         ang_range: tuple[float, float] = (-180, 180),
                         Npoints: int = 201,
                         origin: tuple[float, float, float] = (0,0,0),
                         ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        from .sc import stratton_chu
        surface = self.basis.mesh.boundary_surface(faces.tags, origin)
        self.interpolate(*surface.exyz)
        refdir = _parse_axis(ref_direction).np
        plane_normal = _parse_axis(plane_normal).np

        theta, phi = arc_on_plane(refdir, plane_normal, ang_range, Npoints)
        E,H = stratton_chu(self.E, self.H, surface, theta, phi, self.k0)
        angs = np.linspace(*ang_range, Npoints)*np.pi/180
        return angs, E ,H
    
class _DataSetProxy: 
    """
    A “ghost” wrapper around a real DataSet.
    Any attr/method access is intercepted here first.
    """
    def __init__(self, field: str, dss: DataSet):
        # stash both the SimData (in case you need context)
        # and the real DataSet
        self._field = field
        self._dss = dss

    def __getattribute__(self, name: str):
       

        if name in ('_field','_dss'):
            return object.__getattribute__(self, name)
        xax = []
        yax = []
        field = object.__getattribute__(self, '_field')
        if callable(getattr(self._dss[0], name)):
            def wrapped(*args, **kwargs):
                
                for ds in self._dss:
                    # 1) grab the real attribute
                    xval = getattr(ds, field)
                    func = getattr(ds, name)
                    
                    yval = func(*args, **kwargs)
                    xax.append(xval)
                    yax.append(yval)
                return np.array(xax), np.array(yax)
            return wrapped
        else:
            for ds in self._dss:
                xax.append(getattr(ds, field))
                yax.append(getattr(ds, name))
            return np.array(xax), np.array(yax)


class EMSimData(SimData[EMDataSet]):
    """The EMSimData class contains all EM simulation data from a Time Harmonic simulation
    along all sweep axes.
    """
    datatype: type = EMDataSet
    def __init__(self):
        super().__init__()
        #self._basis: FEMBasis = basis
        self._injections = dict()
        self._axis = 'freq'

    def __getitem__(self, field: EMField) -> np.ndarray:
        return getattr(self, field)

    def howto(self) -> None:
        """To access data in the EMSimData class use the .ax method to extract properties selected
        along an access of global variables. The axes are all global properties that the EMDatasets manage.
        
        For example the following would return all S(2,1) parameters along the frequency axis.
        
        >>> freq, S21 = dataset.ax('freq').S(2,1)

        Alternatively, one can manually select any solution indexed in order of generation using.

        >>> S21 = dataset.item(3).S(2,1)

        To find the E or H fields at any coordinate, one can use the Dataset's .interpolate method. 
        This method returns the same Dataset object after which the computed fields can be accessed.

        >>> Ex = dataset.item(3).interpolate(xs,ys,zs).Ex

        Lastly, to find the solutions for a given frequency or other value, you can also just call the dataset
        class:
        
        >>> Ex, Ey, Ez = dataset(freq=1e9).interpolate(xs,ys,zs).E

        """

    def ax(self, field: EMField) -> EMDataSet:
        """Return a EMDataSet proxy object that you can request properties for along a provided axis.

        The EMSimData class contains a list of EMDataSet objects. Any global variable like .freq of the 
        EMDataSet object can be used as inner-axes after which the outer axis can be selected as if
        you are extract a single one.

        Args:
            field (EMField): The global field variable to select the data along

        Returns:
            EMDataSet: An EMDataSet object (actually a proxy for)

        Example:
        The following will select all S11 parameters along the frequency axis:

        >>> freq, S11 = dataset.ax('freq').S(1,1)

        """
        # find the real DataSet
        return _DataSetProxy(field, self.datasets)

    def model_S(self, i: int, j: int, Npoles: int = 10, inc_real: bool = False) -> SparamModel:
        """Returns an S-parameter model object that can be sampled at a dense frequency range.
        The S-parameter model object uses vector fitting inside the datasets frequency points
        to determine a model for the linear system.

        Args:
            i (int): The first S-parameter index
            j (int): The second S-parameter index
            Npoles (int, optional): The number of poles to use (approx 2x divice order). Defaults to 10.
            inc_real (bool, optional): Wether to allow for a real-pole. Defaults to False.

        Returns:
            SparamModel: The SparamModel object
        """
        fs, S = self.ax('freq').S(i,j)
        return SparamModel(fs, S, n_poles=Npoles, inc_real=inc_real)

    def model_S_matrix(self, frequencies: np.ndarray,
                       Npoles: int = 10,
                       inc_real: bool = False) -> np.ndarray:
        """Generates a full S-parameter matrix on the provided frequency points using the Vector Fitting algorithm.

        This function output can be used directly with the .save_matrix() method.

        Args:
            frequencies (np.ndarray): The sample frequencies
            Npoles (int, optional): The number of poles to fit. Defaults to 10.
            inc_real (bool, optional): Wether allow for a real pole. Defaults to False.

        Returns:
            np.ndarray: The (Nf,Np,Np) S-parameter matrix
        """
        Nports = len(self.datasets[0].excitation)
        nfreq = frequencies.shape[0]

        Smat = np.zeros((nfreq,Nports,Nports), dtype=np.complex128)
        
        for i in range(1,Nports+1):
            for j in range(1,Nports+1):
                S = self.model_S(i,j)(frequencies)
                Smat[:,i-1,j-1] = S

        return Smat

    def export_touchstone(self, 
                          filename: str,
                          Z0ref: float = None,
                          format: Literal['RI','MA','DB'] = 'RI'):
        """Export the S-parameter data to a touchstone file

        This function assumes that all ports are numbered in sequence 1,2,3,4... etc with
        no missing ports. Otherwise it crashes. Will be update/improved soon with more features.

        Additionally, one may provide a reference impedance. If this argument is provided, a port impedance renormalization
        will be performed to that common impedance.

        Args:
            filename (str): The File name
            Z0ref (float): The reference impedance to normalize to. Defaults to None
            format (Literal[DB, RI, MA]): The dataformat used in the touchstone file.
        """
        
        logger.info(f'Exporting S-data to {filename}')
        Nports = len(self.datasets[0].excitation)
        freqs, _ = self.ax('freq').k0

        Smat = np.zeros((len(freqs),Nports,Nports), dtype=np.complex128)
        
        for i in range(1,Nports+1):
            for j in range(1,Nports+1):
                _, S = self.ax('freq').S(i,j)
                Smat[:,i-1,j-1] = S
        
        self.save_smatrix(filename, Smat, freqs, format=format, Z0ref=Z0ref)

    def save_smatrix(self, 
                     filename: str,
                     Smatrix: np.ndarray,
                     frequencies: np.ndarray, 
                     Z0ref: float = None,
                     format: Literal['RI','MA','DB'] = 'RI') -> None:
        """Save an S-parameter matrix to a touchstone file.
        
        Additionally, a reference impedance may be supplied. In this case, a port renormalization will be performed on the S-matrix.

        Args:
            filename (str): The filename
            Smatrix (np.ndarray): The S-parameter matrix with shape (Nfreq, Nport, Nport)
            frequencies (np.ndarray): The frequencies with size (Nfreq,)
            Z0ref (float, optional): An optional reference impedance to normalize to. Defaults to None.
            format (Literal["RI","MA",'DB], optional): The S-parameter format. Defaults to 'RI'.
        """
        from .touchstone import generate_touchstone

        if Z0ref is not None:
            Z0s = [port.Z0 for port in self.datasets[0].port_modes]
            logger.debug(f'Renormalizing impedances {Z0s}Ω to {Z0ref}Ω')
            Smatrix = renormalise_s(Smatrix, Z0s, Z0ref)


        generate_touchstone(filename, frequencies, Smatrix, format)
        
        logger.info('Export complete!')