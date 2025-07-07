from __future__ import annotations
from ..geometry import GeoEdge, GeoObject, GeoPoint, GeoPolygon, GeoSurface, GeoVolume
from .operations import subtract, add, embed
from .shapes import Box, Alignment, CoaxCyllinder, Cyllinder, Sphere, Plate, XYPlate
from ..cs import CoordinateSystem, GCS
from .pcb import PCBLayouter
from collections import defaultdict
from typing import Iterable, Set
from ..material import Material, AIR
from loguru import logger
from typing import TypeVar
from .modeler import Modeler

T = TypeVar('T', GeoSurface, GeoVolume)

class NamePool:
    """
    Allocate unique names like  'Vol0', 'Vol1', 'Face0', …
    
    Example
    -------
    >>> pool = NamePool(already_taken={"Vol0"})
    >>> pool.get("Vol")
    'Vol1'
    >>> pool.get("Face")
    'Face0'
    >>> pool.reserve("Vol3")        # mark an external name as used
    >>> pool.get("Vol")
    'Vol2'
    """

    def __init__(self, already_taken: Iterable[str] | None = None) -> None:
        self._taken: Set[str] = set(already_taken or [])
        # one counter per prefix, starts from 0 by default
        self._next_idx: defaultdict[str, int] = defaultdict(int)

    # -----------------------------------------------------------------
    def reserve(self, name: str) -> None:
        """Tell the pool that *name* is already in use elsewhere."""
        self._taken.add(name)

    # -----------------------------------------------------------------
    def get(self, prefix: str) -> str:
        """Return the next unused label for the given *prefix*."""
        i = self._next_idx[prefix]

        while True:                       # loop until we hit a free slot
            candidate = f"{prefix}{i}"
            if candidate not in self._taken:
                self._taken.add(candidate)
                self._next_idx[prefix] = i + 1
                return candidate
            i += 1                        # skip names that were reserved

def _package(arg):
    if not isinstance(arg, (tuple, list)):
        arg = (arg,)
    
    return tuple(a for a in arg if isinstance(a, GeoObject))

class _Interceptor:

    def __init__(self, classobj, builder: Builder):
        self._classobj = classobj
        self._builder: Builder = builder

    def __getattr__(self, name):
        _attr = getattr(self._classobj, name)
        if callable(_attr):

            def _call_wrapper(*args, **kwargs):
                returnvars = _attr(*args, **kwargs)
                self._builder.add_object(*_package(returnvars))
                return self.intercept(returnvars)
            return _call_wrapper
        else:
            return _attr

    def intercept(self, returnvars):
        if isinstance(returnvars, type(self._classobj)):
            return self
        return returnvars
    
class _PCBProxy:

    def __init__(self, builder):
        self.builder = builder
        self.pcbs: dict[int, PCBLayouter] = {}

    def new(self, number: int = 0) -> type[PCBLayouter]:

        def wrapper(*args, **kwargs):
            self.pcbs[number] = PCBLayouter(*args, **kwargs)
            return _Interceptor(self.pcbs[number], self.builder)
        return wrapper

    def __call__(self, number: int) -> PCBLayouter:
        return _Interceptor(self.pcbs[number], self.builder)



class Builder:


    def __init__(self):

        self.volumes: dict[str, GeoVolume] = dict()
        self.surfaces: dict[str, GeoSurface] = dict()
        self.edges: dict[str, GeoEdge] = dict()
        self.points: dict[str, GeoPoint] = dict()
        self.pcbs: _PCBProxy = _PCBProxy(self)
        self._namepool: NamePool = NamePool()
        
        self._name: str = None

        self.Box: type[Box] = self._constructor('Box',Box)
        self.Sphere: type[Sphere] = self._constructor('Sphere',Sphere)
        self.Cyllinder: type[Cyllinder] = self._constructor('Cyllinder', Cyllinder)
        self.CoaxCyllinder: type[CoaxCyllinder] = self._constructor('Coax', CoaxCyllinder)
        self.Plate: type[Plate] = self._constructor('Plate',Plate)
        self.XYPlate: type[XYPlate] = self._constructor('Plate', XYPlate)

        self.modeler: Modeler = Modeler()
    

    def all_geometries(self) -> list[GeoObject]:
        objects = []
        for dim in range(4):
            for obj in self._obj(dim).values():
                if obj._exists:
                    objects.append(obj)
        return objects
    
    def _constructor(self, name: str, constructor: type):
        def constr(*args, **kwargs):
            obj = constructor(*args, **kwargs)
            self._obj(obj.dim)[self._get_name(name)] = obj
            return obj
        return constr
    
    @property
    def geodict(self) -> dict[str, GeoObject]:
        dct = dict()
        for i in range(4):
            dct.update(self._obj(i))
        return dct
    
    def _get_name(self, name: str) -> str:
        if self._name is not None:
            name = self._name
            self._name = None
            return name
        return self._namepool.get(name)

    def _obj(self, dim: int) -> dict[int, GeoObject]:
        if dim==0:
            return self.points
        elif dim==1:
            return self.edges
        elif dim==2:
            return self.surfaces
        else:
            return self.volumes
        
    def call(self, name: str) -> Builder:
        self._name = name
        return self

    def __getitem__(self, name: str) -> GeoObject:
        return self.geodict[name]
    
    def add_object(self, *geoms: GeoObject):
        """Add a series of geometries to the Builder
        """
        for geo in geoms:
            self._obj(geo.dim)[self._get_name('Obj')] = geo

    def add(self, main: T, tool: T, 
             remove_object: bool = True,
             remove_tool: bool = True) -> T:
        ''' Adds two GMSH objects together, returning a new object that is the union of the two.
        
        Parameters
        ----------
        main : GeoSurface | GeoVolume
        tool : GeoSurface | GeoVolume
        remove_object : bool, optional
            If True, the main object will be removed from the model after the operation. Default is True.
        remove_tool : bool, optional
            If True, the tool object will be removed from the model after the operation. Default is True.
        
        Returns
        -------
        GeoSurface | GeoVolume
            A new object that is the union of the main and tool objects.
        '''
        new = add(main, tool, remove_object, remove_tool)
        self._obj(new.dim)[self._get_name('Bool')] = new
        return new

    def remove(self, main: T, tool: T, 
             remove_object: bool = True,
             remove_tool: bool = True) -> T:
        ''' Subtractes a tool object GMSH from the main object, returning a new object that is the difference of the two.
        
        Parameters
        ----------
        main : GeoSurface | GeoVolume
        tool : GeoSurface | GeoVolume
        remove_object : bool, optional
            If True, the main object will be removed from the model after the operation. Default is True.
        remove_tool : bool, optional
            If True, the tool object will be removed from the model after the operation. Default is True.
        
        Returns
        -------
        GeoSurface | GeoVolume
            A new object that is the difference of the main and tool objects.
        '''

        new = subtract(main, tool, remove_object, remove_tool)
        self._obj(new.dim)[self._get_name('Bool')] = new
        return new
    
    def embed(self, main: GeoVolume, other: GeoSurface) -> None:
        ''' Embeds a surface into a volume in the GMSH model.
        Parameters
        ----------
        main : GeoVolume
            The volume into which the surface will be embedded.
        other : GeoSurface
            The surface to be embedded into the volume.
        
        Returns
        -------
        None
        '''
        embed(main, other)

    