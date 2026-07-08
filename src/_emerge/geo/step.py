import gmsh
from ..geometry import GeoPoint, GeoEdge, GeoVolume, GeoSurface, GeoObject
from pathlib import Path
import numpy as np

class STEPItems:

    def __init__(self, filename: str, unit: float = 1.0):
        """Imports the provided STEP file.
        Specify the unit in case of scaling issues where mm units are not taken into consideration.

        Args:
            filename (str): The filename
            unit (float, optional): The STEP file size unit. Defaults to 1.0.

        Raises:
            FileNotFoundError: If a file does not exist
        """
        stl_path = Path(filename)
        gmsh.option.setNumber("Geometry.OCCScaling", 1)
        gmsh.option.setNumber("Geometry.OCCImportLabels", 2) 

        if not stl_path.exists:
            raise FileNotFoundError(f'File with name {stl_path} does not exist.')
        
        dimtags = gmsh.model.occ.import_shapes(filename, format='step')

        gmsh.model.occ.affine_transform(dimtags, np.array([unit, 0, 0, 0,
                                                           0, unit, 0, 0,
                                                           0, 0, unit, 0,
                                                           0, 0, 0, 1]))
        #dimtags = gmsh.model.occ.heal_shapes(dimtags, tolerance=1e-6)
        
        self.points: dict[str, GeoPoint] = dict()
        self.edges: dict[str, GeoEdge] = dict()
        self.surfaces: dict[str, GeoSurface] = dict()
        self.volumes: dict[str, GeoVolume] = dict()

        i = 0
        for dim, tag in dimtags:
            name = gmsh.model.getPhysicalName(dim, tag)
            if name == '':
                name = f'Obj{i}'
                i+=1
            if dim == 0:
                self.points[name] = GeoPoint(tag)
            elif dim == 1:
                self.edges[name] = GeoEdge(tag)
            elif dim == 2:
                self.surfaces[name] = GeoSurface(tag)
            elif dim == 3:
                self.volumes[name] = GeoVolume(tag)
    
    @property
    def _dicts(self):
        yield self.points
        yield self.edges
        yield self.surfaces
        yield self.volumes

    @property
    def objects(self) -> tuple[GeoObject,...]:
        objects = tuple()
        for dct in self._dicts:
            objects = objects + tuple(dct.values())
        return objects
    
    def __getitem__(self, name: str) -> GeoObject:
        if name in self.points:
            return self.points[name]
        elif name in self.edges:
            return self.edges[name]
        elif name in self.surfaces:
            return self.surfaces[name]
        elif name in self.volumes:
            return self.volumes[name]
            