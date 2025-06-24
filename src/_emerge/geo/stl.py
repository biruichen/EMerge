import gmsh
from ..geometry import GeoVolume
from pathlib import Path
import numpy as np
class STLObject(GeoVolume):

    def __init__(self, filename: str):
        stl_path = Path(filename)
        if not stl_path.exists:
            raise ValueError(f'File with name {stl_path} does not exist.')
        gmsh.open(str(stl_path))
        angle = 180.
        includeBoundary = True
        forceParametrizablePatches = True
        curveAngle = 180
        gmsh.model.mesh.classifySurfaces(angle * np.pi / 180., includeBoundary,
                                     forceParametrizablePatches,
                                     curveAngle * np.pi / 180.)
        gmsh.model.mesh.createGeometry()

        #gmsh.open(str(stl_path))
        #gmsh.model.mesh.classifySurfaces(90 * 3.1415 / 180., True, True)
        #gmsh.model.mesh.createGeometry()