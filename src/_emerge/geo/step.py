import gmsh
from ..geometry import GeoPoint, GeoEdge, GeoVolume, GeoSurface
from pathlib import Path
import numpy as np

class STEPObject(GeoVolume):

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
        if not stl_path.exists:
            raise FileNotFoundError(f'File with name {stl_path} does not exist.')
        
        dimtags = gmsh.model.occ.import_shapes(filename)
        
        

        gmsh.model.occ.affine_transform(dimtags, np.array([unit, 0, 0, 0,
                                                           0, unit, 0, 0,
                                                           0, 0, unit, 0,
                                                           0, 0, 0, 1]))
        dimtags = gmsh.model.occ.heal_shapes(dimtags, tolerance=1e-4)
        tags = [dt[1] for dt in dimtags if dt[0]==3]
        
        gmsh.model.occ.synchronize()
        
        super().__init__(tags)

