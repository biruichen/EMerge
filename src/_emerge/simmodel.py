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
from .mesher import Mesher, unpack_lists
from .geometry import GeoObject
from .physics.edm.emfreq3d import Electrodynamics3D
from .mesh3d import Mesh3D
from .selection import Selector, FaceSelection, Selection
from .logsettings import logger_format
from .geo.modeler import Modeler
from .plot.display import BaseDisplay
from .plot.pyvista import PVDisplay

from typing import Literal, Type
from loguru import logger
import numpy as np
import sys
import gmsh
import inspect
import joblib
from pathlib import Path


class SimulationError(Exception):
    pass

class Simulation3D:

    def __init__(self, 
                 modelname: str, 
                 display: Type[BaseDisplay] = None,
                 loglevel: Literal['DEBUG','INFO','WARNING','ERROR'] = 'INFO',
                 load_file: bool = False,
                 save_file: bool = False):
        """Generate a Simulation3D class object.

        As a minimum a file name should be provided. Additionally you may provide it with any
        class that inherits from BaseDisplay. This will then be used for geometry displaying.

        Args:
            modelname (str): The model name
            display (BaseDisplay, optional): The BaseDisplay class type to use. Defaults to None.
            loglevel ("DEBUG","INFO","WARNING","ERROR, optional): _description_. Defaults to 'INFO'.
        
        """
        caller_file = Path(inspect.stack()[1].filename).resolve()
        base_path = caller_file.parent

        self.modelname = modelname
        self.modelpath = base_path / (modelname.lower()+'_data')
        self.mesher: Mesher = Mesher()
        self.physics: Electrodynamics3D = Electrodynamics3D(self.mesher)
        self.mesh: Mesh3D = Mesh3D(self.mesher)
        self.select: Selector = Selector()
        self.modeler: Modeler = Modeler()
        self.display: PVDisplay = None
        self._geometries: list[GeoObject] = []
        self.set_loglevel(loglevel)

        ## STATES
        self._defined_geometries: bool = False

        if display is not None:
            self.display = display(self.mesh)
        else:
            self.display = PVDisplay(self.mesh)

        self.save_file: bool = save_file
        self.load_file: bool = load_file

        self.obj: dict[str, GeoObject] = dict()

    def __setitem__(self, name: str, value: GeoObject) -> None:
        self.obj[name] = value

    def __getitem__(self, name: str) -> GeoObject:
        return self.obj[name]
    
    def save(self) -> None:
        """Saves the current model in the provided project directory."""
        # Ensure directory exists
        if not self.modelpath.exists():
            self.modelpath.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created directory: {self.modelpath}")

        # Save mesh
        mesh_path = self.modelpath / 'mesh.msh'
        gmsh.write(str(mesh_path))
        logger.info(f"Saved mesh to: {mesh_path}")

        # Pack and save data
        physics = self.physics.pack_data()
        objects = self.obj
        dataset = dict(physics=physics, objects=objects, mesh=self.mesh)
        data_path = self.modelpath / 'simdata.emerge'
        joblib.dump(dataset, str(data_path))
        logger.info(f"Saved simulation data to: {data_path}")

    def load(self) -> None:
        """Loads the model from the project directory."""
        mesh_path = self.modelpath / 'mesh.msh'
        data_path = self.modelpath / 'simdata.emerge'

        if not mesh_path.exists() or not data_path.exists():
            raise FileNotFoundError("Missing required mesh or data file.")

        # Load mesh
        gmsh.open(str(mesh_path))
        logger.info(f"Loaded mesh from: {mesh_path}")
        self.mesh.update()

        # Load data
        datapack = joblib.load(str(data_path))
        self.physics.load_data(datapack['physics'])
        self.obj = datapack['objects']
        self.mesh = datapack['mesh']
        logger.info(f"Loaded simulation data from: {data_path}")

    def set_loglevel(self, loglevel: Literal['DEBUG','INFO','WARNING','ERROR']) -> None:
        handler = {"sink": sys.stdout, "level": loglevel, "format": logger_format}
        logger.configure(handlers=[handler])

    def view(self, 
             selections: list[Selection] = None, 
             use_gmsh: bool = False,
             opacity: float = None,
             show_edges: bool = None) -> None:
        """View the current geometry in either the BaseDisplay object (PVDisplay only) or
        the GMSH viewer.

        Args:
            selections (list[Selection], optional): Additional selections to highlight. Defaults to None.
            use_gmsh (bool, optional): Whether to use the GMSH display. Defaults to False.
            opacity (float, optional): The global opacity of all objects.. Defaults to None.
            show_edges (bool, optional): Whether to show the geometry edges. Defaults to None.
        """
        if not (self.display is not None and self.mesh.defined) or use_gmsh:
            gmsh.model.occ.synchronize()
            gmsh.fltk.run()
            return
        try:
            for obj in self._geometries:
                self.display.add_object(obj, show_edges=show_edges, opacity=opacity)
            if selections:
                [self.display.add_object(sel, color='red', opacity=0.7) for sel in selections]
            self.display.show()
            return
        except NotImplementedError as e:
            logger.warning('The provided BaseDisplay class does not support object display. Please make' \
            'sure that this method is properly implemented.')
    
       
    
    def define_geometry(self, *geometries: list[GeoObject]) -> None:
        """Provide the physics engine with the geometries that are contained and ought to be included
        in the simulation. Please make sure to include all geometries. Its currently unclear how the
        system behaves if only a part of all geometries are included.

        """
        if not geometries:
            geometries = list(self.obj.values())
        self._geometries = unpack_lists(geometries)
        self.mesher.submit_objects(self._geometries)
        self.physics._initialize_bcs()
        self._defined_geometries = True

    
    def generate_mesh(self):
        """Generate the mesh. 
        This can only be done after define_geometry(...) is called and if frequencies are defined.

        Args:
            name (str, optional): The mesh file name. Defaults to "meshname.msh".

        Raises:
            ValueError: ValueError if no frequencies are defined.
        """
        if not self._defined_geometries:
            raise SimulationError('No geometries are defined. Make sure to call .define_geometries(...) first.')
        if self.physics.frequencies is None:
            raise ValueError('No frequencies defined for the simulation. Please set frequencies before generating the mesh.')

        gmsh.model.occ.synchronize()
        self.mesher.set_mesh_size(self.physics.get_discretizer(), self.physics.resolution)

        gmsh.model.mesh.generate(3)
        self.mesh.update()
        gmsh.model.occ.synchronize()
        self.physics.mesh = self.mesh

    def get_boundary(self, face: FaceSelection = None, tags: list[int] = None) -> tuple[np.ndarray, np.ndarray]:
        ''' Return boundary data. 
        
        Parameters
        ----------
        obj: GeoObject
        tags: list[int]
        
        Returns:
        ----------
        nodes: np.ndarray
        triangles: np.ndarray
        '''
        if tags is None:
            tags = face.tags
        tri_ids = self.mesh.get_triangles(tags)
        return self.mesh.nodes, self.mesh.tris[:,tri_ids]

    def step1(self) -> None:
        '''
        Step 1: Initialize the model and create a new geometry.

        >>> with em.Simulation3D("My Model") as model:
                object1 = em.modeler.Box(...)
                object2 = em.modeler.CoaxCyllinder(...)
                model.define_geometry(...)
        '''
        pass
    def step2(self) -> None:
        '''
        Step 2: Create the mesh

        example:
        >>> model.generate_mesh()
        '''
        pass
    def step3(self) -> None:
        '''
        Step 3: Setup the physics

        >>> model.physics.resolution = 0.2
        >>> model.physics.assign(*bcs)
        >>> model.physics.set_frequency(np.linspace(f1, f2, N))
        >>> data = model.run_frequency_domain()
        '''
        pass
    def step4(self) -> None:
        '''
        Step 4: Post processing

        The data is provided as a set of data defined for global variables. Each simulation run gets
        A value for these global variables. Therefor, you can select which variable you want as your inner axis.

        example:
        >>> freq, S21 = data.ax('freq').S(2,1)

        Returns the S21 parameter AND the frequency for which it is  known.
        '''

    def howto_ff(self) -> None:
        '''
        >>> topsurf = model.mesh.boundary_surface(model.select.face.near(0, 0, H).tags)
        >>> Ein, Hin = data.item(0).interpolate(*topsurf.exyz).EH
        >>> theta = np.linspace(-np.pi/2, 1.5*np.pi, 201)
        >>> phi = 0*theta
        >>> E, H = em.physics.edm.stratton_chu(Ein, Hin, topsurf, theta, phi, data.item(0).glob('k0')[0])
        '''
        pass

    def __enter__(self):
        gmsh.initialize()
        if not self.load_file:
            gmsh.model.add(self.modelname)
        else:
            self.load()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self.save_file:
            self.save()
        gmsh.finalize()


   
# class Simulation2D:
#     def __init__(self, modelname: str):
#         self.modelname = modelname
#         self.geo: Geometry = Geometry()
#         self.physics: EMFreqDomain2D = EMFreqDomain2D(self.geo)
#         self.mesh: Mesh2D = None
#         self.resolution: float = 0.2

#     
#     def define_geometry(self, polygons: list[shp.Polygon]) -> None:
#         self.geo._merge_polygons(polygons)
#         self.geo.compile()
#         self.physics._initialize_bcs()

#     
#     def generate_mesh(self, name: str = None, element_order: int = 2) -> Mesh2D:
#         logger.info('Generating mesh')
#         discretizer = self.physics.get_discretizer()
#         logger.info('Updating mesh point disscretization size.')
#         self.geo.update_point_ds(discretizer, self.resolution)
#         logger.info('Calling GMSH mesh routine.')
#         self.geo._commit_gmsh()
#         if name is None:
#             name = self.modelname + '_mesh'
#         logger.info('Generating Mesh2D object')
#         self.mesh = Mesh2D(name, element_order=element_order)
#         logger.info('Mesh complete')
#         return self.mesh
    
#     
#     def update_boundary_conditions(self):
#         logger.info('Updating boundary conditions')
#         for bc in self.physics.boundary_conditions:
#             logger.debug(f'Parsing BC: {bc}')
#             if bc.dimension is BCDimension.EDGE:
#                 vertices = []
#                 for tag in bc.tags:
#                     vertices.append(self.mesh.get_edge(tag))
#                 bc.node_indices = [np.array(_list, dtype=np.int32) for _list in _stitch_lists(vertices)]
#             elif bc.dimension is BCDimension.NODE:
#                 vertices = []
#                 for tag in bc.tags:
#                     vertices.append(self.mesh.get_point(tag))
#                 bc.node_indices = vertices
#             else:
#                 raise NotImplementedError(f'Boundary condition type {bc.dimension} is not implemented yet.')
    
#     def run_frequency_domain(self, solver: callable = None) -> FEMBasis:
#         logger.info('Starting frequency domain simulation routine.')
#         if self.mesh is None:
#             logger.info('No mesh detected.')
#             self.generate_mesh()
#         self.update_boundary_conditions()
#         self.physics.solve(self.mesh, solver=solver)
        
#         return self.physics.solution

#     def run_eigenmodes(self, nsolutions: int = 6) -> FEMBasis:
#         logger.info('Starting eigenmode simulation routine.')
#         if self.mesh is None:
#             logger.info('No mesh detected.')
#             self.generate_mesh()
#         self.update_boundary_conditions()
#         self.physics.eigenmode(self.mesh, num_sols=nsolutions)
#         return self.physics.solution
    
#     def __enter__(self):
#         gmsh.initialize()
#         gmsh.model.add(self.modelname)
#         return self

#     def __exit__(self, exc_type, exc_value, traceback):
#         gmsh.finalize()

