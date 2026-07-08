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
from .physics.microwave.microwave_3d import Microwave3D
from .mesh3d import Mesh3D
from .selection import Selector, FaceSelection, Selection
from .logsettings import logger_format
from .geo.builder import Builder
from .plot.display import BaseDisplay
from .plot.pyvista import PVDisplay
from .dataset import SimulationDataset
from .periodic import PeriodicCell
from typing import Literal, Type, Generator, Any
from loguru import logger
import numpy as np
import sys
import gmsh
import joblib
import os
import inspect
from pathlib import Path
from atexit import register
import signal

_GMSH_ERROR_TEXT = """
--------------------------
Known problems/solutions:
(1) - PLC Error:  A segment and a facet intersect at point
    This can be caused when approximating thin curved volumes. Try to decrease the mesh size for that region.
--------------------------
"""

class SimulationError(Exception):
    pass


class Simulation3D:

    def __init__(self, 
                 modelname: str, 
                 display: Type[BaseDisplay] = None,
                 loglevel: Literal['DEBUG','INFO','WARNING','ERROR'] = 'INFO',
                 load_file: bool = False,
                 save_file: bool = False,
                 logfile: bool = False,
                 microwave: bool = True):
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
        
        self.mesh: Mesh3D = Mesh3D(self.mesher)
        self.select: Selector = Selector()
        self.display: PVDisplay = None
        self.geo: Builder = Builder()
        self._geometries: list[GeoObject] = []
        self.set_loglevel(loglevel)

        ## STATES
        self.__active: bool = False
        self._defined_geometries: bool = False
        self._cell: PeriodicCell = None

        if display is not None:
            self.display = display(self.mesh)
        else:
            self.display = PVDisplay(self.mesh)
        if logfile:
            self.set_logfile(logfile)

        self.obj: dict[str, GeoObject] = dict()

        self.save_file: bool = save_file
        self.load_file: bool = load_file

        self.data: SimulationDataset = SimulationDataset()

        ## Physics
        self.mw: Microwave3D = Microwave3D(self.mesher, self.data.mw)

        self._initialize_simulation()

        self._update_data()
        
        

    def __setitem__(self, name: str, value: GeoObject) -> None:
        self.obj[name] = value

    def __getitem__(self, name: str) -> GeoObject:
        return self.obj[name]
    
    def _update_data(self) -> None:
        self.mw.data = self.data.mw
        
    def set_mesh(self, mesh: Mesh3D) -> None:
        """Set the current model mesh to a given mesh."""
        self.mesh = mesh
        self.mw.mesh = mesh
        self.mesher.mesh = mesh
        self.display._mesh = mesh
    
    def save(self) -> None:
        """Saves the current model in the provided project directory."""
        # Ensure directory exists
        if not self.modelpath.exists():
            self.modelpath.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created directory: {self.modelpath}")

        # Save mesh
        mesh_path = self.modelpath / 'mesh.msh'
        brep_path = self.modelpath / 'model.brep'

        gmsh.option.setNumber('Mesh.SaveParametric', 1)
        gmsh.option.setNumber('Mesh.SaveAll', 1)
        gmsh.model.geo.synchronize()
        gmsh.model.occ.synchronize()

        gmsh.write(str(mesh_path))
        gmsh.write(str(brep_path))
        logger.info(f"Saved mesh to: {mesh_path}")

        # Pack and save data
        objects = self.obj
        dataset = dict(simdata=self.data, objects=objects, mesh=self.mesh)
        data_path = self.modelpath / 'simdata.emerge'
        joblib.dump(dataset, str(data_path))
        logger.info(f"Saved simulation data to: {data_path}")

    def load(self) -> None:
        """Loads the model from the project directory."""
        mesh_path = self.modelpath / 'mesh.msh'
        brep_path = self.modelpath / 'model.brep'
        data_path = self.modelpath / 'simdata.emerge'

        if not mesh_path.exists() or not data_path.exists():
            raise FileNotFoundError("Missing required mesh or data file.")

        # Load mesh
        gmsh.open(str(brep_path))
        gmsh.merge(str(mesh_path))
        gmsh.model.geo.synchronize()
        gmsh.model.occ.synchronize()
        logger.info(f"Loaded mesh from: {mesh_path}")
        #self.mesh.update([])

        # Load data
        datapack = joblib.load(str(data_path))
        self.data = datapack['simdata']
        self.obj = datapack['objects']
        self.set_mesh(datapack['mesh'])
        logger.info(f"Loaded simulation data from: {data_path}")

    def load_data(self, key: str) -> Any:
        return self.save_data[key]
    
    def set_loglevel(self, loglevel: Literal['DEBUG','INFO','WARNING','ERROR']) -> None:
        """Set the loglevel for loguru.

        Args:
            loglevel ('DEBUG','INFO','WARNING','ERROR'): The loglevel
        """
        handler = {"sink": sys.stdout, "level": loglevel, "format": logger_format}
        logger.configure(handlers=[handler])

    def set_logfile(self) -> None:
        """Adds a file output for the logger."""
        logger.add(str(self.modelpath / 'logging.log'), mode='w', level='DEBUG', format=logger_format, colorize=False, backtrace=True, diagnose=True)
        
    def view(self, 
             selections: list[Selection] = None, 
             use_gmsh: bool = False,
             volume_opacity: float = 0.1,
             surface_opacity: float = 1,
             show_edges: bool = True) -> None:
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
                if obj.dim==2:
                    opacity=surface_opacity
                elif obj.dim==3:
                    opacity=volume_opacity
                self.display.add_object(obj, show_edges=show_edges, opacity=opacity)
            if selections:
                [self.display.add_object(sel, color='red', opacity=0.7) for sel in selections]
            self.display.show()
            return
        except NotImplementedError as e:
            logger.warning('The provided BaseDisplay class does not support object display. Please make' \
            'sure that this method is properly implemented.')
    
    def set_periodic_cell(self, cell: PeriodicCell, excluded_faces: list[FaceSelection] = None):
        """Sets the periodic cell information based on the PeriodicCell class object"""
        self.mw.bc._cell = cell
        self._cell = cell

    def define_geometry(self, *geometries: list[GeoObject]) -> None:
        """Provide the physics engine with the geometries that are contained and ought to be included
        in the simulation. Please make sure to include all geometries. Its currently unclear how the
        system behaves if only a part of all geometries are included.

        """
        geometries = geometries + tuple(self.obj.values()) + tuple(self.geo.all_geometries())
        self._geometries = unpack_lists(geometries)
        self.mesher.submit_objects(self._geometries)
        self._defined_geometries = True
        self.display._facetags = [dt[1] for dt in gmsh.model.get_entities(2)]
    
    def generate_mesh(self):
        """Generate the mesh. 
        This can only be done after define_geometry(...) is called and if frequencies are defined.

        Args:
            name (str, optional): The mesh file name. Defaults to "meshname.msh".

        Raises:
            ValueError: ValueError if no frequencies are defined.
        """
        if not self._defined_geometries:
            self.define_geometry()
        
        # Set the cell periodicity in GMSH
        if self._cell is not None:
            self.mesher.set_periodic_cell(self._cell)
        
        # Check if frequencies are defined: TODO: Replace with a more generic check
        if self.mw.frequencies is None:
            raise ValueError('No frequencies defined for the simulation. Please set frequencies before generating the mesh.')

        gmsh.model.occ.synchronize()

        # Set the mesh size
        self.mesher.set_mesh_size(self.mw.get_discretizer(), self.mw.resolution)
        try:
            gmsh.model.mesh.generate(3)
        except Exception:
            logger.error('GMSH Mesh error detected.')
            print(_GMSH_ERROR_TEXT)
            raise

        self.mw._initialize_bcs()
        
        self.mesh.update(self.mesher._get_periodic_bcs())
        gmsh.model.occ.synchronize()
        self.mw.mesh = self.mesh

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

    def parameter_sweep(self, clear_mesh: bool = True, **parameters: np.ndarray) -> Generator[tuple[float,...], None, None]:
        """Executes a parameteric sweep iteration.

        The first argument clear_mesh determines if the mesh should be cleared and rebuild in between sweeps. This is usually needed
        except when you change only port-properties or material properties. The parameters of the sweep can be provided as a set of 
        keyword arguments. As an example if you defined the axes: width=np.linspace(...) and height=np.linspace(...). You can
        add them as arguments using .parameter_sweep(True, width=width, height=height).

        The rest of the simulation commands should be inside the iteration scope

        Args:
            clear_mesh (bool, optional): Wether to clear the mesh in between sweeps. Defaults to True.

        Yields:
            Generator[tuple[float,...], None, None]: The parameters provided

        Example:
         >>> for W, H in model.parameter_sweep(True, width=widths, height=heights):
         >>>    // build simulation
         >>>    data = model.frequency_domain()
         >>> // Extract the data
         >>> widths, heights, frequencies, S21 = data.ax('width','height','freq').S(2,1)
        """
        paramlist = sorted(list(parameters.keys()))
        dims = np.meshgrid(*[parameters[key] for key in paramlist], indexing='ij')
        dims_flat = [dim.flatten() for dim in dims]
        self.mw.cache_matrices = False
        for iter in range(dims_flat[0].shape[0]):
            if clear_mesh:
                logger.info('Cleaning up mesh.')
                gmsh.clear()
                self.mesh = Mesh3D(self.mesher)
                self.mw.reset()
            self.mw._params = {key: dim[iter] for key,dim in zip(paramlist, dims_flat)}
            self._params = {key: dim[iter] for key,dim in zip(paramlist, dims_flat)}
            logger.info(f'Iterating: {self.mw._params}')
            if len(dims_flat)==1:
                yield dims_flat[0][iter]
            else:
                yield (dim[iter] for dim in dims_flat)
        self.mw.cache_matrices = True

    def __enter__(self) -> Simulation3D:
        """This method is depricated with the new atexit system. It still exists for backwards compatibility.

        Returns:
            Simulation3D: the Simulation3D object
        """
        return self

    def __exit__(self, type, value, tb):
        """This method no longer does something. It only serves as backwards compatibility."""
        self._exit_gmsh()
        return False
    
    def _install_signal_handlers(self):
        # on SIGINT (Ctrl-C) or SIGTERM, call our exit routine
        for sig in (signal.SIGINT, signal.SIGTERM):
            signal.signal(sig, self._handle_signal)
    
    def _handle_signal(self, signum, frame):
        """
        Signal handler: do our cleanup, then re-raise
        the default handler so that exit code / traceback
        is as expected.
        """
        try:
            # run your atexit-style cleanup
            self._exit_gmsh()
        except Exception:
            # log but don’t block shutdown
            logger.exception("Error during signal cleanup")
        finally:
            # restore default handler and re‐send the signal
            signal.signal(signum, signal.SIG_DFL)
            os.kill(os.getpid(), signum)

 
    def _initialize_simulation(self):
        """Initializes the Simulation data and GMSH API with proper shutdown routines.
        """
        # If GMSH is not yet initialized (Two simulation in a file)
        if gmsh.isInitialized() == 0:
            logger.debug('Initializing GMSH')
            gmsh.initialize()
            
            # Set an exit handler for Ctrl+C cases
            self._install_signal_handlers()

            # Restier the Exit GMSH function on proper program abortion
            register(self._exit_gmsh)

        # Create a new GMSH model or load it
        if not self.load_file:
            gmsh.model.add(self.modelname)
            self.data: SimulationDataset = SimulationDataset()
        else:
            self.load()

        # Set the Simulation state to active
        self.__active = True
        return self

    def _exit_gmsh(self):
        # If the simulation object state is still active (GMSH is running)
        if not self.__active:
            return
        logger.debug('Exiting program')
        # Save the file first
        if self.save_file:
            self.save()
        # Finalize GMSH
        gmsh.finalize()
        logger.debug('GMSH Shut down successful')
        # set the state to active
        self.__active = False

   