
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


# Last Cleanup: 2025-01-01
from typing import Any
from ..mesh3d import Mesh3D
from ..geometry import GeoObject
from ..selection import Selection
from ..physics.microwave.microwave_3d import Microwave3D
from ..settings import Settings

class DataSet:
    def __init__(self):
        pass

class SimState:
    
    def __init__(self, settings: Settings, params: dict[str, float]):
        self.params: dict[str, float] = params
        self.set: Settings = settings
        self.mesh: Mesh3D = Mesh3D()
        self.selections: dict[str, Selection] = dict()
        self.geos: dict[str, GeoObject] = dict()
        self.tasks: list[Any] = []
        self.data: DataSet = DataSet()
        self.mw: Microwave3D = Microwave3D()
        
class SimData:
    
    def __init__(self):
        self.states: list[SimState] = []