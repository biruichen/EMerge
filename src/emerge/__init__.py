"""A Python based FEM solver.
Copyright (C) 2025  name of Robert Fennis

This program is free software; you can redistribute it and/or
modify it under the terms of the GNU General Public License
as published by the Free Software Foundation; either version 2
of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program; if not, see
<https://www.gnu.org/licenses/>.

"""
print("""        ___
      _/   |
   __/    _|     _____ __  __
 _/_|   _/  \   |  ___|  \/  |
/ / |  /     \  | |_  | \  / | ___ _ __ __ _  ___
| | | ||      | |  _| | |\/| |/ _ \ `__/ _` |/ _ \\
| | |  \\      | | |___| |  | |  __/ | | (_| |  __/
\_|  \ | |  _/  |_____|_|  |_|\___|_|  \__, |\___|
     | | |_/   _______________________  __/ | ____
     | |                               |___/
      \|""")

from loguru import logger
from .logsettings import logger_format
import sys

logger.remove()
logger.add(sys.stderr, format=logger_format)

from .simmodel import Simulation3D
from .material import Material, FR4, AIR, VACUUM, COPPER
from . import physics
from . import bc
from .solver import superlu_info, SolverBicgstab, SolverGMRES, SolveRoutine, ReverseCuthillMckee, Sorter, SolverPardiso, SolverUMFPACK
from .cs import CoordinateSystem, Plane, Axis, XAX, YAX, ZAX, XYPLANE, XZPLANE, YZPLANE, YXPLANE, ZXPLANE, ZYPLANE
from .coord import Line
from . import plot
from . import geo
from .selection import Selection, FaceSelection, DomainSelection, EdgeSelection
from .mth.common_functions import norm
