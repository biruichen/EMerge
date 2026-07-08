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
import os
from pathlib import Path
import shutil

def generate_project(projectname: str, filename: str):
    """
    Generates a new project structure with boilerplate simulation and post-processing files.

    Args:
        projectname (str): Relative path to the new project directory.
        filename (str): Base name for the simulation and post-processing files.
    """
    # Convert to Path objects
    project_path = Path(projectname)
    base_dir = Path(__file__).parent  # Points to src/emerge/projects/
    
    # Ensure the project directory exists
    project_path.mkdir(parents=True, exist_ok=True)
    
    # Define file paths
    sim_file = project_path / f"{filename}_sim.py"
    post_file = project_path / f"{filename}_post.py"
    data_dir = project_path / f"{filename}_data"

    # Paths to the template files
    gen_base_path = base_dir / "_gen_base.txt"
    load_base_path = base_dir / "_load_base.txt"

    # Read and write _sim.py
    with open(gen_base_path, 'r') as src, open(sim_file, 'w') as dst:
        dst.write(src.read().replace('#FILE#', f'"{filename}"'))

    # Read and write _post.py
    with open(load_base_path, 'r') as src, open(post_file, 'w') as dst:
        dst.write(src.read().replace('#FILE#', f'"{filename}"'))

    # Create the data directory
    data_dir.mkdir(exist_ok=True)
    
    # ALLOWED PRINT
    print(f"Project '{projectname}' created with files: {sim_file.name}, {post_file.name} and directory: {data_dir.name}")
