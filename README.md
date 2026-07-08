## Introduction

Hello everybody. Thanks for showing interest in this repository.

Feel free to download your version of EMerge and start playing around with it!
If you have suggestions/changes/questions either use the Github issue system or join the Discord using the following link:

**https://discord.gg/7PF4WcS6uA**

## How to install

Clone this repository or download the files. While in the EMerge path containing the src/emerge folder, install the module using:
```
pip install .
```
If you want to install the library with PyPardiso on Intel machines, you can install the optional dependency with EMerge using:
```
pip install ".[pypardiso]"
```

## Compatibility

As far as I know, the library should work on all systems. PyPARDISO is not supported on ARM but the current SuperLU and UMFPACK solvers work on ARM as well. Both SuperLU and UMFPACK can run on multi-processing implementations as long as you do entry-point protection:
```
import emerge as em

def main():
    # setup simulation

    model.mw.frequency_domain(True, ..., multi_processing=True)

if __name__ == "__main__":
    main()
```
Otherwise, the parallel solver will default to SuperLU which is significantly slower on larger problems.

## Required libraries

To run this FEM library you need the following libraries

 - numpy
 - scipy
 - pypardiso
 - gmsh
 - loguru
 - numba
 - matplotlib (for the matplotlib base display)
 - pyvista (for the PyVista base display)
 - numba-progress
 - scikit-umfpack

## NOTICE

First time runs will be very slow because Numba needs to generate local C-compiled functions of the assembler and other mathematical functions. These compilations are chached so this should only take time once.
