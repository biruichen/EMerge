from .baselib import CompiledLib
from loguru import logger


MATHLIB: CompiledLib = CompiledLib

try:
    import emerge_iron
    from .iron import IRONLib
    MATHLIB = IRONLib
    logger.debug(f'Using EMerge-IRON as interpolation backend.')
except ImportError:
    logger.debug(f'Using Numba(Default) as interpolation backend.')