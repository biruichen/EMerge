import numpy as np
from emerge._emerge.physics.microwave.microwave_data import EHField
from emsutil.const import EPS0

def Qv(field: EHField) -> float:
    """ Drop in callable for volumetric loss integrations.
    
    Example:
    ---------
    >>> loss = data.field.find(freq=...).intvol(selection, Qv).real
    """
    return -1/2*2*np.pi*field.freq*EPS0*field.normE**2 * field.er.imag