import numpy as np
from ...material import Material
from ...const import Z0 as n0

PI = np.pi
TAU = 2*PI

def microstrip_z0(W: float, th: float, er: float):
    u = W/th
    fu = 6 + (TAU - 6)*np.exp(-((30.666/u)**(0.7528)))
    Z0 = n0/(TAU*np.sqrt(er))* np.log(fu/u + np.sqrt(1+(2/u)**2))
    return Z0


class PCBCalculator:

    def __init__(self, thickness: float, layers: np.ndarray, material: Material, unit: float):
        self.th = thickness
        self.layers = layers
        self.mat = material
        self.unit = unit

    def z0(self, Z0: float, layer: int = -1, ground_layer: int = 0, f0: float = 1e9):
        th = abs(self.layers[layer] - self.layers[ground_layer])*self.unit
        ws = np.geomspace(1e-6,1e-1,101)
        Z0ms = microstrip_z0(ws, th, self.mat.er.scalar(f0))
        return np.interp(Z0, Z0ms[::-1], ws[::-1])/self.unit
