# from ...selection import FaceSelection, SELECTOR_OBJ
# import numpy as np
# from typing import Generator
# from .microwave_bc import Periodic, FloquetPort
# from ...geo.extrude import XYPolygon, GeoPrism
# from ...geo import XYPlate, Alignment
# from ...periodic import PeriodicCell

# class MWPeriodicCell(PeriodicCell):

#     def __post_init__(self):
#         self._bcs: list[Periodic] = []
#         self._ports: list[FloquetPort] = []

#     def volume(self, z1: float, z2: float) -> GeoPrism:
#         """Genereates a volume with the cell geometry ranging from z1 tot z2

#         Args:
#             z1 (float): The start height
#             z2 (float): The end height

#         Returns:
#             GeoPrism: The resultant prism
#         """
#         raise NotImplementedError('This method is not implemented for this subclass.')
    
#     def floquet_port(self, z: float) -> FloquetPort:
#         raise NotImplementedError('This method is not implemented for this subclass.')
    
#     def cell_data(self) -> Generator[tuple[FaceSelection,FaceSelection,tuple[float, float, float]], None, None]:
#         """An iterator that yields the two faces of the hex cell plus a cell periodicity vector

#         Yields:
#             Generator[np.ndarray, np.ndarray, np.ndarray]: The face and periodicity data
#         """
#         raise NotImplementedError('This method is not implemented for this subclass.')

#     @property
#     def bcs(self) -> list[Periodic]:
#         """Returns a list of Periodic boundary conditions for the given PeriodicCell

#         Args:
#             exclude_faces (list[FaceSelection], optional): A possible list of faces to exclude from the bcs. Defaults to None.

#         Returns:
#             list[Periodic]: The list of Periodic boundary conditions
#         """
#         if not self._bcs:
#             bcs = []
#             for f1, f2, a in self.cell_data():
#                 if self.excluded_faces is not None:
#                     f1 = f1 - self.excluded_faces
#                     f2 = f2 - self.excluded_faces
#                 bcs.append(Periodic(f1, f2, a))
#             self._bcs = bcs
#         return self._bcs
    
#     def set_scanangle(self, theta: float, phi: float, degree: bool = True) -> None:
#         """Sets the scanangle for the periodic condition. (0,0) is defined along the Z-axis

#         Args:
#             theta (float): The theta angle
#             phi (float): The phi angle
#             degree (bool): If the angle is in degrees. Defaults to True
#         """
#         if degree:
#             theta = theta*np.pi/180
#             phi = phi*np.pi/180

        
#         ux = np.sin(theta)*np.cos(phi)
#         uy = np.sin(theta)*np.sin(phi)
#         uz = np.cos(theta)
#         for bc in self._bcs:
#             bc.ux = ux
#             bc.uy = uy
#             bc.uz = uz
#         for port in self._ports:
#             port.scan_theta = theta
#             port.scan_phi = phi

        