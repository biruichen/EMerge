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

import numpy as np
from ..microwave_bc import PEC, BoundaryCondition, RectangularWaveguide, RobinBC, PortBC, Periodic
from ....elements.nedelec2 import Nedelec2
from ....elements.nedleg2 import NedelecLegrange2
from ....mth.optimized import gaus_quad_tri
from scipy.sparse import csr_matrix, eye
from loguru import logger
from ..simjob import SimJob
from collections import defaultdict

C0 = 299792458
EPS0 = 8.854187818814e-12

def matprint(mat: np.ndarray) -> None:
    factor = np.max(np.abs(mat.flatten()))
    if factor == 0:
        factor = 1
    print(mat.real/factor)

def select_bc(bcs: list[BoundaryCondition], bctype):
    return [bc for bc in bcs if isinstance(bc,bctype)]

def diagnose_matrix(mat: np.ndarray) -> None:

    if not isinstance(mat, np.ndarray):
        logger.debug('Converting sparse array to flattened array')
        mat = mat[mat.nonzero()].A1
        #mat = np.array(nonzero_mat)
    
    ''' Prints all indices of Nan's and infinities in a matrix '''
    ids = np.where(np.isnan(mat))
    if len(ids[0]) > 0:
        logger.error(f'Found NaN at {ids}')
    ids = np.where(np.isinf(mat))
    if len(ids[0]) > 0:
        logger.error(f'Found Inf at {ids}')
    ids = np.where(np.abs(mat) > 1e10)
    if len(ids[0]) > 0:
        logger.error(f'Found large values at {ids}')
    logger.info('Diagnostics finished')

#############
def gen_key(coord, mult):
    return tuple([int(round(c*mult)) for c in coord])

def plane_basis_from_points(points: np.ndarray) -> np.ndarray:
    """
    Compute an orthonormal basis from a cloud of 3D points dominantly
    lying on one plane.

    Parameters
    ----------
    points : ndarray, shape (3, N)
        3D coordinates of the point cloud.

    Returns
    -------
    basis : ndarray, shape (3, 3)
        Matrix whose columns are:
            - first principal direction (plane X axis)
            - second principal direction (plane Y axis)
            - plane normal vector (Z axis)
    """
    if points.shape[0] != 3:
        raise ValueError("Input must have shape (3, N)")

    # Compute centroid
    centroid = points.mean(axis=1, keepdims=True)

    # Center the data
    points_centered = points - centroid

    # Compute covariance matrix (3x3)
    C = (points_centered @ points_centered.T) / points.shape[1]

    # Eigen decomposition
    eigvals, eigvecs = np.linalg.eigh(C)

    # Sort eigenvectors by descending eigenvalue
    idx = np.argsort(eigvals)[::-1]
    eigvecs = eigvecs[:, idx]

    # Columns of eigvecs = principal axes
    return eigvecs

class Assembler:

    def __init__(self):
        self.cached_matrices = None
        self.conductivity_limit = 1e7

    
    def assemble_bma_matrices(self,
                              field: Nedelec2,
                        er: np.ndarray, 
                        ur: np.ndarray, 
                        sig: np.ndarray,
                        k0: float,
                        port: PortBC,
                        bcs: list[BoundaryCondition]) -> tuple[np.ndarray, np.ndarray, np.ndarray, NedelecLegrange2]:
        
        from .generalized_eigen import generelized_eigenvalue_matrix
        
        mesh = field.mesh
        tri_ids = mesh.get_triangles(port.tags)

        origin = tuple([c-n for c,n in zip(port.cs.origin, port.cs.gzhat)])

        boundary_surface = mesh.boundary_surface(port.tags, origin)
        nedlegfield = NedelecLegrange2(boundary_surface, port.cs)

        ermesh = er[:,:,tri_ids]
        urmesh = ur[:,:,tri_ids]
        sigmesh = sig[tri_ids]

        ermesh = ermesh - 1j * sigmesh/(k0*C0*EPS0)

        E, B = generelized_eigenvalue_matrix(nedlegfield, ermesh, urmesh, port.cs._basis, k0)
        

       

        pecs: list[PEC] = [bc for bc in bcs if isinstance(bc,PEC)]
        if len(pecs) > 0:
            logger.debug('Implementing PEC BCs')

        pec_ids = []

        # Process all concutors. Everything above the conductivity limit is considered pec.
        for it in range(boundary_surface.n_tris):
            if sigmesh[it] > self.conductivity_limit:
                pec_ids.extend(list(nedlegfield.tri_to_field[:,it]))

        # Process all PEC Boundary Conditions
        pec_edges = []
        pec_vertices = []
        for pec in pecs:
            face_tags = pec.tags

            tri_ids = mesh.get_triangles(face_tags)
            edge_ids = list(mesh.tri_to_edge[:,tri_ids].flatten())
            
            for ii in edge_ids:
                i2 = nedlegfield.mesh.from_source_edge(ii)
                if i2 is None:
                    continue
                eids = nedlegfield.edge_to_field[:, i2]
                pec_ids.extend(list(eids))
                pec_edges.append(eids[0])
                pec_vertices.append(eids[3]-nedlegfield.n_xy)
                pec_vertices.append(eids[4]-nedlegfield.n_xy)
                

            for ii in tri_ids:
                i2 = nedlegfield.mesh.from_source_tri(ii)
                if i2 is None:
                    continue
                tids = nedlegfield.tri_to_field[:, i2]
                pec_ids.extend(list(tids))

        port._field = nedlegfield
        port._pece = pec_edges
        port._pecv = pec_vertices
        # Process all port boundary Conditions
        pec_ids = set(pec_ids)
        solve_ids = [i for i in range(nedlegfield.n_field) if i not in pec_ids]

        #matplot(E, solve_ids)
        #matplot(B, solve_ids)
        return E, B, np.array(solve_ids), nedlegfield

    def assemble_freq_matrix(self, field: Nedelec2, 
                        er: np.ndarray, 
                        ur: np.ndarray, 
                        sig: np.ndarray,
                        bcs: list[BoundaryCondition],
                        frequency: float,
                        cache_matrices: bool = False) -> SimJob:
        
        from .curlcurl import tet_mass_stiffness_matrices
        from .robinbc import assemble_robin_bc, assemble_robin_bc_excited

        mesh = field.mesh
        w0 = 2*np.pi*frequency
        k0 = w0/C0

        er = er - 1j*sig/(w0*EPS0)*np.repeat(np.eye(3)[:, :, np.newaxis], er.shape[2], axis=2)
        
        f_dependent_properties = np.any((sig > 0) & (sig < self.conductivity_limit))
        
        if cache_matrices and not f_dependent_properties and self.cached_matrices is not None:
            logger.debug('    Retreiving cached matricies')
            E, B = self.cached_matrices
        else:
            logger.debug('    Assembling matrices')
            E, B = tet_mass_stiffness_matrices(field, er, ur)
            self.cached_matrices = (E, B)

        K: csr_matrix = (E - B*(k0**2)).tocsr()

        NF = E.shape[0]

        pecs: list[PEC] = [bc for bc in bcs if isinstance(bc,PEC)]
        robin_bcs: list[RectangularWaveguide] = [bc for bc in bcs if isinstance(bc,RobinBC)]
        ports: list[PortBC] = [bc for bc in bcs if isinstance(bc, PortBC)]
        periodic: list[Periodic] = [bc for bc in bcs if isinstance(bc, Periodic)]

        b = np.zeros((E.shape[0],)).astype(np.complex128)
        port_vectors = {port.port_number: np.zeros((E.shape[0],)).astype(np.complex128) for port in ports}
        # Process all PEC Boundary Conditions
        pec_ids = []
        
        logger.debug('    Implementing PEC Boundary Conditions.')
        
        # Conductivity above al imit, consider it all PEC
        for itet in range(field.n_tets):
            if sig[itet] > self.conductivity_limit:
                pec_ids.extend(field.tet_to_field[:,itet])
        
        # PEC Boundary conditions
        for pec in pecs:
            if len(pec.tags)==0:
                continue
            face_tags = pec.tags
            tri_ids = mesh.get_triangles(face_tags)
            edge_ids = list(mesh.tri_to_edge[:,tri_ids].flatten())

            for ii in edge_ids:
                eids = field.edge_to_field[:, ii]
                pec_ids.extend(list(eids))

            for ii in tri_ids:
                tids = field.tri_to_field[:, ii]
                pec_ids.extend(list(tids))

        # Robin BCs
        if len(robin_bcs) > 0:
            logger.debug('    Implementing Robin Boundary Conditions.')
        
        gauss_points = gaus_quad_tri(4)

        for bc in robin_bcs:
            for tag in bc.tags:
                face_tags = [tag,]

                tri_ids = mesh.get_triangles(face_tags)
                nodes = mesh.get_nodes(face_tags)
                edge_ids = list(mesh.tri_to_edge[:,tri_ids].flatten())

                gamma = bc.get_gamma(k0)

                def Ufunc(x,y): 
                    return bc.get_Uinc(x,y,k0)
                
                ibasis = bc.get_inv_basis()
                if ibasis is None:
                    basis = plane_basis_from_points(mesh.nodes[:,nodes]) + 1e-16
                    ibasis = np.linalg.pinv(basis)
                if bc._include_force:

                    B_p, b_p = assemble_robin_bc_excited(field, tri_ids, Ufunc, gamma, ibasis, bc.cs.origin, gauss_points)
                    
                    port_vectors[bc.port_number] += b_p
                
                else:
                    B_p = assemble_robin_bc(field, tri_ids, gamma)
                if bc._include_stiff:
                    K = K + B_p
        
        if len(periodic) > 0:
            logger.debug('    Implementing Periodic Boundary Conditions.')

        # Periodic BCs
        Pmats = []
        remove = set()
        has_periodic = False

        for bc in periodic:
            Pmat = eye(NF, format='lil', dtype=np.complex128)
            has_periodic = True
            tri_ids_1 = mesh.get_triangles(bc.face1.tags)
            tri_ids_2 = mesh.get_triangles(bc.face2.tags)
            dv = np.array(bc.dv)
            trimapdict = defaultdict(lambda: [None, None])
            edgemapdict = defaultdict(lambda: [None, None])
            mult = int(10**(-np.round(np.log10(min(mesh.edge_lengths.flatten())))+3))
            
            edge_ids_1 = set()
            edge_ids_2 = set()
            
            phi = bc.phi(k0)
            for i1, i2 in zip(tri_ids_1, tri_ids_2):
                trimapdict[gen_key(mesh.tri_centers[:,i1], mult)][0] = i1
                trimapdict[gen_key(mesh.tri_centers[:,i2]-dv, mult)][1] = i2
                
                ie1, ie2, ie3 = mesh.tri_to_edge[:,i1]
                edge_ids_1.update({ie1, ie2, ie3})
                ie1, ie2, ie3 = mesh.tri_to_edge[:,i2]
                edge_ids_2.update({ie1, ie2, ie3})
            
            for i1, i2 in zip(list(edge_ids_1), list(edge_ids_2)):
                edgemapdict[gen_key(mesh.edge_centers[:,i1], mult)][0] = i1
                edgemapdict[gen_key(mesh.edge_centers[:,i2]-dv, mult)][1] = i2

            
            for t1, t2 in trimapdict.values():
                f11, f21 = field.tri_to_field[[3,7],t1]
                f12, f22 = field.tri_to_field[[3,7],t2]
                Pmat[f12,f12] = 0
                Pmat[f22,f22] = 0
                if Pmat[f12,f11] == 0:
                    Pmat[f12,f11] += phi
                else:
                    Pmat[f12,f11] *= phi
                
                if Pmat[f22,f21] == 0:
                    Pmat[f22,f21] += phi
                else:
                    Pmat[f22,f21] *= phi
                remove.add(f12)
                remove.add(f22)
                
            for e1, e2 in edgemapdict.values():
                f11, f21 = field.edge_to_field[:,e1]
                f12, f22 = field.edge_to_field[:,e2]
                Pmat[f12,f12] = 0
                Pmat[f22,f22] = 0
                if Pmat[f12,f11] == 0:
                    Pmat[f12,f11] += phi
                else:
                    Pmat[f12,f11] *= phi
                
                if Pmat[f22,f21] == 0:
                    Pmat[f22,f21] += phi
                else:
                    Pmat[f22,f21] *= phi
                remove.add(f12)
                remove.add(f22)
                
            Pmats.append(Pmat)
        
        if Pmats:
            Pmat = Pmats[0]
            for P2 in Pmats[1:]:
                Pmat = Pmat @ P2
            Pmat = Pmat.tocsr()
            remove = np.array(sorted(list(remove)))
            all_indices = np.arange(NF)
            keep_indices = np.setdiff1d(all_indices, remove)
            Pmat = Pmat[:,keep_indices]
        else:
            Pmat = None
        
        pec_ids = set(pec_ids)
        solve_ids = np.array([i for i in range(E.shape[0]) if i not in pec_ids])
        
        if has_periodic:
            mask = np.zeros((NF,))
            mask[solve_ids] = 1
            mask = mask[keep_indices]
            solve_ids = np.argwhere(mask==1).flatten()

        logger.debug(f'Number of tets: {mesh.n_tets}')
        logger.debug(f'Number of DoF: {K.shape[0]}')
        simjob = SimJob(K, b, k0*299792458/(2*np.pi), True)
        
        simjob.port_vectors = port_vectors
        simjob.solve_ids = solve_ids

        if has_periodic:
            simjob.P = Pmat
            simjob.Pd = Pmat.getH()
            simjob.has_periodic = has_periodic

        return simjob
    
    def assemble_eig_matrix(self, field: Nedelec2, 
                        er: np.ndarray, 
                        ur: np.ndarray, 
                        sig: np.ndarray,
                        bcs: list[BoundaryCondition],
                        frequency: float) -> SimJob:
        
        from .curlcurl import tet_mass_stiffness_matrices
        from .robinbc import assemble_robin_bc

        mesh = field.mesh
        w0 = 2*np.pi*frequency
        k0 = w0/C0

        er = er - 1j*sig/(w0*EPS0)*np.repeat(np.eye(3)[:, :, np.newaxis], er.shape[2], axis=2)
        
        logger.debug('    Assembling matrices')
        E, B = tet_mass_stiffness_matrices(field, er, ur)
        self.cached_matrices = (E, B)

        NF = E.shape[0]

        pecs: list[PEC] = [bc for bc in bcs if isinstance(bc,PEC)]
        robin_bcs: list[RectangularWaveguide] = [bc for bc in bcs if isinstance(bc,RobinBC)]
        periodic: list[Periodic] = [bc for bc in bcs if isinstance(bc, Periodic)]

        # Process all PEC Boundary Conditions
        pec_ids = []
        
        logger.debug('    Implementing PEC Boundary Conditions.')
        
        # Conductivity above a limit, consider it all PEC
        for itet in range(field.n_tets):
            if sig[itet] > self.conductivity_limit:
                pec_ids.extend(field.tet_to_field[:,itet])
        
        # PEC Boundary conditions
        for pec in pecs:
            if len(pec.tags)==0:
                continue
            face_tags = pec.tags
            tri_ids = mesh.get_triangles(face_tags)
            edge_ids = list(mesh.tri_to_edge[:,tri_ids].flatten())

            for ii in edge_ids:
                eids = field.edge_to_field[:, ii]
                pec_ids.extend(list(eids))

            for ii in tri_ids:
                tids = field.tri_to_field[:, ii]
                pec_ids.extend(list(tids))

        # Robin BCs
        if len(robin_bcs) > 0:
            logger.debug('    Implementing Robin Boundary Conditions.')
        
        for bc in robin_bcs:
            for tag in bc.tags:
                face_tags = [tag,]#bc.tags

                tri_ids = mesh.get_triangles(face_tags)
                nodes = mesh.get_nodes(face_tags)
                edge_ids = list(mesh.tri_to_edge[:,tri_ids].flatten())

                gamma = bc.get_gamma(k0)
                
                ibasis = bc.get_inv_basis()
                if ibasis is None:
                    basis = plane_basis_from_points(mesh.nodes[:,nodes]) + 1e-16
                    ibasis = np.linalg.pinv(basis)
                B_p = assemble_robin_bc(field, tri_ids, gamma)
                if bc._include_stiff:
                    B = B + B_p
        
        if len(periodic) > 0:
            logger.debug('    Implementing Periodic Boundary Conditions.')

        # Periodic BCs
        Pmats = []
        remove = set()
        has_periodic = False

        for bc in periodic:
            Pmat = eye(NF, format='lil', dtype=np.complex128)
            has_periodic = True
            tri_ids_1 = mesh.get_triangles(bc.face1.tags)
            tri_ids_2 = mesh.get_triangles(bc.face2.tags)
            dv = np.array(bc.dv)
            trimapdict = defaultdict(lambda: [None, None])
            edgemapdict = defaultdict(lambda: [None, None])
            mult = int(10**(-np.round(np.ceil(min(mesh.edge_lengths.flatten())))+2))
            edge_ids_1 = set()
            edge_ids_2 = set()
            
            phi = bc.phi(k0)
            for i1, i2 in zip(tri_ids_1, tri_ids_2):
                trimapdict[gen_key(mesh.tri_centers[:,i1], mult)][0] = i1
                trimapdict[gen_key(mesh.tri_centers[:,i2]-dv, mult)][1] = i2
                
                ie1, ie2, ie3 = mesh.tri_to_edge[:,i1]
                edge_ids_1.update({ie1, ie2, ie3})
                ie1, ie2, ie3 = mesh.tri_to_edge[:,i2]
                edge_ids_2.update({ie1, ie2, ie3})
            
            for i1, i2 in zip(list(edge_ids_1), list(edge_ids_2)):
                edgemapdict[gen_key(mesh.edge_centers[:,i1], mult)][0] = i1
                edgemapdict[gen_key(mesh.edge_centers[:,i2]-dv, mult)][1] = i2

            
            for t1, t2 in trimapdict.values():
                f11, f21 = field.tri_to_field[[3,7],t1]
                f12, f22 = field.tri_to_field[[3,7],t2]
                Pmat[f12,f12] = 0
                Pmat[f22,f22] = 0
                if Pmat[f12,f11] == 0:
                    Pmat[f12,f11] += phi
                else:
                    Pmat[f12,f11] *= phi
                
                if Pmat[f22,f21] == 0:
                    Pmat[f22,f21] += phi
                else:
                    Pmat[f22,f21] *= phi
                remove.add(f12)
                remove.add(f22)
                
            for e1, e2 in edgemapdict.values():
                f11, f21 = field.edge_to_field[:,e1]
                f12, f22 = field.edge_to_field[:,e2]
                Pmat[f12,f12] = 0
                Pmat[f22,f22] = 0
                if Pmat[f12,f11] == 0:
                    Pmat[f12,f11] += phi
                else:
                    Pmat[f12,f11] *= phi
                
                if Pmat[f22,f21] == 0:
                    Pmat[f22,f21] += phi
                else:
                    Pmat[f22,f21] *= phi
                remove.add(f12)
                remove.add(f22)
                
            Pmats.append(Pmat)
        
        if Pmats:
            Pmat = Pmats[0]
            for P2 in Pmats[1:]:
                Pmat = Pmat @ P2
            Pmat = Pmat.tocsr()
            remove = np.array(sorted(list(remove)))
            all_indices = np.arange(NF)
            keep_indices = np.setdiff1d(all_indices, remove)
            Pmat = Pmat[:,keep_indices]
        else:
            Pmat = None
        
        pec_ids = set(pec_ids)
        solve_ids = np.array([i for i in range(E.shape[0]) if i not in pec_ids])
        
        if has_periodic:
            mask = np.zeros((NF,))
            mask[solve_ids] = 1
            mask = mask[keep_indices]
            solve_ids = np.argwhere(mask==1).flatten()

        logger.debug(f'Number of tets: {mesh.n_tets}')
        logger.debug(f'Number of DoF: {E.shape[0]}')
        simjob = SimJob(E, None, frequency, False, B=B)
        
        simjob.solve_ids = solve_ids

        if has_periodic:
            simjob.P = Pmat
            simjob.Pd = Pmat.getH()
            simjob.has_periodic = has_periodic

        return simjob