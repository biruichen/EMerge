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
from numba import njit, f8, c16, i8, types
from ....elements import Nedelec2
from typing import Callable


_FACTORIALS = np.array([1, 1, 2, 6, 24, 120, 720, 5040, 40320, 362880], dtype=np.int64)
 
def optim_matmul(B: np.ndarray, data: np.ndarray):
    dnew = np.zeros_like(data)
    dnew[0,:] = B[0,0]*data[0,:] + B[0,1]*data[1,:] + B[0,2]*data[2,:]
    dnew[1,:] = B[1,0]*data[0,:] + B[1,1]*data[1,:] + B[1,2]*data[2,:]
    dnew[2,:] = B[2,0]*data[0,:] + B[2,1]*data[1,:] + B[2,2]*data[2,:]
    return dnew

@njit(f8(i8, i8, i8, i8), cache=True, fastmath=True, nogil=True)
def area_coeff(a, b, c, d):
    klmn = np.array([0,0,0,0,0,0,0])
    klmn[a] += 1
    klmn[b] += 1
    klmn[c] += 1
    klmn[d] += 1
    output = 2*(_FACTORIALS[klmn[1]]*_FACTORIALS[klmn[2]]*_FACTORIALS[klmn[3]]
                  *_FACTORIALS[klmn[4]]*_FACTORIALS[klmn[5]]*_FACTORIALS[klmn[6]])/_FACTORIALS[(np.sum(klmn[1:])+2)]
    return output
    

NFILL = 5
AREA_COEFF_CACHE_BASE = np.zeros((NFILL,NFILL,NFILL,NFILL), dtype=np.float64)
for I in range(NFILL):
    for J in range(NFILL):
        for K in range(NFILL):
            for L in range(NFILL):
                AREA_COEFF_CACHE_BASE[I,J,K,L] = area_coeff(I,J,K,L)



@njit(f8(f8[:], f8[:]), cache=True, fastmath=True, nogil=True)
def dot(a: np.ndarray, b: np.ndarray):
    return a[0]*b[0] + a[1]*b[1]

@njit(f8[:](f8[:], f8[:]), cache=True, fastmath=True, nogil=True)
def cross(a: np.ndarray, b: np.ndarray):
    crossv = np.empty((3,), dtype=np.float64)
    crossv[0] = a[1]*b[2] - a[2]*b[1]
    crossv[1] = a[2]*b[0] - a[0]*b[2]
    crossv[2] = a[0]*b[1] - a[1]*b[0]
    return crossv

@njit(types.Tuple((f8[:],f8[:]))(f8[:,:], i8[:,:], f8[:,:], i8[:]), cache=True, nogil=True)
def generate_points(vertices_local, tris, DPTs, surf_triangle_indices):
    NS = surf_triangle_indices.shape[0]
    xall = np.zeros((DPTs.shape[1], NS))
    yall = np.zeros((DPTs.shape[1], NS)) 

    for i in range(NS):
        itri = surf_triangle_indices[i]
        vertex_ids = tris[:, itri]
        
        x1, x2, x3 = vertices_local[0, vertex_ids]
        y1, y2, y3 = vertices_local[1, vertex_ids]
        
        xall[:,i] = x1*DPTs[1,:] + x2*DPTs[2,:] + x3*DPTs[3,:]
        yall[:,i] = y1*DPTs[1,:] + y2*DPTs[2,:] + y3*DPTs[3,:]
    
    xflat = xall.flatten()
    yflat = yall.flatten()
    return xflat, yflat

@njit(f8[:,:](f8[:], f8[:], f8[:]), cache=True, nogil=True, fastmath=True)
def compute_distances(xs: np.ndarray, ys: np.ndarray, zs: np.ndarray) -> np.ndarray:
    N = xs.shape[0]
    Ds = np.empty((N,N), dtype=np.float64)
    for i in range(N):
        for j in range(i,N):
            Ds[i,j] = np.sqrt((xs[i]-xs[j])**2 + (ys[i]-ys[j])**2 + (zs[i]-zs[j])**2) 
            Ds[j,i] = Ds[i,j]  
    return Ds

@njit(types.Tuple((c16[:,:],c16[:]))(f8[:,:], c16, c16[:,:], f8[:,:]), cache=True, nogil=True)
def ned2_tri_stiff_force(lcs_vertices, gamma, lcs_Uinc, DPTs):
    ''' Nedelec-2 Triangle Stiffness matrix and forcing vector (For Boundary Condition of the Third Kind)

    '''
    local_edge_map = np.array([[0,1,0],[1,2,2]])
    Bmat = np.zeros((8,8), dtype=np.complex128)
    bvec = np.zeros((8,), dtype=np.complex128)

    xs = lcs_vertices[0,:]
    ys = lcs_vertices[1,:]

    x1, x2, x3 = xs
    y1, y2, y3 = ys

    a1 = x2*y3-y2*x3
    a2 = x3*y1-y3*x1
    a3 = x1*y2-y1*x2
    b1 = y2-y3
    b2 = y3-y1
    b3 = y1-y2
    c1 = x3-x2
    c2 = x1-x3
    c3 = x2-x1

    As = np.array([a1, a2, a3])
    Bs = np.array([b1, b2, b3])
    Cs = np.array([c1, c2, c3])

    Ds = compute_distances(xs, ys, np.zeros_like(xs))

    GL1 = np.array([b1, c1])
    GL2 = np.array([b2, c2])
    GL3 = np.array([b3, c3])

    GLs = (GL1, GL2, GL3)

    Area = 0.5 * np.abs((x1 - x3) * (y2 - y1) - (x1 - x2) * (y3 - y1))
    signA = -np.sign((x1 - x3) * (y2 - y1) - (x1 - x2) * (y3 - y1))

    letters = [1,2,3,4,5,6]

    tA, tB, tC = letters[0], letters[1], letters[2]
    GtA, GtB, GtC = GLs[0], GLs[1], GLs[2]
    
    Lt1, Lt2 = Ds[2, 0], Ds[1, 0]
    
    Ux = lcs_Uinc[0,:]
    Uy = lcs_Uinc[1,:]

    x = x1*DPTs[1,:] + x2*DPTs[2,:] + x3*DPTs[3,:]
    y = y1*DPTs[1,:] + y2*DPTs[2,:] + y3*DPTs[3,:]

    Ws = DPTs[0,:]

    COEFF = gamma/(2*Area)**2
    AREA_COEFF = AREA_COEFF_CACHE_BASE * Area
    for ei in range(3):
        ei1, ei2 = local_edge_map[:, ei]
        Li = Ds[ei1, ei2]
        
        A = letters[ei1]
        B = letters[ei2]

        GA = GLs[ei1]
        GB = GLs[ei2]

        for ej in range(3):
            ej1, ej2 = local_edge_map[:, ej]
            Lj = Ds[ej1, ej2]

            C = letters[ej1]
            D = letters[ej2]

            GC = GLs[ej1]
            GD = GLs[ej2]

            Bmat[ei,ej] += Li*Lj*(AREA_COEFF[A,B,C,D]*dot(GA,GC)-AREA_COEFF[A,B,C,C]*dot(GA,GD)-AREA_COEFF[A,A,C,D]*dot(GB,GC)+AREA_COEFF[A,A,C,C]*dot(GB,GD))
            Bmat[ei,ej+4] += Li*Lj*(AREA_COEFF[A,B,D,D]*dot(GA,GC)-AREA_COEFF[A,B,C,D]*dot(GA,GD)-AREA_COEFF[A,A,D,D]*dot(GB,GC)+AREA_COEFF[A,A,C,D]*dot(GB,GD))
            Bmat[ei+4,ej] += Li*Lj*(AREA_COEFF[B,B,C,D]*dot(GA,GC)-AREA_COEFF[B,B,C,C]*dot(GA,GD)-AREA_COEFF[A,B,C,D]*dot(GB,GC)+AREA_COEFF[A,B,C,C]*dot(GB,GD))
            Bmat[ei+4,ej+4] += Li*Lj*(AREA_COEFF[B,B,D,D]*dot(GA,GC)-AREA_COEFF[B,B,C,D]*dot(GA,GD)-AREA_COEFF[A,B,D,D]*dot(GB,GC)+AREA_COEFF[A,B,C,D]*dot(GB,GD))
            
        Bmat[ei,3] += Li*Lt1*(AREA_COEFF[A,B,tA,tB]*dot(GA,GtC)-AREA_COEFF[A,B,tB,tC]*dot(GA,GtA)-AREA_COEFF[A,A,tA,tB]*dot(GB,GtC)+AREA_COEFF[A,A,tB,tC]*dot(GB,GtA))
        Bmat[ei,7] += Li*Lt2*(AREA_COEFF[A,B,tB,tC]*dot(GA,GtA)-AREA_COEFF[A,B,tC,tA]*dot(GA,GtB)-AREA_COEFF[A,A,tB,tC]*dot(GB,GtA)+AREA_COEFF[A,A,tC,tA]*dot(GB,GtB))
        Bmat[3,ei] += Lt1*Li*(AREA_COEFF[tA,tB,A,B]*dot(GA,GtC)-AREA_COEFF[tA,tB,A,A]*dot(GB,GtC)-AREA_COEFF[tB,tC,A,B]*dot(GA,GtA)+AREA_COEFF[tB,tC,A,A]*dot(GB,GtA))
        Bmat[7,ei] += Lt2*Li*(AREA_COEFF[tB,tC,A,B]*dot(GA,GtA)-AREA_COEFF[tB,tC,A,A]*dot(GB,GtA)-AREA_COEFF[tC,tA,A,B]*dot(GA,GtB)+AREA_COEFF[tC,tA,A,A]*dot(GB,GtB))
        Bmat[ei+4,3] += Li*Lt1*(AREA_COEFF[B,B,tA,tB]*dot(GA,GtC)-AREA_COEFF[B,B,tB,tC]*dot(GA,GtA)-AREA_COEFF[A,B,tA,tB]*dot(GB,GtC)+AREA_COEFF[A,B,tB,tC]*dot(GB,GtA))
        Bmat[ei+4,7] += Li*Lt2*(AREA_COEFF[B,B,tB,tC]*dot(GA,GtA)-AREA_COEFF[B,B,tC,tA]*dot(GA,GtB)-AREA_COEFF[A,B,tB,tC]*dot(GB,GtA)+AREA_COEFF[A,B,tC,tA]*dot(GB,GtB))
        Bmat[3,ei+4] += Lt1*Li*(AREA_COEFF[tA,tB,B,B]*dot(GA,GtC)-AREA_COEFF[tA,tB,A,B]*dot(GB,GtC)-AREA_COEFF[tB,tC,B,B]*dot(GA,GtA)+AREA_COEFF[tB,tC,A,B]*dot(GB,GtA))
        Bmat[7,ei+4] += Lt2*Li*(AREA_COEFF[tB,tC,B,B]*dot(GA,GtA)-AREA_COEFF[tB,tC,A,B]*dot(GB,GtA)-AREA_COEFF[tC,tA,B,B]*dot(GA,GtB)+AREA_COEFF[tC,tA,A,B]*dot(GB,GtB))
            
        A1, A2 = As[ei1], As[ei2]
        B1, B2 = Bs[ei1], Bs[ei2]
        C1, C2 = Cs[ei1], Cs[ei2]

        Ee1x = Li*(B1*(A2 + B2*x + C2*y)/(4*Area**2) - B2*(A1 + B1*x + C1*y)/(4*Area**2))*(A1 + B1*x + C1*y)/(2*Area)
        Ee1y = Li*(C1*(A2 + B2*x + C2*y)/(4*Area**2) - C2*(A1 + B1*x + C1*y)/(4*Area**2))*(A1 + B1*x + C1*y)/(2*Area)
        Ee2x = Li*(B1*(A2 + B2*x + C2*y)/(4*Area**2) - B2*(A1 + B1*x + C1*y)/(4*Area**2))*(A2 + B2*x + C2*y)/(2*Area)
        Ee2y = Li*(C1*(A2 + B2*x + C2*y)/(4*Area**2) - C2*(A1 + B1*x + C1*y)/(4*Area**2))*(A2 + B2*x + C2*y)/(2*Area)

        bvec[ei] += signA*Area*np.sum(Ws*(Ee1x*Ux + Ee1y*Uy))
        bvec[ei+4] += signA*Area*np.sum(Ws*(Ee2x*Ux + Ee2y*Uy))
    
    Bmat[3,3] += Lt1*Lt1*(AREA_COEFF[tA,tB,tA,tB]*dot(GtC,GtC)-AREA_COEFF[tA,tB,tB,tC]*dot(GtA,GtC)-AREA_COEFF[tB,tC,tA,tB]*dot(GtA,GtC)+AREA_COEFF[tB,tC,tB,tC]*dot(GtA,GtA))
    Bmat[3,7] += Lt1*Lt2*(AREA_COEFF[tA,tB,tB,tC]*dot(GtA,GtC)-AREA_COEFF[tA,tB,tC,tA]*dot(GtB,GtC)-AREA_COEFF[tB,tC,tB,tC]*dot(GtA,GtA)+AREA_COEFF[tB,tC,tC,tA]*dot(GtA,GtB))
    Bmat[7,3] += Lt2*Lt1*(AREA_COEFF[tB,tC,tA,tB]*dot(GtA,GtC)-AREA_COEFF[tB,tC,tB,tC]*dot(GtA,GtA)-AREA_COEFF[tC,tA,tA,tB]*dot(GtB,GtC)+AREA_COEFF[tC,tA,tB,tC]*dot(GtA,GtB))
    Bmat[7,7] += Lt2*Lt2*(AREA_COEFF[tB,tC,tB,tC]*dot(GtA,GtA)-AREA_COEFF[tB,tC,tC,tA]*dot(GtA,GtB)-AREA_COEFF[tC,tA,tB,tC]*dot(GtA,GtB)+AREA_COEFF[tC,tA,tC,tA]*dot(GtB,GtB))
    
    A1, A2, A3 = As
    B1, B2, B3 = Bs
    C1, C2, C3 = Cs
   
    Ef1x = Lt1*(-B1*(A2 + B2*x + C2*y)*(A3 + B3*x + C3*y)/(8*Area**3) + B3*(A1 + B1*x + C1*y)*(A2 + B2*x + C2*y)/(8*Area**3))
    Ef1y = Lt1*(-C1*(A2 + B2*x + C2*y)*(A3 + B3*x + C3*y)/(8*Area**3) + C3*(A1 + B1*x + C1*y)*(A2 + B2*x + C2*y)/(8*Area**3))
    Ef2x = Lt2*(B1*(A2 + B2*x + C2*y)*(A3 + B3*x + C3*y)/(8*Area**3) - B2*(A1 + B1*x + C1*y)*(A3 + B3*x + C3*y)/(8*Area**3))
    Ef2y = Lt2*(C1*(A2 + B2*x + C2*y)*(A3 + B3*x + C3*y)/(8*Area**3) - C2*(A1 + B1*x + C1*y)*(A3 + B3*x + C3*y)/(8*Area**3))
    
    bvec[3] += signA*Area*np.sum(Ws*(Ef1x*Ux + Ef1y*Uy))
    bvec[7] += signA*Area*np.sum(Ws*(Ef2x*Ux + Ef2y*Uy))
    Bmat = Bmat * COEFF
    return Bmat, bvec

@njit(types.Tuple((c16[:], c16[:]))(f8[:,:], i8[:,:], c16[:], c16[:], i8[:], c16, c16[:,:,:], f8[:,:], i8[:,:]), cache=True, nogil=True)
def compute_bc_entries(vertices_local, tris, Bmat, Bvec, surf_triangle_indices, gamma, Ulocal_all, DPTs, tri_to_field):
    N = 64
    for i, itri in enumerate(surf_triangle_indices):

        vertex_ids = tris[:, itri]

        Ulocal = Ulocal_all[:,:, i]

        Bsub, bvec = ned2_tri_stiff_force(vertices_local[:,vertex_ids], gamma, Ulocal, DPTs)
        
        indices = tri_to_field[:, itri]
        
        Bmat[itri*N:(itri+1)*N] = Bsub.ravel()
        Bvec[indices] = Bvec[indices] + bvec
    return Bmat, Bvec

@njit(c16[:,:](f8[:,:], f8[:], c16), cache=True, nogil=True)
def ned2_tri_stiff(vertices, edge_lengths, gamma):
    ''' Nedelec-2 Triangle Stiffness matrix and forcing vector (For Boundary Condition of the Third Kind)

    '''
    local_edge_map = np.array([[0,1,0],[1,2,2]])
    Bmat = np.zeros((8,8), dtype=np.complex128)

    xs = vertices[0,:]
    ys = vertices[1,:]
    zs = vertices[2,:]

    ax1 = np.array([xs[1]-xs[0], ys[1]-ys[0], zs[1]-zs[0]])
    ax2 = np.array([xs[2]-xs[0], ys[2]-ys[0], zs[2]-zs[0]])
    ax1 = ax1/np.linalg.norm(ax1)
    ax2 = ax2/np.linalg.norm(ax2)

    axn = cross(ax1, ax2)
    ax2 = -cross(axn, ax1)
    basis = np.zeros((3,3), dtype=np.float64)
    basis[:,0] = ax1
    basis[:,1] = ax2
    basis[:,2] = axn
    basis = np.linalg.pinv(basis)
    lcs_vertices = basis @ np.ascontiguousarray(vertices)

    xs = lcs_vertices[0,:]
    ys = lcs_vertices[1,:]

    x1, x2, x3 = xs
    y1, y2, y3 = ys

    b1 = y2-y3
    b2 = y3-y1
    b3 = y1-y2
    c1 = x3-x2
    c2 = x1-x3
    c3 = x2-x1

    Ds = compute_distances(xs, ys, np.zeros_like(xs))

    GL1 = np.array([b1, c1])
    GL2 = np.array([b2, c2])
    GL3 = np.array([b3, c3])

    GLs = (GL1, GL2, GL3)

    Area = 0.5 * np.abs((x1 - x3) * (y2 - y1) - (x1 - x2) * (y3 - y1))

    letters = [1,2,3,4,5,6]

    tA, tB, tC = letters[0], letters[1], letters[2]
    GtA, GtB, GtC = GLs[0], GLs[1], GLs[2]
    
    Lt1, Lt2 = Ds[2, 0], Ds[1, 0]
    
    COEFF = gamma/(2*Area)**2
    AREA_COEFF = AREA_COEFF_CACHE_BASE * Area
    for ei in range(3):
        ei1, ei2 = local_edge_map[:, ei]
        Li = edge_lengths[ei]
        
        A = letters[ei1]
        B = letters[ei2]

        GA = GLs[ei1]
        GB = GLs[ei2]

        for ej in range(3):
            ej1, ej2 = local_edge_map[:, ej]
            Lj = edge_lengths[ej]

            C = letters[ej1]
            D = letters[ej2]

            GC = GLs[ej1]
            GD = GLs[ej2]

            Bmat[ei,ej] += Li*Lj*COEFF*(AREA_COEFF[A,B,C,D]*dot(GA,GC)-AREA_COEFF[A,B,C,C]*dot(GA,GD)-AREA_COEFF[A,A,C,D]*dot(GB,GC)+AREA_COEFF[A,A,C,C]*dot(GB,GD))
            Bmat[ei,ej+4] += Li*Lj*COEFF*(AREA_COEFF[A,B,D,D]*dot(GA,GC)-AREA_COEFF[A,B,C,D]*dot(GA,GD)-AREA_COEFF[A,A,D,D]*dot(GB,GC)+AREA_COEFF[A,A,C,D]*dot(GB,GD))
            Bmat[ei+4,ej] += Li*Lj*COEFF*(AREA_COEFF[B,B,C,D]*dot(GA,GC)-AREA_COEFF[B,B,C,C]*dot(GA,GD)-AREA_COEFF[A,B,C,D]*dot(GB,GC)+AREA_COEFF[A,B,C,C]*dot(GB,GD))
            Bmat[ei+4,ej+4] += Li*Lj*COEFF*(AREA_COEFF[B,B,D,D]*dot(GA,GC)-AREA_COEFF[B,B,C,D]*dot(GA,GD)-AREA_COEFF[A,B,D,D]*dot(GB,GC)+AREA_COEFF[A,B,C,D]*dot(GB,GD))
            
        Bmat[ei,3] += Li*Lt1*COEFF*(AREA_COEFF[A,B,tA,tB]*dot(GA,GtC)-AREA_COEFF[A,B,tB,tC]*dot(GA,GtA)-AREA_COEFF[A,A,tA,tB]*dot(GB,GtC)+AREA_COEFF[A,A,tB,tC]*dot(GB,GtA))
        Bmat[ei,7] += Li*Lt2*COEFF*(AREA_COEFF[A,B,tB,tC]*dot(GA,GtA)-AREA_COEFF[A,B,tC,tA]*dot(GA,GtB)-AREA_COEFF[A,A,tB,tC]*dot(GB,GtA)+AREA_COEFF[A,A,tC,tA]*dot(GB,GtB))
        Bmat[3,ei] += Lt1*Li*COEFF*(AREA_COEFF[tA,tB,A,B]*dot(GA,GtC)-AREA_COEFF[tA,tB,A,A]*dot(GB,GtC)-AREA_COEFF[tB,tC,A,B]*dot(GA,GtA)+AREA_COEFF[tB,tC,A,A]*dot(GB,GtA))
        Bmat[7,ei] += Lt2*Li*COEFF*(AREA_COEFF[tB,tC,A,B]*dot(GA,GtA)-AREA_COEFF[tB,tC,A,A]*dot(GB,GtA)-AREA_COEFF[tC,tA,A,B]*dot(GA,GtB)+AREA_COEFF[tC,tA,A,A]*dot(GB,GtB))
        Bmat[ei+4,3] += Li*Lt1*COEFF*(AREA_COEFF[B,B,tA,tB]*dot(GA,GtC)-AREA_COEFF[B,B,tB,tC]*dot(GA,GtA)-AREA_COEFF[A,B,tA,tB]*dot(GB,GtC)+AREA_COEFF[A,B,tB,tC]*dot(GB,GtA))
        Bmat[ei+4,7] += Li*Lt2*COEFF*(AREA_COEFF[B,B,tB,tC]*dot(GA,GtA)-AREA_COEFF[B,B,tC,tA]*dot(GA,GtB)-AREA_COEFF[A,B,tB,tC]*dot(GB,GtA)+AREA_COEFF[A,B,tC,tA]*dot(GB,GtB))
        Bmat[3,ei+4] += Lt1*Li*COEFF*(AREA_COEFF[tA,tB,B,B]*dot(GA,GtC)-AREA_COEFF[tA,tB,A,B]*dot(GB,GtC)-AREA_COEFF[tB,tC,B,B]*dot(GA,GtA)+AREA_COEFF[tB,tC,A,B]*dot(GB,GtA))
        Bmat[7,ei+4] += Lt2*Li*COEFF*(AREA_COEFF[tB,tC,B,B]*dot(GA,GtA)-AREA_COEFF[tB,tC,A,B]*dot(GB,GtA)-AREA_COEFF[tC,tA,B,B]*dot(GA,GtB)+AREA_COEFF[tC,tA,A,B]*dot(GB,GtB))
      
    
    Bmat[3,3] += Lt1*Lt1*COEFF*(AREA_COEFF[tA,tB,tA,tB]*dot(GtC,GtC)-AREA_COEFF[tA,tB,tB,tC]*dot(GtA,GtC)-AREA_COEFF[tB,tC,tA,tB]*dot(GtA,GtC)+AREA_COEFF[tB,tC,tB,tC]*dot(GtA,GtA))
    Bmat[3,7] += Lt1*Lt2*COEFF*(AREA_COEFF[tA,tB,tB,tC]*dot(GtA,GtC)-AREA_COEFF[tA,tB,tC,tA]*dot(GtB,GtC)-AREA_COEFF[tB,tC,tB,tC]*dot(GtA,GtA)+AREA_COEFF[tB,tC,tC,tA]*dot(GtA,GtB))
    Bmat[7,3] += Lt2*Lt1*COEFF*(AREA_COEFF[tB,tC,tA,tB]*dot(GtA,GtC)-AREA_COEFF[tB,tC,tB,tC]*dot(GtA,GtA)-AREA_COEFF[tC,tA,tA,tB]*dot(GtB,GtC)+AREA_COEFF[tC,tA,tB,tC]*dot(GtA,GtB))
    Bmat[7,7] += Lt2*Lt2*COEFF*(AREA_COEFF[tB,tC,tB,tC]*dot(GtA,GtA)-AREA_COEFF[tB,tC,tC,tA]*dot(GtA,GtB)-AREA_COEFF[tC,tA,tB,tC]*dot(GtA,GtB)+AREA_COEFF[tC,tA,tC,tA]*dot(GtB,GtB))


    return Bmat

def assemble_robin_bc_excited(field: Nedelec2,
                surf_triangle_indices: np.ndarray,
                Ufunc: Callable,
                gamma: np.ndarray,
                local_basis: np.ndarray,
                origin: np.ndarray,
                DPTs: np.ndarray):
    Bmat = field.empty_tri_matrix()
    Bmat[:] = 0
    
    Bvec = np.zeros((field.n_field,), dtype=np.complex128)

    vertices_local = optim_matmul(local_basis, field.mesh.nodes - origin[:,np.newaxis])# local_basis @ ()

    xflat, yflat = generate_points(vertices_local, field.mesh.tris, DPTs, surf_triangle_indices)

    Ulocal = Ufunc(xflat, yflat)

    Ulocal_all = Ulocal.reshape((3, DPTs.shape[1], surf_triangle_indices.shape[0]))

    Bmat, Bvec = compute_bc_entries(vertices_local, field.mesh.tris, Bmat, Bvec, surf_triangle_indices, gamma, Ulocal_all, DPTs, field.tri_to_field)
    Bmat = field.generate_csr(Bmat)

    return Bmat, Bvec

def assemble_robin_bc(field: Nedelec2,
                surf_triangle_indices: np.ndarray,
                gamma: np.ndarray):

    Bmat = field.empty_tri_matrix()
    row, col = field.empty_tri_rowcol()
    Bmat[:] = 0

    for i, itri in enumerate(surf_triangle_indices):

        vertex_ids = field.mesh.tris[:, itri]

        edge_lengths = field.tri_to_edge_lengths(itri)

        Bsub = ned2_tri_stiff(field.mesh.nodes[:,vertex_ids], edge_lengths, gamma)
        
        Bmat[field.trislice(itri)] = Bsub.ravel()

    Bmat = field.generate_csr(Bmat)
    return Bmat