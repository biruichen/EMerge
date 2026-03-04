from .baselib import CompiledLib
import numpy as np

class IRONLib(CompiledLib):

    @staticmethod
    def ned2_tet_interp(coords: np.ndarray,
                    solutions: np.ndarray, 
                    tets: np.ndarray, 
                    tris: np.ndarray,
                    edges: np.ndarray,
                    nodes: np.ndarray,
                    tet_to_field: np.ndarray,
                    tetids: np.ndarray):
        from emerge_iron import tetrahedral_interp
        out = tetrahedral_interp(coords, solutions, tets, tris, edges, nodes, tet_to_field, tetids)
        return (out[0,:].flatten(), out[1,:].flatten(), out[2,:].flatten())
    
    @staticmethod
    def ned2_tet_interp_curl(coords: np.ndarray,
                         solutions: np.ndarray, 
                         tets: np.ndarray, 
                         tris: np.ndarray,
                         edges: np.ndarray,
                         nodes: np.ndarray,
                         tet_to_field: np.ndarray,
                         c: np.ndarray,
                         tetids: np.ndarray):
        from emerge_iron import tetrahedral_interp_curl
        out = tetrahedral_interp_curl(coords, solutions, tets, tris, edges, nodes, tet_to_field, c, tetids)
        return (out[0,:].flatten(), out[1,:].flatten(), out[2,:].flatten())

    @staticmethod
    def index_interp(coords: np.ndarray,
                tets: np.ndarray, 
                nodes: np.ndarray,
                tetids: np.ndarray):
        from emerge_iron import index_interp
        return index_interp(coords, tets, nodes, tetids)