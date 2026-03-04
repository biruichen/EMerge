
import numpy as np

class CompiledLib:
    @staticmethod
    def ned2_tet_interp(coords: np.ndarray,
                    solutions: np.ndarray, 
                    tets: np.ndarray, 
                    tris: np.ndarray,
                    edges: np.ndarray,
                    nodes: np.ndarray,
                    tet_to_field: np.ndarray,
                    tetids: np.ndarray):
        from .base.interp import ned2_tet_interp
        return ned2_tet_interp(coords, solutions, tets, tris, edges, nodes, tet_to_field, tetids)
    
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
        from .base.interp import ned2_tet_interp_curl

        return ned2_tet_interp_curl(coords, solutions, tets, tris, edges, nodes, tet_to_field, c, tetids)
    
    @staticmethod
    def ned2_tri_interp_full(coords: np.ndarray,
                    solutions: np.ndarray, 
                    tris: np.ndarray,
                    nodes: np.ndarray,
                    tri_to_field: np.ndarray):
        from .base.interp import ned2_tri_interp_full
        return ned2_tri_interp_full(coords, solutions, tris, nodes, tri_to_field)
    
    @staticmethod
    def ned2_tri_interp(coords: np.ndarray,
                    solutions: np.ndarray, 
                    tris: np.ndarray,
                    nodes: np.ndarray,
                    tri_to_field: np.ndarray):
        from .base.interp import ned2_tri_interp
        return ned2_tri_interp(coords, solutions, tris, nodes, tri_to_field)
    
    @staticmethod
    def ned2_tri_interp_curl(coords: np.ndarray,
                    solutions: np.ndarray, 
                    tris: np.ndarray,
                    nodes: np.ndarray,
                    tri_to_field: np.ndarray,
                    diadic: np.ndarray,
                    beta: float):
        from .base.interp import ned2_tri_interp_curl
        return ned2_tri_interp_curl(coords, solutions, tris, nodes, tri_to_field, diadic, beta)
    

    @staticmethod
    def index_interp(coords: np.ndarray,
                tets: np.ndarray, 
                nodes: np.ndarray,
                tetids: np.ndarray):
        from .base.interp import index_interp
        return index_interp(coords, tets, nodes, tetids)
    