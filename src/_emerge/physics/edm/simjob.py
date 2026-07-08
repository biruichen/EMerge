import numpy as np
import os
from scipy.sparse import csr_matrix, save_npz, load_npz


class SimJob:

    def __init__(self, 
                 A: csr_matrix,
                 b: np.ndarray,
                 freq: float,
                 cache_factorization: bool,
                 ):

        self.A: csr_matrix = A
        self.b: np.ndarray = b
        self.P: csr_matrix = None
        self.Pd: csr_matrix = None
        self.has_periodic: bool = False

        self.freq: float = freq
        self.k0: float = 2*np.pi*freq/299792458
        self.cache_factorization: bool = cache_factorization
        self._fields: dict[int, np.ndarray] = dict()
        self.port_vectors: dict = None
        self.solve_ids = None

        self.store_limit = None
        self.relative_path = None
        self._store_location = {}
        self._stored: bool = False

        self._active_port: int = None

        self.store_if_needed()

    def maybe_store(self, matrix, name):
        
        if self.store_limit is None:
            return matrix
        
        if matrix is not None and matrix.nnz > self.store_limit:
            # Create temp directory if needed
            os.makedirs(self.relative_path, exist_ok=True)
            path = os.path.join(self.relative_path, f"csr_{str(self.freq).replace('.','_')}_{name}.npz")
            save_npz(path, matrix, compressed=False)
            self._store_location[name] = path
            self._stored = True
            return None  # Unload from memory
        return matrix
    
    def store_if_needed(self):
        self.A = self.maybe_store(self.A, 'A')
        if self.has_periodic:
            self.P = self.maybe_store(self.P, 'P')
            self.Pd = self.maybe_store(self.Pd, 'Pd')

    def load_if_needed(self, name):
        if name in self._store_location:
            return load_npz(self._store_location[name])
        return getattr(self, name)

    def iter_Ab(self):
        reuse_factorization = False

        for key, mode in self.port_vectors.items():
            # Set port as active and add the port mode to the forcing vector
            self._active_port = key

            b_active = self.b + mode
            A = self.load_if_needed('A')
            

            if self.has_periodic:
                P = self.load_if_needed('P')
                Pd = self.load_if_needed('Pd')
                b_active = Pd @ b_active
                A = Pd @ A @ P

            yield A, b_active, self.solve_ids, reuse_factorization

            reuse_factorization = True
        
        self.cleanup()
    
    def submit_solution(self, solution):
        # Solve the Ax=b problem

        if self.has_periodic:
            solution = self.P @ solution
        # From now reuse the factorization

        self._fields[self._active_port] = solution

        self.routine = None
    
    def cleanup(self):
        if not self._stored:
            return
        
        if not os.path.isdir(self.relative_path):
            return

        # Remove only the files we saved
        for path in self._store_location.values():
            if os.path.isfile(path):
                os.remove(path)

        # If the directory is now empty, remove it
        if not os.listdir(self.relative_path):
            os.rmdir(self.relative_path)