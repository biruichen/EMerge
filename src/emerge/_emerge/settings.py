
from typing import Literal

class Settings:
    
    def __init__(self):
        self.mw_2dbc: bool = True
        self.mw_2dbc_lim: float = 10.0
        self.mw_2dbc_peclim: float = 1e8
        self.mw_3d_peclim: float = 1e7
        
DEFAULT_SETTINGS = Settings()