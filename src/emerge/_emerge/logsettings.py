from loguru import logger
import sys
from typing import Literal
from enum import Enum
from pathlib import Path
import os


############################################################
#                          FORMATS                         #
############################################################

TRACE_FORMAT = (
    "{time: YY/MM/DD - (ddd) - HH:mm:ss.SSSS} | <green>{elapsed}</green> [ <level>{level}</level> ] "
    " <level>{message}</level>"
)
DEBUG_FORMAT = (
    "<green>{elapsed}</green> [<level>{level}</level>] "
    " <level>{message}</level>"
)
INFO_FORMAT = (
    "<green>{elapsed}</green> [<level>{level}</level>] "
    " <level>{message}</level>"
)
WARNING_FORMAT = (
    "<green>{elapsed}</green> [<level>{level}</level>] "
    " <level>{message}</level>"
)
ERROR_FORMAT = (
    "<green>{elapsed}</green> [<level>{level}</level>] "
    " <level>{message}</level>"
)
FORMAT_DICT = {
    'TRACE': TRACE_FORMAT,
    'DEBUG': DEBUG_FORMAT,
    'INFO': INFO_FORMAT,
    'WARNING': WARNING_FORMAT,
    'ERROR': ERROR_FORMAT,
}

LLTYPE = Literal['TRACE','DEBUG','INFO','WARNING','ERROR'] 


############################################################
#                      LOG CONTROLLER                     #
############################################################

class LogController:
    
    def __init__(self):
        logger.remove()
        self.std_handlers: list[int] = []
        self.file_handlers: list[int] = []
        self.level: LLTYPE = 'INFO'
        self.file_level: LLTYPE = 'INFO'
    
    def set_default(self):
        value = os.getenv("EMERGE_STD_LOGLEVEL", default="INFO")
        self.set_std_loglevel(value)

    def add_std_logger(self, loglevel: LLTYPE) -> None:
        handle_id = logger.add(sys.stderr, 
                level=loglevel, 
                format=FORMAT_DICT.get(loglevel, INFO_FORMAT))
        self.std_handlers.append(handle_id)

    def set_std_loglevel(self, loglevel: str):
        handler = {"sink": sys.stdout, 
                   "level": loglevel, 
                   "format": FORMAT_DICT.get(loglevel, INFO_FORMAT)}
        logger.configure(handlers=[handler])
        self.level = loglevel
        os.environ["EMERGE_STD_LOGLEVEL"] = loglevel
        

    def set_write_file(self, path: Path, loglevel: str = 'TRACE'):

        handler_id = logger.add(str(path / 'logging.log'), mode='w', level=loglevel, format=FORMAT_DICT.get(loglevel, INFO_FORMAT), colorize=False, backtrace=True, diagnose=True)
        self.file_handlers.append(handler_id)
        self.file_level = loglevel
        os.environ["EMERGE_FILE_LOGLEVEL"] = loglevel

LOG_CONTROLLER = LogController()