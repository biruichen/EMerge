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

from loguru import logger
import sys
from typing import Literal
from pathlib import Path
import os
from collections import deque

_LOG_BUFFER = deque()

def _log_sink(message):
    _LOG_BUFFER.append(message)
############################################################
#                          FORMATS                         #
############################################################

TRACE_FORMAT = (
    "{time:ddd YY/MM/DD HH:mm:ss.SSSS} {level:<7} {thread.id:<15} {line:>4}: "
    "{message}"
)

DEBUG_FORMAT = (
    "<green>{elapsed}</green>  <level>{level:<7}</level>: "
    "<level>{message}</level>"
)
INFO_FORMAT = (
    "<green>{elapsed}</green>  <level>{level:<7}</level>: "
    "<level>{message}</level>"
)
WARNING_FORMAT = (
    "<green>{elapsed}</green>  <level>{level:<7}</level>: "
    "<level>{message}</level>"
)
ERROR_FORMAT = (
    "<green>{elapsed}</green>  <level>{level:<7}</level>: "
    "<level>{message}</level>"
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
        self.level: str = 'INFO'
        self.file_level: str = 'INFO'
    
    def set_default(self):
        value = os.getenv("EMERGE_STD_LOGLEVEL", default="INFO")
        self.set_std_loglevel(value)

    def add_std_logger(self, loglevel: LLTYPE) -> None:
        handle_id = logger.add(sys.stderr, 
                level=loglevel, 
                format=FORMAT_DICT.get(loglevel, INFO_FORMAT))
        self.std_handlers.append(handle_id)

    def _set_log_buffer(self):
        logger.add(_log_sink)
        
    def _flush_log_buffer(self):
        for msg in list(_LOG_BUFFER):
            logger.opt(depth=6).log(msg.record["level"].name, msg.record["message"])
        _LOG_BUFFER.clear()
        
    def set_std_loglevel(self, loglevel: str):
        handler = {"sink": sys.stdout, 
                   "level": loglevel, 
                   "format": FORMAT_DICT.get(loglevel, INFO_FORMAT)}
        logger.configure(handlers=[handler]) # type: ignore
        self.level = loglevel
        os.environ["EMERGE_STD_LOGLEVEL"] = loglevel

    def set_write_file(self, path: Path, loglevel: str = 'TRACE'):
        handler_id = logger.add(str(path / 'logging.log'), mode='w', level=loglevel, format=FORMAT_DICT.get(loglevel, INFO_FORMAT), colorize=False, backtrace=True, diagnose=True)
        self.file_handlers.append(handler_id)
        self.file_level = loglevel
        os.environ["EMERGE_FILE_LOGLEVEL"] = loglevel

LOG_CONTROLLER = LogController()