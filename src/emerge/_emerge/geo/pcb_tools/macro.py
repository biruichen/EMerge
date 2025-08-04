from typing import Any
from dataclasses import dataclass
import math
import re

def rotation_angle(a: tuple[float, float], b: tuple[float, float]) -> float:
    """
    Returns the signed angle in degrees you must rotate vector a
    to align with vector b. Positive = CCW, Negative = CW.
    
    a, b: (vx, vy) tuples in the plane.
    """
    ax, ay = a
    bx, by = b

    # compute 2D cross product (scalar) and dot product
    cross = ax * by - ay * bx
    dot   = ax * bx + ay * by

    # atan2(cross, dot) gives angle between -π and π
    angle_rad = math.atan2(cross, dot)
    angle_deg = math.degrees(angle_rad)
    return -angle_deg

@dataclass
class Instruction:
    instr: str
    args: tuple[int | float, ...]
    kwargs: dict[str,float] | None = None

    def __post_init__(self):
        if self.kwargs is None:
            self.kwargs = {}

symbols = ['=','>','v','^','<','@','/','\\','T']
char_class = ''.join(re.escape(c) for c in symbols)

pattern = re.compile(rf'([{char_class}])([\d\,\.\-]+)')

def parse_macro(pathstring: str, width: int | float, direction: tuple[float, float]) -> list[Instruction]:
    instructions = pattern.findall(pathstring.replace(' ',''))
    
    oi = []
    for com, val in instructions:
        if ',' in val:
            val_list: list[float] = [float(x) for x in val.split(',')]
            ival, width = val_list[0], val_list[1]
        else:
            ival = float(val)
        if com == '=':
            oi.append(Instruction('straight',(ival,),{'width': width})) #type: ignore
        elif com == '>':
            oi.append(Instruction('turn',(rotation_angle(direction, (1., 0.)),) ))
            oi.append(Instruction('straight',(ival,),{'width': width}))
            direction = (1. ,0. )
        elif com == '<':
            oi.append(Instruction('turn',(rotation_angle(direction, (-1., 0.)),) ))
            oi.append(Instruction('straight',(ival,),{'width': width}))
            direction = (-1.,0.)
        elif com == 'v':
            oi.append(Instruction('turn',(rotation_angle(direction, (0.,-1.)),) ))
            oi.append(Instruction('straight',(ival,),{'width': width}))
            direction = (0.,-1.)
        elif com == '^':
            oi.append(Instruction('turn',(rotation_angle(direction, (0.,1.)),) ))
            oi.append(Instruction('straight',(ival,),{'width': width}))
            direction = (0.,1.)
        elif com == '\\':
            oi.append(Instruction('turn',(90,) ))
            oi.append(Instruction('straight',(ival,),{'width': width}))
            direction = (direction[1],-direction[0])
        elif com == '/':
            oi.append(Instruction('turn',(-90,) ))
            oi.append(Instruction('straight',(ival,),{'width': width}))
            direction = (-direction[1],direction[0])
        elif com == 'T':
            oi.append(Instruction('taper',(ival,),{'width': width}))
        elif com == '@':
            oi.append(Instruction('turn',(ival,),{'width': width}))
    return oi