from __future__ import annotations
import matplotlib.pyplot as plt
import numpy as np
from typing import Any
from itertools import cycle

class Style:

    def __init__(self):
        self.linecolors: list = ['k']
        self.linestyles: list[str] = ['-']
        self.markers: list[str] = [None]

        self.color_cycler = cycle(self.linecolors)
        self.style_cycler = cycle(self.linestyles)
        self.marker_cycle = cycle(self.markers)

    def get_line_properties(self, color: str, linestyle: str, marker: str) -> dict[str, Any]:
        if color is None:
            color = next(self.color_cycler)
        if linestyle is None:
            linestyle = next(self.style_cycler)
        if marker is None:
            marker = next(self.marker_cycle)
        return dict(color=color, linestyle=linestyle, marker=marker)

    def dress(self, axis):
        axis.grid(True, axis='both', color='k')

class Grapher:

    def __init__(self):
        self.axes: list = []
        self.axes_grid: list[list] = []
        self.fig = None
        self.style: Style = Style()
        self.nrows: int = 0
        self.ncols: int = 0
        self.current: int = 0

        self.cur_axis = None
    
    @property
    def ax(self) -> plt.axis:
        if self.cur_axis is None:
            return self.axes[0]
        return self.cur_axis
    
    def reset(self):
        self.cur_axis = None

    def __call__(self, i: int = None, j: int = None) -> Grapher:
        if i is None:
            self.cur_axis = self.axes[0]
        return self
        if j is None and i is not None:
            self.cur_axis = self.axes[i]
        else:
            self.cur_axis = self.axes[i,j]
        return self
    
    def new(self, rows: int = 1, cols: int = 1) -> Grapher:
        self.fig, self.axes = plt.subplots(rows, cols)
        if rows==1 and cols==1:
            self.axes = [self.axes]
        self.nrows = rows
        self.ncols = cols 
        return self

    def line(self,
             xs: np.ndarray,
             ys: np.ndarray,
             color: str = None,
             linestyle: str = None,
             marker: str = None,
             dB: bool = False) -> Grapher:
        if dB:
            ys = 20*np.log10(np.abs(ys))
        self.ax.plot(xs, ys, **self.style.get_line_properties(color=color, linestyle=linestyle, marker=marker))
        self.reset()
        return self

    def show(self):
        for ax in self.axes:
            self.style.dress(ax)
        plt.show()

gr = Grapher().new()

xs = np.linspace(1e9, 2e9, 1001)
ys = np.sin(xs/1e8)

gr().line(xs, ys).show()