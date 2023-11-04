import numpy as np
from scipy.interpolate import CubicSpline
from numpy import ndarray


class Spline:
    def __init__(self, xs: ndarray, ys: ndarray):
        self.cs = []
        for i in range(ys.shape[-1]):
            self.cs.append(CubicSpline(xs, ys[:, i]))

    def sample_point(self, x: float):
        return np.array([cs(x) for cs in self.cs])
