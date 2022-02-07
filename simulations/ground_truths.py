import math
from typing import List

import numpy as np


def fft_indices(n) -> List:
    a = list(range(0, math.floor(n / 2) + 1))
    b = reversed(range(1, math.floor(n / 2)))
    b = [-i for i in b]
    return a + b


def gaussian_random_field(pk, x_dim: int, y_dim: int) -> np.array:
    """Generate 2D gaussian random field: https://andrewwalker.github.io/statefultransitions/post/gaussian-fields/"""

    def pk2(kx, ky):
        if kx == 0 and ky == 0:
            return 0.0

        return np.sqrt(pk(np.sqrt(kx ** 2 + ky ** 2)))

    noise = np.fft.fft2(np.random.normal(size=(y_dim, x_dim)))
    amplitude = np.zeros((y_dim, x_dim))

    for i, kx in enumerate(fft_indices(y_dim)):
        for j, ky in enumerate(fft_indices(x_dim)):
            amplitude[i, j] = pk2(kx, ky)

    random_field = np.fft.ifft2(noise * amplitude).real
    normalized_random_field = (random_field - np.min(random_field)) / (np.max(random_field) - np.min(random_field))

    return normalized_random_field
