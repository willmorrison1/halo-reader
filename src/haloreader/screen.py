import numpy as np
from scipy.ndimage import uniform_filter

from haloreader.type_guards import is_ndarray
from haloreader.variable import Variable


def compute_noise_screen(
    intensity: Variable, doppler_velocity: Variable, range_: Variable,
    instensity_threshold=1.0025, velocity_threshold=2
) -> Variable:
    if (
        not is_ndarray(intensity.data)
        or not is_ndarray(doppler_velocity.data)
        or not is_ndarray(range_.data)
    ):
        raise TypeError
    # kernel size and threshold values have been chosen just by
    # visually checking the output
    intensity_mean_mask = uniform_filter(
        intensity.data, size=(21, 3)) > instensity_threshold
    velocity_abs_mean_mask = (
        uniform_filter(np.abs(doppler_velocity.data),
                       size=(21, 3)) < velocity_threshold
    )
    pulse_noise = np.zeros_like(intensity.data, dtype=bool)
    # 90 meters approx 3 times pulse length
    pulse_noise[:, range_.data < 90] = True
    below_one = intensity.data < 1
    return Variable(
        name="noise_screen",
        dimensions=intensity.dimensions,
        data=np.logical_not(intensity_mean_mask | velocity_abs_mean_mask)
        | pulse_noise
        | below_one,
    )
