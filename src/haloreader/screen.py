import numpy as np
from scipy.ndimage import uniform_filter

from haloreader.type_guards import is_ndarray
from haloreader.variable import Variable

INTENSITY_THRESHOLD_DEFAULT = 1.0075857757502917
VELOCITY_THRESHOLD_DEFAULT = 2


def compute_noise_screen(
    intensity: Variable, doppler_velocity: Variable, range_: Variable,
    **kwargs,
) -> Variable:
    instensity_threshold = kwargs.get(
        "instensity_threshold", INTENSITY_THRESHOLD_DEFAULT)
    velocity_threshold = kwargs.get(
        "velocity_threshold", VELOCITY_THRESHOLD_DEFAULT)
    if (
        not is_ndarray(intensity.data)
        or not is_ndarray(doppler_velocity.data)
        or not is_ndarray(range_.data)
        or not isinstance(instensity_threshold, (int, float))
        or not isinstance(velocity_threshold, (int, float))
    ):
        raise TypeError

    # kernel size and threshold values have been chosen just by
    # visually checking the output
    intensity_mean_mask = uniform_filter(
        intensity.data, size=(21, 3)) < instensity_threshold
    velocity_abs_mean_mask = (
        uniform_filter(np.abs(doppler_velocity.data),
                       size=(21, 3)) > velocity_threshold
    )
    pulse_noise = np.zeros_like(intensity.data, dtype=bool)
    # 90 meters approx 3 times pulse length
    pulse_noise[:, range_.data < 90] = True
    below_one = intensity.data < 1
    return Variable(
        name="noise_screen",
        dimensions=intensity.dimensions,
        data=(intensity_mean_mask | velocity_abs_mean_mask)
        | pulse_noise
        | below_one,
        comment=(f"Screen has instensity_threshold: {instensity_threshold},"
                 f"velocity_threshold: {velocity_threshold}")
    )
