import numpy as np
import scipy.signal as sps


def mask_array(array, mask):
    if mask is None:
        masked_array = array
    elif type(mask) == float:
        pct_mask = int(len(array)*mask)
        masked_array = np.concatenate((array[:pct_mask], array[-pct_mask:]))
    elif type(mask) == int:
        masked_array = np.concatenate((array[:mask], array[-mask:]))
    elif type(mask) in [np.array, list, slice]:
        masked_array = array[mask]
    else:
        raise ValueError("Mask type must be number, array, or slice")

    return masked_array


def reduce_by_midpoint(array):
    midpoint = array[int(np.round((len(array)-1)/2.0))]
    return (array-midpoint)/midpoint


def moving_average(array, filter_win_length=3, poly_order=3):
    assert (filter_win_length % 2 == 1) and (filter_win_length >= 3), "Filter window length must be odd and greater than 3."
    return sps.savgol_filter(array, filter_win_length, poly_order)


def find_nearest(x, array):
    """Return the value of array that is closest to the value of x"""
    ix = np.abs(array-x).argmin()
    return array[ix]
