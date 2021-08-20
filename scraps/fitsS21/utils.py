"""A set of simple utility functions for array math."""
import numpy as np
import scipy.signal as sps


def reduce_by_midpoint(array):
    """Subtract off and divide by middle array element.

    Sorts the array before picking mid-point, but returned
    array is not sorted."""
    midpoint = sorted(array)[int(np.round((len(array) - 1) / 2.0))]
    return (array - midpoint) / midpoint


def filter_data(array, filter_win_length=0):
    """Filter data with a Savitsky-Golay filter of window-length
    filter_win_length.

    filter_win_length must be odd and >= 3. This function will
    enforce that requirement by adding 1 to filter_win_length
    until it is satisfied."""

    # If no window length is supplied, defult to 1% of the data vector or 3
    if filter_win_length == 0:
        filter_win_length = int(np.round(len(array) / 100.0))
        if filter_win_length % 2 == 0:
            filter_win_length += 1
        if filter_win_length < 3:
            filter_win_length = 3

    return sps.savgol_filter(array, filter_win_length, 1)


def mask_array_ends(array, mask=None):
    """Return the ends of an array.

    If mask is an int, returns mask items from each end of array.
    If mask is a float, treats mask as a fraction of array length.
    If mask is a an array or a slice, return array[mask]."""
    if mask is None:
        masked_array = array
    elif type(mask) == float:
        pct_mask = int(len(array) * mask)
        masked_array = np.concatenate((array[:pct_mask], array[-pct_mask:]))
    elif type(mask) == int:
        masked_array = np.concatenate((array[:mask], array[-mask:]))
    elif type(mask) in [np.array, list, slice]:
        masked_array = array[mask]
    else:
        raise ValueError("Mask type must be number, array, or slice")

    return masked_array
