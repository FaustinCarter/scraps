import warnings

from scraps.fitsS21 import hanger_resonator

warnings.warn(
    DeprecationWarning(
        "This module has been deprecated in favor of scraps.fitsS21.hanger_resonator"
    )
)


def cmplxIQ_fit(paramsVec, res, residual=True, **kwargs):
    """Return complex S21 resonance model or, if data is specified, a residual.

    This function is deprecated and will be removed in a future version. Use hanger_resonator.hanger_fit.

    Parameters
    ----------
    params : list-like
        A an ``lmfit.Parameters`` object containing (df, f0, qc, qi, gain0, gain1, gain2, pgain0, pgain1, pgain2)
    res : scraps.Resonator object
        A Resonator object.
    residual : bool
        Whether to return a residual (True) or to return the model calcuated at the frequencies present in res (False).

    Keyword Arguments
    -----------------
    freqs : list-like
        A list of frequency points at which to calculate the model. Only used if `residual=False`

    remove_baseline : bool
        Whether or not to remove the baseline during calculation (i.e. ignore pgain and gain polynomials). Default is False.

    only_baseline: bool
        Whether or not to calculate and return only the baseline. Default is False.

    Returns
    -------
    model or (model-data)/eps : ``numpy.array``
        If residual=True is specified, the return is the residuals weighted by the uncertainties. If residual=False, the return is the model
        values calculated at the frequency points. The returned array is in the form
        ``I + Q`` or ``residualI + residualQ``.

    """

    warnings.warn(
        DeprecationWarning(
            "This function has been renamed hanger_resonator.hanger_fit. cmplxIQ_fit will be removed in a future version"
        )
    )

    return hanger_resonator.hanger_fit(paramsVec, res, residual, **kwargs)


def cmplxIQ_params(res, **kwargs):
    """Initialize fitting parameters used by the cmplxIQ_fit function.

    Parameters
    ----------
    res : ``scraps.Resonator`` object
        The object you want to calculate parameter guesses for.

    Keyword Arguments
    -----------------
    fit_quadratic_phase : bool
        This determines whether the phase baseline is fit by a line or a
        quadratic function. Default is False for fitting only a line.

    hardware : string {'VNA', 'mixer'}
        This determines whether or not the Ioffset and Qoffset parameters are
        allowed to vary by default.

    use_filter : bool
        Whether or not to use a smoothing filter on the data before calculating
        parameter guesses. This is especially useful for very noisy data where
        the noise spikes might be lower than the resonance minimum.

    filter_win_length : int
        The length of the window used in the Savitsky-Golay filter that smoothes
        the data when ``use_filter == True``. Default is ``0.1 * len(data)`` or
        3, whichever is larger.

    Returns
    -------
    params : ``lmfit.Parameters`` object

    """

    warnings.warn(
        DeprecationWarning(
            "This function has been renamed hanger_resonator.hanger_params. cmplxIQ_fit will be removed in a future version"
        )
    )

    return hanger_resonator.hanger_params(res, **kwargs)
