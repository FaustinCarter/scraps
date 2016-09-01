import numpy as np
from scipy.special import digamma
import scipy.constants as sc

def qi_tlsAndMBT(params, temps, powers, data=None, eps=None):
    """A model of internal quality factor vs temperature and power, weighted by uncertainties.

    Parameters
    ----------
    params : ``lmfit.Parameters`` object
        Parameters must include ``['Fd', 'df', 'fRef', 'alpha', 'delta0']``.

    temps : ``numpy.Array``
        Array of temperature values to evaluate model at. May be 2D.

    powers : ``numpy.Array``
        Array of power values to evaluate model at. May be 2D.

    data : ``numpy.Array``
        Data values to compare to model. May also be ``None``, in which case
        function returns model.

    eps : ``numpy.Array``
        Uncertianties with which to weight residual. May also be ``None``, in
        which case residual is unwieghted.

    Returns
    -------

    residual : ``numpy.Array``
        The weighted or unweighted vector of residuals if ``data`` is passed.
        Otherwise, it returns the model.

    Note
    ----
    The following constraint must be satisfied::

        all(numpy.shape(x) == numpy.shape(data) for x in [temps, powers, eps])

    """
    pass

def f0_tlsAndMBT(params, temps, powers, data = None, eps = None):
    """A model of frequency shift vs temperature and power, weighted by uncertainties.

    Parameters
    ----------
    params : ``lmfit.Parameters`` object
        Parameters must include ``['Fd', 'df', 'fRef', 'alpha', 'delta0']``.

    temps : ``numpy.Array``
        Array of temperature values to evaluate model at. May be 2D.

    powers : ``numpy.Array``
        Array of power values to evaluate model at. May be 2D.

    data : ``numpy.Array``
        Data values to compare to model. May also be ``None``, in which case
        function returns model.

    eps : ``numpy.Array``
        Uncertianties with which to weight residual. May also be ``None``, in
        which case residual is unwieghted.

    Returns
    -------

    residual : ``numpy.Array``
        The weighted or unweighted vector of residuals if ``data`` is passed.
        Otherwise, it returns the model.

    Note
    ----
    The following constraint must be satisfied::

        all(numpy.shape(x) == numpy.shape(data) for x in [temps, powers, eps])

    """
    #Unpack parameter values from params
    Fd = params['Fd'].value
    fRef = params['f0'].value
    alpha = params['alpha'].value
    delta0 = params['delta0'].value*sc.e


    #Calculate model from parameters
    model = f0+f0*(Fd/sc.pi* #TLS contribution
             (np.real(digamma(0.5+sc.h*f0/(1j*2*sc.pi*sc.k*(temps))))
              -np.log(sc.h*f0/(2*sc.pi*sc.k*(temps))))-

             alpha/2.0* #MBD contribution
             (np.sqrt((2*sc.pi*sc.k*temps)/delta0)*
              np.exp(-delta0/(sc.k*temps))+
              2*np.exp(-delta0/(sc.k*temps))*
              np.exp(-sc.h*f0/(2*sc.k*temps))*
              np.i0(sc.h*f0/(2*sc.k*temps))))


    #Weight the residual if eps is supplied
    if data is not None:
        if eps is not None:
            residual = (model-data)/eps
        else:
            residual = (model-data)

        return residual
    else:
        return model
