import numpy as np
from scipy.special import digamma, i0, k0
import scipy.constants as sc
import scipy.interpolate as si


def qi_tlsAndMBT(params, temps, powers, data=None, eps=None, **kwargs):
    """A model of internal quality factor vs temperature and power, weighted by uncertainties.

    Parameters
    ----------
    params : ``lmfit.Parameters`` object
        Parameters must include ``['Fd', 'q0', 'f0', 'alpha', 'delta0']``.

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

    It is almost certain that this model does NOT apply to your device as the
    assumptions it makes are highly constraining and ignore several material
    parameters. It is included here more as an example for how to write a model
    than anything else, and it does at least qualitatively describe the behavior
    of most superconducting resonators.

    This model is taken from J. Gao's Caltech dissertation (2008) and the below
    equations are from that work.

    (2.54) gives for MBD: ``1/Q(T)-1/Q(0) = alpha * R(T)/X(0)``

    (5.72) and (5.65) give for TLS: ``1/Q(T)-1/Q(0) = Fd*tanh(hf/2kT)/sqrt(1+P/P0)``

    R(T)/X(0) calculated from (2.80), (2.89), and (2.90), using the ``deltaBCS``
    function in this module for returning gap as a function of temperature.

    """

    Fd = params['Fd'].value
    Pc = params['Pc'].value
    f0 = params['f0'].value
    q0 = params['q0'].value
    delta0 = params['delta0'].value*sc.e
    alpha = params['alpha'].value

    units = kwargs.pop('units', 'mK')
    assert units in ['mK', 'K'], "Units must be 'mK' or 'K'."

    if units == 'mK':
        ts = temps*0.001

    #Assuming thick film local limit
    #Other good options are 1 or 1/3
    gamma = kwargs.pop('gamma', 0.5)

    #Calculate tc from BCS relation
    tc = delta0/(1.76*sc.k)

    #Get the reduced energy gap
    deltaR = deltaBCS(ts/tc)

    #And the energy gap at T
    deltaT = delta0*deltaR

    #Pack all these together for convenience
    zeta = sc.h*f0/(2*sc.k*ts)

    #An optional power calibration in dB
    #Without this, the parameter Pc is meaningless
    pwr_cal_dB = kwargs.pop('pwr_cal_dB', 0)
    ps = powers+pwr_cal_dB

    #Working in inverse Q since they add and subtract  nicely

    #Calculate the inverse Q from TLS
    invQtls = Fd*np.tanh(zeta)/np.sqrt(1.0+10**(ps/10.0)/Pc)

    #Calculte the inverse Q from MBD
    invQmbd = alpha*gamma*4*deltaR*np.exp(-deltaT/(sc.k*ts))*np.sinh(zeta)*k0(zeta)

    #Get the difference from the total Q and
    model = 1.0/(invQtls + invQmbd + 1.0/q0)

    #Weight the residual if eps is supplied
    if data is not None:
        if eps is not None:
            residual = (model-data)/eps
        else:
            residual = (model-data)

        return residual
    else:
        return model



def f0_tlsAndMBT(params, temps, powers, data = None, eps = None, **kwargs):
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

    It is almost certain that this model does NOT apply to your device as the
    assumptions it makes are highly constraining and ignore several material
    parameters. It is included here more as an example for how to write a model
    than anything else, and it does at least qualitatively describe the behavior
    of most superconducting resonators.

    This model is taken from J. Gao's Caltech dissertation (2008) and the below
    equations are from that work.

    (2.54) gives for MBD (f(T)-f(0))/f(0) = -alpha*0.5*(X(T)-X(0))/X(0)

    (5.71) gives for TLS "" = Fd/pi * (usual TLS expression from Phillips)

    (X(T)-X(0))/X(0) calculated from (2.80), (2.89), and (2.90), using the ``deltaBCS``
    function in this module for returning gap as a function of temperature.

    """
    #Unpack parameter values from params
    Fd = params['Fd'].value
    f0 = params['f0'].value
    alpha = params['alpha'].value
    delta0 = params['delta0'].value*sc.e

    #Set temperature units
    units = kwargs.pop('units', 'mK')
    assert units in ['mK', 'K'], "Units must be 'mK' or 'K'."

    if units == 'mK':
        ts = temps*0.001

    #Calculate tc from BCS relation
    tc = delta0/(1.76*sc.k)

    #Get the reduced energy gap
    deltaR = deltaBCS(ts/tc)

    #And the energy gap at T
    deltaT = delta0*deltaR

    #Pack all these together for convenience
    zeta = sc.h*f0/(2*sc.k*ts)

    #Assuming thick film local limit
    #Other good options are 1 or 1/3
    gamma = kwargs.pop('gamma', 0.5)

    #TLS contribution
    dfTLS = Fd/sc.pi*(np.real(digamma(0.5+zeta/(1j*sc.pi)))-np.log(zeta/sc.pi))

    #MBD contribution
    dfMBD = alpha*gamma*0.5*(deltaR*(1-
                        2*np.exp(-deltaT/(sc.k*ts))
                        *np.exp(-zeta)
                        *i0(zeta))-1)


    #Calculate model from parameters
    model = f0+f0*(dfTLS + dfMBD)


    #Weight the residual if eps is supplied
    if data is not None:
        if eps is not None:
            residual = (model-data)/eps
        else:
            residual = (model-data)

        return residual
    else:
        return model

@np.vectorize
def deltaBCS(temp):
    r"""Return the reduced BCS gap deltar = delta(T)/delta(T=0).

    Parameters
    ----------
    temp : float
        reduced temperature t=(T/Tc) where Tc is critical temperature.

    Returns
    -------
    deltar : float
        Superconducting gap, delta, normalized by delta(T=0)

    Note
    ----
    Function interpolates data from Muhlschlegel (1959). For temperature below
    1.8 K, the following functional form is used::

        gap = np.exp(-np.sqrt(3.562*temp)*np.exp(-1.764/temp))

    For temperatures below 50 mK, it returns 1.

    """

    #These data points run from t=0.18 to t=1
    #in steps of 0.02 from Muhlschlegel (1959)
    delta_calc = [1.0, 0.9999, 0.9997, 0.9994, 0.9989,
           0.9982, 0.9971, 0.9957, 0.9938, 0.9915,
           0.9885, 0.985,  0.9809, 0.976,  0.9704,
           0.9641, 0.9569, 0.9488, 0.9399, 0.9299,
           0.919,  0.907,  0.8939, 0.8796, 0.864,
           0.8471, 0.8288, 0.8089, 0.7874, 0.764,
           0.7386, 0.711,  0.681,  0.648,  0.6117,
           0.5715, 0.5263, 0.4749, 0.4148, 0.3416,
           0.2436, 0.0]

    t_calc = np.linspace(0.18, 1, len(delta_calc))

    if 1 >= temp >= 0.3:
        #interpolate data from table
        gap = float(si.interp1d(t_calc, delta_calc, kind='cubic')(temp))
    elif temp > 1 or temp < 0:
        gap = 0.0
    elif temp < 0.05:
        gap = 1.0
    else:
        gap = np.exp(-np.sqrt(3.562*temp)*np.exp(-1.764/temp))

    return gap
