import numpy as np
from scipy.special import digamma
import scipy.constants as sc

import barmat


def qi_tlsAndMBT(params, temps, powers, data=None, eps=None, **kwargs):
    """A model of internal quality factor vs temperature and power, weighted by uncertainties.

    Parameters
    ----------
    params : ``lmfit.Parameters`` object
        Parameters must include ``['Fd', 'q0', 'f0', 'alpha', 'tc', 'vf',
        'london0', 'mfp', 'bcs']``.

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

    This model does a complete numerical calculation of the Mattis-Bardeen
    surface impedance.

    Some equations are taken from J. Gao's Caltech dissertation (2008):

    (2.54) gives for MBD: ``1/Q(T)-1/Q(0) = alpha * R(T)/X(0)``

    (5.72) and (5.65) give for TLS: ``1/Q(T)-1/Q(0) = Fd*tanh(hf/2kT)/sqrt(1+P/P0)``

    R(T)/X(0) is calculated using the barmat python package at:
    http://github.com/FaustinCarter/barmat.

    """

    #TLS params
    Fd = params['Fd'].value
    Pc = params['Pc'].value
    f0 = params['f0'].value
    q0 = params['q0'].value

    #Barmat params
    tc = params['tc'].value
    vf = params['vf'].value
    london0 = params['london0'].value
    mfp = params['mfp'].value
    bcs = params['bcs'].value

    #Kinetic inductance param
    alpha = params['alpha'].value

    units = kwargs.pop('units', 'mK')
    assert units in ['mK', 'K'], "Units must be 'mK' or 'K'."

    if units == 'mK':
        temps = temps*0.001

    #Pack all these together for convenience
    zeta = sc.h*f0/(2*sc.k*temps)

    #An optional power calibration in dB
    #Without this, the parameter Pc is meaningless
    pwr_cal_dB = kwargs.pop('pwr_cal_dB', 0)
    ps = powers+pwr_cal_dB

    #Working in inverse Q since they add and subtract  nicely

    #Calculate the inverse Q from TLS
    invQtls = Fd*np.tanh(zeta)/np.sqrt(1.0+10**(ps/10.0)/Pc)

    #Calculate the inverse Q from MBD using barmat
    fr = sc.h*f0/(sc.e*barmat.tools.get_delta0(tc, bcs))

    Z = barmat.get_Zvec(temps[0]/tc, tc, vf, london0, mfp=mfp, bcs=bcs, fr=fr,
                        axis='temperature', output_depths=False,
                        boundary='diffuse')

    invQmbd = alpha*(Z.real - Z.real[0])/Z.imag[0]

    invQmbd = np.array([invQmbd]*len(temps))

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

    This model does a complete numerical calculation of the Mattis-Bardeen
    surface impedance. It assumes no power dependence in the frequency shift

    Some equations are taken from J. Gao's Caltech dissertation (2008):

    (2.54) gives for MBD (f(T)-f(0))/f(0) = -alpha*0.5*(X(T)-X(0))/X(0)

    (5.71) gives for TLS "" = Fd/pi * (usual TLS expression from Phillips)

    (X(T)-X(0))/X(0) calculated is calculated using the barmat python package at:
    http://github.com/FaustinCarter/barmat.

    """
    #Unpack parameter values from params

    #TLS params
    Fd = params['Fd'].value
    f0 = params['f0'].value

    #Barmat params
    tc = params['tc'].value
    vf = params['vf'].value
    london0 = params['london0'].value
    mfp = params['mfp'].value
    bcs = params['bcs'].value

    #Kinetic inductance param
    alpha = params['alpha'].value

    #Set temperature units
    units = kwargs.pop('units', 'mK')
    assert units in ['mK', 'K'], "Units must be 'mK' or 'K'."

    if units == 'mK':
        temps = temps*0.001

    #Pack all these together for convenience
    zeta = sc.h*f0/(2*sc.k*temps)

    #TLS contribution
    dfTLS = Fd/sc.pi*(np.real(digamma(0.5+zeta/(1j*sc.pi)))-np.log(zeta/sc.pi))

    #MBD contribution
    #Calculate the dfMBD from MBD using barmat
    fr = sc.h*f0/(sc.e*barmat.tools.get_delta0(tc, bcs))

    Z = barmat.get_Zvec(temps[0]/tc, tc, vf, london0, mfp=mfp, bcs=bcs, fr=fr,
                        axis='temperature', output_depths=True,
                        boundary='diffuse')

    dfMBD = -0.5*alpha*(Z.imag - Z.imag[0])/Z.imag[0]

    dfMBd = np.array([dfMBD]*len(temps))

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
