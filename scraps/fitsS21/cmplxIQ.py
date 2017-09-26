import numpy as np
import scipy.signal as sps
import scipy.special as spc
import lmfit as lf

def cmplxIQ_fit(paramsVec, freqs, data=None, eps=None, **kwargs):
    """Return complex S21 resonance model or, if data is specified, a residual.

    Parameters
    ----------
    params : list-like
        A an ``lmfit.Parameters`` object containing (df, f0, qc, qi, gain0, gain1, gain2, pgain0, pgain1, pgain2)
    freqs : list-like
        A list of frequency points at which the model is calculated
    data : list-like (optional)
        A list of complex data in the form I + Q where I and Q are both lists of data and
        ``len(I) == len(Q) == len(freqs)``. If data is not passed, then the return value is the model
        calculated at each frequency point.
    eps : list-like (optional)
        A list of errors, one for each point in data.
    kwargs : dict (optional)
        Currently no keyword arguments are accepted.

    Returns
    -------
    model or (model-data) : ``numpy.array``
        If data is specified, the return is the residuals. If not, the return is the model
        values calculated at the frequency points. The returned array is in the form
        ``I + Q`` or ``residualI + residualQ``.

    """
    #Check if the paramsVec looks like a lmfit params object. If so, unpack to list
    if hasattr(paramsVec, 'valuesdict'):
        paramsDict = paramsVec.valuesdict()
        paramsVec = [value for value in paramsDict.values()]

    #intrinsic resonator parameters
    df = paramsVec[0] #frequency shift due to mismatched impedances
    f0 = paramsVec[1] #resonant frequency
    qc = paramsVec[2] #coupling Q
    qi = paramsVec[3] #internal Q

    #0th, 1st, and 2nd terms in a taylor series to handle magnitude gain different than 1
    gain0 = paramsVec[4]
    gain1 = paramsVec[5]
    gain2 = paramsVec[6]

    #0th and 1st terms in a taylor series to handle phase gain different than 1
    pgain0 = paramsVec[7]
    pgain1 = paramsVec[8]

    #Need to add an if statement here to keep compatibility with older version
    #Will be deprecated in the future
    if len(paramsVec) == 12:
        pgain2 = paramsVec[9]

        #Voltage offset at mixer output. Not needed for VNA
        Ioffset = paramsVec[10]
        Qoffset = paramsVec[11]
    else:
        print("Warning: new model also fits for quadratic phase. Setting pgain2 = 0.")
        print("If using Resonator.do_lmfit() pass kwarg: pgain2_vary = False for legacy behavior.")
        pgain2 = 0

        #Voltage offset at mixer output. Not needed for VNA
        Ioffset = paramsVec[9]
        Qoffset = paramsVec[10]

    #Make everything referenced to the shifted, unitless, reduced frequency
    fs = f0+df
    ff = (freqs-fs)/fs

    #Except for the gain, which should reference the file midpoint
    #This is important because the baseline coefs shouldn't drift
    #around with changes in f0 due to power or temperature

    #Of course, this philosophy goes out the window if different sweeps have
    #different ranges.
    fm = freqs[int(np.round((len(freqs)-1)/2.0))]
    ffm = (freqs-fm)/fm

    #Calculate the total Q_0
    q0 = 1./(1./qi+1./qc)

    #Calculate magnitude and phase gain
    gain = gain0 + gain1*ffm+ 0.5*gain2*ffm**2
    pgain = np.exp(1j*(pgain0 + pgain1*ffm + 0.5*pgain2*ffm**2))

    #Allow for voltage offset of I and Q
    offset = Ioffset + 1j*Qoffset

    #Calculate model from params at each point in freqs
    modelCmplx = -gain*pgain*(1./qi+1j*2.0*(ff+df/fs))/(1./q0+1j*2.0*ff)+offset

    #Package complex data in 1D vector form
    modelI = np.real(modelCmplx)
    modelQ = np.imag(modelCmplx)
    model = np.concatenate((modelI, modelQ),axis=0)

    #Calculate eps from stdev of first 10 pts of data if not supplied
    if eps is None and data is not None:
        dataI, dataQ = np.split(data, 2)
        epsI = np.std(sps.detrend(dataI[0:10]))
        epsQ = np.std(sps.detrend(dataQ[0:10]))
        eps = np.concatenate((np.full_like(dataI, epsI), np.full_like(dataQ, epsQ)))

    #Return model or residual
    if data is None:
        return model
    else:
        return (model-data)/eps

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

    #Check if some other type of hardware is supplied
    hardware = kwargs.pop('hardware', 'VNA')
    assert hardware in ['VNA', 'mixer'], "Unknown hardware type! Choose 'mixer' or 'VNA'."

    #Whether or not to use a smoothing filter on the data before making guesses
    use_filter = kwargs.pop('use_filter', False)
    assert use_filter in [True, False], "Must pass boolean to 'use_filter'."

    #Window length of Savitsky-Golay filter (must be odd and >= 3)
    filter_win_length = kwargs.pop('filter_win_length', 0)

    #If no window length is supplied, defult to 1% of the data vector or 3
    if filter_win_length == 0:
        filter_win_length = int(np.round(len(res.mag)/100.0))
        if filter_win_length % 2 == 0:
            filter_win_length+=1
        if filter_win_length < 3:
            filter_win_length = 3

    assert (filter_win_length % 2 == 1) and (filter_win_length >= 3), "Filter window length must be odd and greater than 3."

    #Whether to fit a line or a parabola to the phase:
    fit_quadratic_phase = kwargs.pop('fit_quadratic_phase', False)

    #There shouldn't be any more kwargs left
    if kwargs:
        raise Exception("Unknown keyword argument supplied")

    if use_filter:
        resMag = sps.savgol_filter(res.mag, filter_win_length, 1)
        resPhase = sps.savgol_filter(res.phase, filter_win_length, 1)
        resUPhase = np.unwrap(resPhase)
    else:
        resMag = res.mag
        resPhase = res.phase
        resUPhase = res.uphase

    #Get index of last datapoint
    findex_end = len(res.freq)-1

    #Set up lmfit parameters object for fitting later

    #Detrend the mag and phase using first and last 5% of data
    findex_5pc = int(len(res.freq)*0.05)

    findex_center = int(np.round(findex_end/2))
    f_midpoint = res.freq[findex_center]

    #Set up a unitless, reduced, mipoint frequency for baselines
    ffm = lambda fx : (fx-f_midpoint)/f_midpoint

    magEnds = np.concatenate((resMag[0:findex_5pc], resMag[-findex_5pc:-1]))
    freqEnds = ffm(np.concatenate((res.freq[0:findex_5pc], res.freq[-findex_5pc:-1])))

    #This fits a second order polynomial
    magBaseCoefs = np.polyfit(freqEnds, magEnds, 2)

    magBase = np.poly1d(magBaseCoefs)

    #Put this back in the resonator object because it is super useful!
    res.magBaseline = magBase(ffm(res.freq))

    #Store the frequency at the magnitude minimum for future use.
    #Pull out the baseline variation first

    findex_min=np.argmin(resMag-magBase(ffm(res.freq)))

    f_at_mag_min = res.freq[findex_min]

    #These points are useful for later code, so add them to the resonator object
    res.fmin = f_at_mag_min
    res.argfmin = findex_min

    #Update best guess with minimum
    f0_guess = f_at_mag_min

    #Update: now calculating against file midpoint
    #This makes sense because you don't want the baseline changing
    #as f0 shifts around with temperature and power

    #Remove any linear variation from the phase (caused by electrical delay)
    phaseEnds = np.concatenate((resUPhase[0:findex_5pc], resUPhase[-findex_5pc:-1]))
    phaseRot = resUPhase[findex_min]-resPhase[findex_min]+np.pi

    if fit_quadratic_phase:
        phase_poly_order = 2
    else:
        phase_poly_order = 1

    #This fits a second order polynomial
    phaseBaseCoefs = np.polyfit(freqEnds, phaseEnds+phaseRot, phase_poly_order)
    phaseBase = np.poly1d(phaseBaseCoefs)

    #Add to resonator object
    res.phaseBaseline = phaseBase(ffm(res.freq))

    #Set some bounds (resonant frequency should not be within 5% of file end)
    f_min = res.freq[findex_5pc]
    f_max = res.freq[findex_end-findex_5pc]

    if f_min < f0_guess < f_max:
        pass
    else:
        f0_guess = res.freq[findex_center]

    #Guess the Q values:
    #1/Q0 = 1/Qc + 1/Qi
    #Q0 = f0/fwhm bandwidth
    #Q0/Qi = min(mag)/max(mag)
    magMax = res.magBaseline[findex_min]
    magMin = resMag[findex_min]

    fwhm = np.sqrt((magMax**2 + magMin**2)/2.)
    fwhm_mask = resMag< fwhm
    bandwidth = res.freq[fwhm_mask][-1]-res.freq[fwhm_mask][0]
    q0_guess = f0_guess/bandwidth

    qi_guess = q0_guess*magMax/magMin

    qc_guess = 1./(1./q0_guess-1./qi_guess)

    #Create a lmfit parameters dictionary for later fitting
    #Set up assymetric lorentzian parameters (Name, starting value, range, vary, etc):
    params = lf.Parameters()
    params.add('df', value = 0, vary=True)
    params.add('f0', value = f0_guess, min = f_min, max = f_max, vary=True)
    params.add('qc', value = qc_guess, min = 1, max = 10**8 ,vary=True)
    params.add('qi', value = qi_guess, min = 1, max = 10**8, vary=True)

    #Allow for quadratic gain variation
    params.add('gain0', value = magBaseCoefs[2], min = 0, vary=True)
    params.add('gain1', value = magBaseCoefs[1], vary=True)
    params.add('gain2', value = magBaseCoefs[0], vary=True)

    #Allow for linear phase variation
    params.add('pgain0', value = phaseBaseCoefs[phase_poly_order], vary=True)
    params.add('pgain1', value = phaseBaseCoefs[phase_poly_order-1], vary=True)

    if fit_quadratic_phase:
        params.add('pgain2', value = phaseBaseCoefs[0], vary=True)
    else:
        params.add('pgain2', value = 0, vary=False)


    #Add in complex offset (should not be necessary on a VNA, but might be needed for a mixer)
    if hardware == 'VNA':
        params.add('Ioffset', value = 0, vary=False)
        params.add('Qoffset', value = 0, vary=False)
    elif hardware == 'mixer':
        params.add('Ioffset', value = 0, vary=True)
        params.add('Qoffset', value = 0, vary=True)

    return params
