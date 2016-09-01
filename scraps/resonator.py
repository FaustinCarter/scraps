import numpy as np
import lmfit as lf
import glob
import scipy.signal as sps

class Resonator(object):
    r"""Fit an S21 measurement of a hanger (or notch) type resonator.

    Parameters
    ----------
    name : string
        The resonator name. Does not have to be unique, but each physical
        resonator in the experiment should have a unique name to avoid
        confusion when using some of the other tools in scraps.

    temp : float
        The temperature in (K) that the S21 measurement was taken at.

    pwr : float
        The power (in dBm) at which the resonator was measured.

    freq : array-like[nDataPoints]
        The frequency points at which the S21 scan was measured.

    I : array-like[nDataPoints]
        The in-phase (or real part) of the complex S21 measurement. Units
        are typically volts, and I should be specified in linear units (as
        opposed to dB).

    Q : array-like[nDataPoints]
        The out-of-phase (or imaginary part) of the complex S21 measurement.
        Units are typically volts, and I should be specified in linear units
        (as opposed to dB).

    sigmaI : array-like[nDataPoints] (optional)
        An array of uncertaintly values for each data point in `I`. Default
        is ``None``.

    sigmaQ : array-like[nDataPoints] (optional)
        An array of uncertaintly values for each data point in `Q`. Default
        is ``None``.

    The following attributes are automatically calculated and added during
    initialization.

    Attributes
    ----------
    name : string
        The resonator name passed at initialization.

    temp : float
        The temperature passed at initialization.

    pwr : float
        The power passed at initialization.

    freq : array-like[nDataPoints]
        The frequency points passed at initialization.

    I : array-like[nDataPoints]
        The I data points passed at initialization.

    Q : array-like[nDataPoints]
        The Q data points passed at initialization.

    sigmaI : array-like[nDataPoints]
        The sigmaI values passed at initialization.

    sigmaQ : array-like[nDataPoints]
        The sigmaQ values passed at initialization.

    S21 : array-like[nDataPoints]
        The complex transmission ``S21 = I + 1j*Q``.

    phase : array-like[nDataPoints]
        The raw phase ``phase = np.arctan2(Q, I)``.

    uphase : array-like[nDataPoints]
        The unwrapped phase is equivalent to the phase, but with jumps of 2 Pi
        removed.

    mag : array-like[nDataPoints]
        The magnitude ``mag = np.abs(S21)`` or, equivalently ``mag =
        np.sqrt(I**2 + Q**2)``.

    hasFit : bool
        Indicates whether or not ``Resonator.do_lmfit`` method has been called.

    lmfit_result : ``lmfit.Result`` object
        The result object created by ``lmfit`` containing all the fit
        information. Some of the fit information is futher extracted for
        convenience in the following Attributes. For an exhaustive list of the
        attributes of lmfit_result see the docs for ``lmfit``. The most useful
        attribute of this object is ``lmfit_result.params``, which contains the
        best-fit parameter values.

    residualI : array-like[nDataPoints]
        The residual of the fit model against the `I` data, wieghted by the
        uncertainties.

    residualQ : array-like[nDataPoints]
        The residual of the fit model against the `Q` data, wieghted by the
        uncertainties.

    resultI : array-like[nDataPoints]
        The best ``lmfit`` fit result to the fit model for `I`.

    resultQ : array-like[nDataPoints]
        The best ``lmfit`` fit result to the fit model for `Q`.

    resultMag : array-like[nDataPoints]
        ``resultMag = np.abs(resultI + 1j*resultQ)``

    resultPhase : array-like[nDataPoints]
        ``resultPhase = np.arctan2(resultQ/resultI)``

    emcee_result : ``lmfit.Result`` object
        This object is nearly identical to the `lmfit_result` object, but also
        contains the maximum-liklihood values for the *varying* parameters of
        the fit model as well as the `chains` returned by ``emcee``. The most
        important attribute is probably ``emcee_result.flatchain``, which can be
        passed directly to ``pygtc`` or ``corner`` to make a really nice
        GTC/Triangle/Corner plot. For an exhaustive list of the attributes of
        emcee_result see the docs for ``lmfit``, specifically the section
        involving the ``lmfit`` implementation of ``emcee``.

    mle_vals : list of float
        The maximum-liklihood estimate values of the *varying* parameter in the
        fit model as calculated by ``emcee``. Unpacked here for convenience from
        ``emcee_result.params``.

    mle_labels: list of string
        The parameter names of the values in `mle_vals`. Provided here for easy
        passing to  ``pygtc`` or ``corner``.

    magBaseLine : array-like[nDataPoints]
        The best initial guess of the baseline of the magnitude. Calculated by
        fitting a quadratic polynomial to the beginning and end of the magnitdue
        vs frequency curve.

    phaseBaseLine: array-like[nDataPoints]
        The best initial guess of the baseline of the phase. Calculated by
        fitting a line to the beginning and end of the phase vs frequency curve.
        This is equivalent to calculating the electrical delay in the
        measurement lines.

    params : ``lmfit.Parameters`` object
        The initial parameter guesses for fitting the `S21` data. See ``lmfit``
        documentation for a complete overview. To get the parameter names, call
        ``params.keys()``. Default is ``None``. Initialize params by calling
        ``Resonator.load_params``. Delete params with
        ``Resonator.torch_params``.

    """


    #Do some initialization
    def __init__(self, name, temp, pwr, freq, I, Q, sigmaI = None, sigmaQ = None):
        r"""Initializes a resonator object by calculating magnitude, phase, and
        a bunch of fit parameters for a hanger (or notch) type S21 measurement.

        """
        self.name = name
        self.temp = temp
        self.pwr = pwr
        self.freq = np.asarray(freq)
        self.I = np.asarray(I)
        self.Q = np.asarray(Q)
        self.sigmaI = np.asarray(sigmaI) if sigmaI is not None else None
        self.sigmaQ = np.asarray(sigmaQ) if sigmaQ is not None else None
        self.S21 = I + 1j*Q
        self.phase = np.arctan2(Q,I) #use arctan2 because it is quadrant-aware
        self.uphase = np.unwrap(self.phase) #Unwrap the 2pi phase jumps
        self.mag = np.abs(self.S21) #Units are volts.
        self.logmag = 20*np.log(self.mag) #Units are dB (20 because V->Pwr)

        #Find the frequency at magnitude minimum (this can, and should, be
        #overwritten by a custom params function)
        self.fmin = self.freq[np.argmin(self.mag)]

        #Whether or not params has been initialized
        self.params = None
        self.hasParams = False

        #Whether or not a lmfit has been run
        self.hasFit = False

        #These won't exist until the lmfit method is called
        self.lmfit_result = None
        self.residualI = None
        self.residualQ = None
        self.resultI = None
        self.resultQ = None
        self.resultMag = None
        self.resultPhase = None

        #Whether or not an emcee has been run
        self.hasChain = False

        #These won't exist until the emcee method is called
        self.emcee_result = None
        self.mle_vals = None
        self.mle_labels = None


    def load_params(self, paramsFn, **kwargs):
        """Load up a lmfit Parameters object for a custom fit function.

        Parameters
        ----------
        paramsFn : method
            The paramsFn method should return a ``lmfit.Paramters`` object. This
            object will be passed to the fit method when ``do_lmfit`` or
            ``do_emcee`` is
            called.

        kwargs : dict
            A dictionary of keyword arguments to pass to paramsFn.

        """
        params = paramsFn(self, **kwargs)
        self.params = params
        self.hasParams = True

    def torch_params(self):
        """Reset params attribute to ``None``."""
        self.params = None
        self.hasParams = False

    def do_lmfit(self, fitFn, **kwargs):
        r"""Run lmfit on an existing resonator object and update the results.

        Parameters
        ----------
        fitFn : function
            fitFn must have the signature ([A,B] means concatenate lists A and
            B): fitFn(params, [Idata, Qdata], [I error, Q error]) and must
            return a 1D list-like object of residuals with form [I residual, Q
            residual].

        kwargs : optional keywords
            Use this to override any of the lmfit parameter initial guesses or
            toggle whether the paramter varys. Example: ``qi=1e6`` is equivalent
            to calling ``Resonator.params['qi'].value = 1e6``. Example:
            ``qi_vary=False`` will set ``Resonator.params['qi'].vary = False``.
            Any parameter name can be used in this way.
        """
        assert self.hasParams == True, "Must load params before running a fit."

        #Update any of the default Parameter guesses
        if kwargs is not None:
            for key, val in kwargs.iteritems():
                #Allow for turning on and off parameter variation
                if '_vary' in key:
                    key = key.split('_')[0]
                    if key in self.params.keys():
                        if (val is True) or (val is False):
                            self.params[key].vary = val
                elif key in self.params.keys():
                    self.params[key].value = val
                else:
                    raise ValueError("Unknown key: "+key)

        #Make complex vectors of the form cData = [reData, imData]
        cmplxData = np.concatenate((self.I, self.Q), axis=0)

        if (self.sigmaI is not None) and (self.sigmaQ is not None):
            cmplxSigma = np.concatenate((self.sigmaI, self.sigmaQ), axis=0)
        else:
            cmplxSigma = None

        #Create a lmfit minimizer object
        minObj = lf.Minimizer(fitFn, self.params, fcn_args=(self.freq, cmplxData, cmplxSigma))

        #Call the lmfit minimizer method and minimize the residual
        lmfit_result = minObj.minimize(method = 'leastsq')

        #Add the data back to the final minimized residual to get the final fit
        #Also calculate all relevant curves
        cmplxResult = fitFn(lmfit_result.params, self.freq)
        cmplxResidual = lmfit_result.residual

        #Split the complex data back up into real and imaginary parts
        residualI, residualQ = np.split(cmplxResidual, 2)
        resultI, resultQ = np.split(cmplxResult, 2)

        resultMag = np.abs(resultI + 1j*resultQ)
        resultPhase = np.arctan2(resultQ,resultI)

        #Set the hasFit flag
        self.hasFit = True

        #Add some results back to the resonator object
        self.lmfit_result = lmfit_result
        self.residualI = residualI
        self.residualQ = residualQ
        self.resultI = resultI
        self.resultQ = resultQ
        self.resultMag = resultMag
        self.resultPhase = resultPhase

        #It's useful to have a list of the best fits for the varying parameters
        self.lmfit_vals = np.asarray([val.value for key, val in lmfit_result.params.iteritems() if val.vary is True])
        self.lmfit_labels = [key for key, val in lmfit_result.params.iteritems() if val.vary is True]

    def torch_lmfit(self):
        r"""Reset all the lmfit attributes to ``None`` and set ``hasFit = False``.
        """
        #Delete all the lmfit results
        self.hasFit = False
        self.lmfit_result = None
        self.residualI = None
        self.residualQ = None
        self.resultI = None
        self.resultQ = None
        self.resultMag = None
        self.resultPhase = None
        self.lmfit_vals = None
        self.lmfit_labels = None


    def do_emcee(self, fitFn, **kwargs):
        r"""Run the Monte-Carlo Markov Chain routine to generate samples for
        each parameter given a model.

        Parameters
        ----------
        fitFn : function
            fitFn must have the signature ([A,B] means concatenate lists A and
            B): fitFn(params, [Idata, Qdata], [I error, Q error]) and must
            return a 1D list-like object of residuals with form [I residual, Q
            residual].

        kwargs : optional keyword arguments
            These are passed through to the ``lmfit.Minimizer.emcee`` method.
            See the ``lmfit`` documentation for more information.

        """
        #Should do the following (have not implemented any of this yet):
        #Pack MLE values into their own params object by adding back in non-varying Parameters
        #Should consider the ability to filter results for better parameter estimations
        #Probably should make a nice easy output to the corner Package
        #Smart way to add in error parameter as nuisance without breaking auto-guessing

        #minimizerObj.emcee already updates parameters object to result
        #This means can call res.emcee_result.params to get results

        assert self.hasParams == True, "Must load params before running emcee."

        cmplxData = np.concatenate((self.I, self.Q), axis=0)

        if (self.sigmaI is not None) and (self.sigmaQ is not None):
            cmplxSigma = np.concatenate((self.sigmaI, self.sigmaQ), axis=0)
        else:
            cmplxSigma = None

        #Create a lmfit minimizer object
        if self.hasFit:
            emcee_params = self.lmfit_result.params
        else:
            emcee_params = self.params

        minObj = lf.Minimizer(fitFn, emcee_params, fcn_args=(self.freq, cmplxData, cmplxSigma))

        #Run the emcee and add the result in
        emcee_result = minObj.emcee(**kwargs)
        self.emcee_result = emcee_result

        #It is useful to have easy access to the maximum-liklihood estimates
        self.mle_vals = np.asarray([val.value for key, val in emcee_result.params.iteritems() if val.vary is True])
        self.mle_labels = [key for key, val in emcee_result.params.iteritems() if val.vary is True]

        #This is also nice to have explicitly for passing to triangle-plotting routines
        self.chain = emcee_result.flatchain.copy()
        self.hasChain = True

    def torch_emcee(self):
        r"""Set the emcee-related attributes to ``None`` and ``hasChain = False``."""
        self.hasChain = False
        self.emcee_result = None
        self.mle_vals = None
        self.mle_labels = None
        self.chain = None


#This creates a resonator object from a data dictionary. Optionally performs a fit, and
#adds the fit data back in to the resonator object
def makeResFromData(dataDict, paramsFn = None, fitFn = None, fitFn_kwargs=None, paramsFn_kwargs=None):
    """Create a Resonator object from a data dictionary.

    Parameters
    ----------
    dataDict : dict
        Must have the following keys: 'I', 'Q', 'temp', 'pwr', 'freq', 'name'.
        Optional keys are: 'sigmaI', 'sigmaQ'

    paramsFn : function (optional)
        A function that initializes and returns an lmfit parameters object for
        passing to fitFn.

    fitFn : function (optional)
        If a fit function is passed, an lmfit minimization will be done
        automatically.

    fitFn_kwargs : dict (optional)
        A dict of keyword arguments passed to fitFn.

    paramsFn_kwargs: dict (optional)
        A dict of keyword arguments passed to paramsFn.

    Returns
    -------
    res : ``Resonator`` object or ``None``
        A Resonator object or ``None`` if there is an error loading the data.

    """
    if fitFn is not None:
        assert paramsFn is not None, "Cannot pass a fitFn without also passing a paramsFn"

    #Check dataDict for validity
    expectedKeys = ['name', 'temp', 'pwr', 'freq', 'I', 'Q']
    assert all(key in dataDict for key in expectedKeys), "Your dataDict is missing one or more keys"

    resName = dataDict['name']
    temp = dataDict['temp']
    pwr = dataDict['pwr']
    freqData = dataDict['freq']
    IData = dataDict['I']
    QData = dataDict['Q']

    #Process the optional keys
    if 'sigmaI' in dataDict.keys():
        sigmaI = dataDict['sigmaI']
    else:
        sigmaI = None

    if 'sigmaQ' in dataDict.keys():
        sigmaQ = dataDict['sigmaQ']
    else:
        sigmaQ = None

    #create Resonator object
    res = Resonator(resName, temp, pwr, freqData, IData, QData, sigmaI, sigmaQ)

    #Process the fit parameters
    if paramsFn is not None:
        if paramsFn_kwargs is not None:
            res.load_params(paramsFn, **paramsFun_kwargs)
        else:
            res.load_params(paramsFn)

    #Run a fit on the resonator if a fit function is specified
    if fitFn is not None:
        if fitFn_kwargs is not None:
            res.do_lmfit(fitFn, **fitFn_kwargs)
        else:
            res.do_lmfit(fitFn)

    #Return resonator object
    return res

def makeResList(fileFunc, dataPath, resName):
    """Create a list of resonator objects from a directory of dataDict

    Parameters
    ----------
    fileFunc : function
        A function that converts a single data file into a dictionary. The
        resulting dictionary must have the following keys: 'I', 'Q', 'temp',
        'pwr', 'freq', 'name', and may have the following ptional keys:
        'sigmaI', 'sigmaQ'

    dataPath : string
        Path to the directory containing the data files that will be processed
        by fileFunc.

    resName : string
        The name of your resonator. This can be anything, but it is useful to
        use the same name for every data file that comes from the same physical
        resonator.

    """
    #Find the files that match the resonator you care about
    fileList = glob.glob(dataPath + '*' + resName + '_*' + '*')

    #loop through files and process all the data
    fileDataDicts = map(fileFunc, fileList)

    #Create resonator objects from the data
    #makeResFromData returns a tuple of (res, temp, pwr),
    #but only care about the first one
    resList = [makeResFromData(fileDataDict) for fileDataDict in fileDataDicts]

    return resList

#Index a list of resonator objects easily
def indexResList(resList, temp, pwr, **kwargs):
    """Index resList by temp and pwr.

    Parameters
    ----------
    resList : list-like
        resList is a list of ``scraps.Resonator`` objects
    temp : numeric
        The temperature of a single Resonator object.
    pwr : int
        The power of a single Resonator object

    itemp : boolean (optional)
        Switch to determine whether lookup uses temp or itemp (rounded value of
        temp). Default is ``False``.

    Returns
    -------
    index : int
        Index is the index of the Resonator in resList

    Notes
    -----
    indexResList does not check for duplicates and will return the first match.

    """
    itemp = kwargs.pop('itemp', False)
    assert itemp in [True, False], "'itemp' must be boolean."


    for index, res in enumerate(resList):
        if itemp is True:
            if res.itemp == temp and res.pwr == pwr:
                return index
        else:
            if res.temp == temp and res.pwr == pwr:
                return index

    return None
