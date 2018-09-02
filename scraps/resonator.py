import numpy as np
import lmfit as lf
import glob
import scipy.signal as sps
import pandas as pd

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
        self.logmag = 20*np.log10(self.mag) #Units are dB (20 because V->Pwr)

        #Find the frequency at magnitude minimum (this can, and should, be
        #overwritten by a custom params function)
        self.fmin = self.freq[np.argmin(self.mag)]

        #Whether or not params has been initialized
        self.params = None
        self.hasParams = False

        #These won't exist until the lmfit method is called
        self.lmfit_result = None

        #These are scheduled for deprecation. They will eventually live in the lmfit_result dictionary
        self.hasFit = False
        self.residualI = None
        self.residualQ = None
        self.resultI = None
        self.resultQ = None
        self.resultMag = None
        self.resultPhase = None

        #These won't exist until the emcee method is called
        self.emcee_result = None

        #These are scheduled for deprecation. They will eventually live in the lmfit_result dictionary
        self.hasChain = False
        self.mle_vals = None
        self.mle_labels = None

    def to_disk(self):
        """To be implemented: dumps resonator to disk as various file types. Default will be netcdf4"""
        pass

    def from_disk(self):
        """To be implemented: load resonator object from disk."""
        pass

    def to_json(self):
        """To be implemented: serialize resonator as a JSON string"""
        pass

    def from_json(self):
        """To be implemented: create rsonator from JSON string"""
        pass

    #TODO: Implement the following for handling pickling:

    #def __getstate__(self):
    #   pass

    #def __setstate__(self):
    #   pass


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

    def do_lmfit(self, fitFn, label='default', fit_type='IQ', **kwargs):
        r"""Run lmfit on an existing resonator object and update the results.

        Parameters
        ----------
        fitFn : function
            fitFn must have the signature fitFn(params, res, residual, **kwargs).
            If residual == True, fitFn must return a 1D list-like object of
            residuals with form [I residual, Q residual] where [A, B] means
            concatenate. Otherwise it must return the model data in the same form.

        label: string
            A label to use as a key when storing results from the fit to the
            lmfit_results dict.

        fit_type: string
            Indicates the type of fit to be run. For some types of fits, certain
            quantities will automatically be calculated and added to the resonator
            object. For instance, 'IQ' will cause the magnitude, phase, I, and Q
            as well as associated residuals to be calculated.

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
            for key, val in kwargs.items():
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

        # #Make complex vectors of the form cData = [reData, imData]
        # cmplxData = np.concatenate((self.I, self.Q), axis=0)

        # if (self.sigmaI is not None) and (self.sigmaQ is not None):
        #     cmplxSigma = np.concatenate((self.sigmaI, self.sigmaQ), axis=0)
        # else:
        #     cmplxSigma = None

        # #Create a lmfit minimizer object
        # minObj = lf.Minimizer(fitFn, self.params, fcn_args=(self.freq, cmplxData, cmplxSigma))

        #Create a lmfit minimizer object
        minObj = lf.Minimizer(fitFn, self.params, fcn_args=(self, True))

        lmfit_result = minObj.minimize(method = 'leastsq')

        #Call the lmfit minimizer method and minimize the residual
        if self.lmfit_result is None:
            self.lmfit_result = {}

        self.lmfit_result[label] = {}
        self.lmfit_result[label]['fit_type'] = fit_type
        self.lmfit_result[label]['result'] = lmfit_result
        self.lmfit_result[label]['values'] = np.asarray([val.value for key, val in lmfit_result.params.items() if val.vary is True])
        self.lmfit_result[label]['labels'] = [key for key, val in lmfit_result.params.items() if val.vary is True]

        #NOTE: These are likely to be deprecated
        if label == 'default':
            self.lmfit_vals = self.lmfit_result[label]['values']
            self.lmfit_labels = self.lmfit_result[label]['labels']


        #Set the hasFit flag NOTE:(scheduled for deprecation)
        self.hasFit = True

        #NOTE: This whole block may be deprecated
        if (fit_type == 'IQ') and (label == 'default'):
            #Add the data back to the final minimized residual to get the final fit
            #Also calculate all relevant curves
            cmplxResult = fitFn(self.lmfit_result[label]['result'].params, self, residual=False)
            cmplxResidual = self.lmfit_result[label]['result'].residual

            #Split the complex data back up into real and imaginary parts
            residualI, residualQ = np.split(cmplxResidual, 2)
            resultI, resultQ = np.split(cmplxResult, 2)

            resultMag = np.abs(resultI + 1j*resultQ)
            resultPhase = np.arctan2(resultQ,resultI)

            #Add some results back to the resonator object
            self.residualI = residualI
            self.residualQ = residualQ
            self.resultI = resultI
            self.resultQ = resultQ
            self.resultMag = resultMag
            self.resultPhase = resultPhase

    def torch_lmfit(self, label='default'):
        r"""Reset all the lmfit attributes to ``None`` and set ``hasFit = False``.

        Parameters
        ----------
        label : string (optional)
            Choose which fit to kill off.
        """

        if self.lmfit_result is not None:
            if label in self.lmfit_result.keys():
                deleted_fit = self.lmfit_result.pop(label)

                if label == 'default':
                    self.lmfit_vals = None
                    self.lmfit_labels = None
            
                if (deleted_fit['fit_type'] == 'IQ') and label == 'default':
                    
                    self.residualI = None
                    self.residualQ = None
                    self.resultI = None
                    self.resultQ = None
                    self.resultMag = None
                    self.resultPhase = None


                if len(self.lmfit_result.keys()) == 0:
                    self.lmfit_result = None
                    self.hasFit = False


    def do_emcee(self, fitFn, label='default', **kwargs):
        r"""Run the Monte-Carlo Markov Chain routine to generate samples for
        each parameter given a model.

        Parameters
        ----------
        fitFn : function
            fitFn must have the signature fitFn(params, res, residual, **kwargs).
            If residual == True, fitFn must return a 1D list-like object of
            residuals with form [I residual, Q residual] where [A, B] means
            concatenate. Otherwise it must return the model data in the same form.

        label : string (optional)
            A label to assign to the fit results. This will be the dict key they
            are stored under in the emcee_results dict. Also, if label matches a
            label in lmfit_results, then that params object will be used to seed
            the emcee fit.

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

        #Create a lmfit minimizer object
        if self.hasFit:
            if self.lmfit_result is not None:
                if label in self.lmfit_result.keys():
                    emcee_params = self.lmfit_result[label]['result'].params
        else:
            assert self.hasParams == True, "Must load params before running emcee."
            emcee_params = self.params

        minObj = lf.Minimizer(fitFn, emcee_params, fcn_args=(self, True))

        #Run the emcee and add the result in
        emcee_result = minObj.emcee(**kwargs)

        if self.emcee_result is None:
            self.emcee_result = {}

        self.emcee_result[label] = {}
        self.emcee_result[label]['result'] = emcee_result
        
        #Get the emcee 50th percentile data and uncertainties at 16th and 84th percentiles
        emcee_vals = np.asarray([np.percentile(emcee_result.flatchain[key], 50) for key in emcee_result.flatchain.keys()])
        err_plus = np.asarray([np.percentile(emcee_result.flatchain[key], 84) for key in emcee_result.flatchain.keys()])
        err_minus = np.asarray([np.percentile(emcee_result.flatchain[key], 16) for key in emcee_result.flatchain.keys()])

        #Pack these values into the fit storage dict
        self.emcee_result[label]['values'] = emcee_vals

        #Make a list of tuples that are (+err, -err) for each paramter
        self.emcee_result[label]['emcee_sigmas'] = list(zip(err_plus-emcee_vals, emcee_vals-err_minus))

        #It is also useful to have easy access to the maximum-liklihood estimates
        self.emcee_result[label]['mle_vals'] = emcee_result.flatchain.iloc[np.argmax(emcee_result.lnprob)]

        #This is useful because only varying parameters have mle vals
        self.emcee_result[label]['mle_labels'] = self.emcee_result[label]['mle_vals'].keys()


        if label == 'default':
            self.emcee_vals = self.emcee_result[label]['values']

            #Make a list of tuples that are (+err, -err) for each paramter
            self.emcee_sigmas = self.emcee_result[label]['emcee_sigmas']

            #It is also useful to have easy access to the maximum-liklihood estimates
            self.mle_vals = self.emcee_result[label]['mle_vals']

            #This is useful because only varying parameters have mle vals
            self.mle_labels = self.emcee_result[label]['mle_labels']

            #This is also nice to have explicitly for passing to triangle-plotting routines
            self.chain = emcee_result.flatchain.copy()
        
        self.hasChain = True

    def burn_flatchain(self, num_samples=0, label='default'):
        r"""Burns off num_samples samples from each of the chains and then reflattens. Recalculates all
        statistical quantities associated with the emcee run and saves them under the original
        label, but with the suffix '_burn' appended to the various keys. Does not modify original chain."""
        
        flatchain_with_burn = pd.DataFrame()
        chains = self.emcee_result[label]['result'].chain
        
        for ix, chain in enumerate(chains.T):
            flatchain_with_burn[self.emcee_result[label]['mle_labels'][ix]] = chain[num_samples:].flat

        #Get the emcee 50th percentile data and uncertainties at 16th and 84th percentiles
        emcee_vals = np.asarray([np.percentile(flatchain_with_burn[key], 50) for key in flatchain_with_burn.keys()])
        err_plus = np.asarray([np.percentile(flatchain_with_burn[key], 84) for key in flatchain_with_burn.keys()])
        err_minus = np.asarray([np.percentile(flatchain_with_burn[key], 16) for key in flatchain_with_burn.keys()])

        #Make a list of tuples that are (+err, -err) for each paramter
        emcee_sigmas = list(zip(err_plus-emcee_vals, emcee_vals-err_minus))

        #Pack these values into the fit storage dict with suffix _burn
        self.emcee_result[label]['values_burn'] = emcee_vals

        #Make a list of tuples that are (+err, -err) for each paramter
        self.emcee_result[label]['emcee_sigmas_burn'] = list(zip(err_plus-emcee_vals, emcee_vals-err_minus))

        #TODO: Implement this!
        #It is also useful to have easy access to the maximum-liklihood estimates
        #self.emcee_result[label]['mle_vals_burn'] = flatchain_with_burn.iloc[np.argmax(emcee_result.lnprob)]

        #Add the burned flatchain in its own key
        self.emcee_result[label]['flatchain_burn'] = flatchain_with_burn

    def torch_emcee(self, label='default'):
        r"""Set the emcee-related attributes to ``None`` and ``hasChain = False``.
        Parameters
        ----------
        label : string (optional)
            Which fit to torch"""
        
        if self.emcee_result is not None:
            if label in self.emcee_result.keys():
                deleted_fit = self.emcee_result.pop(label)

                if label == 'default':
                    self.emcee_vals = None
                    self.emcee_sigmas = None
                    self.mle_vals = None
                    self.mle_labels = None
                    self.chain = None


                if len(self.emcee_result.keys()) == 0:
                    self.hasChain = False
                    self.emcee_result = None

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
    assert all(key in dataDict.keys() for key in expectedKeys), "Your dataDict is missing one or more keys"

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

def makeResList(fileFunc, dataPath, resName, **fileFunc_kwargs):
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

    fileFunc_kwargs : dict
        Keyword arguments to pass through to the fileFunc

    """
    #Find the files that match the resonator you care about
    fileList = glob.glob(dataPath + resName + '_*')

    #loop through files and process all the data
    fileDataDicts = []

    for f in fileList:
        fileDataDicts.append(fileFunc(f, **fileFunc_kwargs))

    #Create resonator objects from the data 
    #makeResFromData returns a tuple of (res, temp, pwr),
    #but only care about the first one
    resList = [makeResFromData(fileDataDict) for fileDataDict in fileDataDicts]

    return resList

#Index a list of resonator objects easily
def indexResList(resList, temp=None, pwr=None, **kwargs):
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
    index : int or list
        Index is the index of the Resonator in resList or a list of indices of
        all matches if only pwr or only temp is specified.

    Notes
    -----
    indexResList does not check for duplicates and will return the first match.

    """
    itemp = kwargs.pop('itemp', False)
    assert itemp in [True, False], "'itemp' must be boolean."

    assert (pwr is not None) or (temp is not None), "Must specify at least either a temp or a pwr."


    if (pwr is not None) and (temp is not None):
        for index, res in enumerate(resList):
            if itemp is True:
                if res.itemp == temp and res.pwr == pwr:
                    return index
            else:
                if np.isclose(res.temp, temp) and res.pwr == pwr:
                    return index
    elif (pwr is None):
        index = []
        for ix, res in enumerate(resList):
            if itemp is True:
                if res.itemp == temp:
                    index.append(ix)
                else:
                    if np.isclose(res.temp, temp):
                        index.append(ix)
    elif (temp is None):
        index = []
        for ix, res in enumerate(resList):
            if res.pwr == pwr:
                index.append(ix)
        return index

    return None

def print_resList(resList):
    """Print all the temperatures and powers in a table-like form"""
    #Get all possible powers
    pwrs = np.unique([res.pwr for res in resList])

    #This will hold a list of temps at each power
    tlists = []
    max_len = 0

    #Populate the lists of temps for each power
    for p in pwrs:
        tlist = [res.temp for res in resList if res.pwr == p]
        tlist.sort()
        tlists.append(tlist)
        if len(tlist) > max_len:
            max_len = len(tlist)

    for ix, tlist in enumerate(tlists):
        pad = max_len - len(tlist)
        tlist = tlist + pad*['NaN']
        tlists[ix] = tlist

    block = zip(*tlists)

    print(repr(list(pwrs)).replace(',', ',\t'))
    for b in block:
        print(repr(b).replace(',', ',\t'))



def block_check_resList(resList, sdev=0.005, prune=False, verbose=True):
    """Helper tool for preparing a resList with missing data for resSweep"""
    #Get all possible powers
    pwrs = np.unique([res.pwr for res in resList])

    #This will hold a list of temps at each power
    tlists = []

    #Populate the lists of temps for each power
    for p in pwrs:
        tlist = [res.temp for res in resList if res.pwr == p]
        tlist.sort()
        tlists.append(tlist)

    #Calculate the lengths and find the shortest one
    lens = [len(tl) for tl in tlists]
    shortest = min(lens)

    if all(el == shortest for el in lens) and verbose:
        print('All lists have same length.')
    else:
        print('Lengths for each set of powers: ',list(zip(pwrs,lens)))

    #Zip the lists into tuples and take the standard deviation
    #of each tuple. All the elements in each tuple should be
    #nominally the same, so the stdev should be small unless
    #one of the elements doesn't match. Return the first
    #instance of the stdev being too high
    block = list(zip(*tlists))
    bad_ix = np.argmax([np.std(x) > sdev for x in block])

    #If the first row is returned, everything could be ok. Check first row.
    if bad_ix == 0:
        if np.std(block[0]) < sdev:
            bad_ix = -1

    if verbose:
        print("Bad index: ", bad_ix)

    if bad_ix >= 0:

        if verbose:
            for i in np.arange(-2,3):
                if (bad_ix+i < len(block)) and (bad_ix+i >= 0):
                    print(repr(block[bad_ix+i]).replace(',', ',\t'))
                    block_ixs = []
                    for block_ix, block_temp in enumerate(block[bad_ix+i]):
                        block_ixs.append(indexResList(resList, block_temp, pwrs[block_ix]))
                    print(repr(block_ixs).replace(',', ',\t'))


            #The longer list is where the extra file is most likely
            #so return the temp, power, and resList index of the
            #suspect.
            for i, x in enumerate(block[bad_ix]):
                if np.abs(x-np.mean(block[bad_ix])) > np.std(block[bad_ix]):

                    tl = tlists[i]
                    t = tl[bad_ix]
                    p = pwrs[i]
                    res_ix = indexResList(resList, t, p)
                    if verbose:
                        print('T=',t, 'P=',p, 'Res index=',res_ix)
                    if prune:
                        resList.pop(res_ix)


