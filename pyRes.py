import numpy as np
import pandas as pd
import lmfit as lf
import scipy.signal as sps

class Resonator(object):
    """Initializes a resonator object by calculating magnitude, phase, and a variety of parameters.

    Return value: None

    Arguments:
    name -- a string containing the resonator name (does not have to be unique)
    temp -- a float indicating the temperature in (K) of the data
    I,Q -- numpy array or list of in-phase and quadrature values (linear scale)
    sigmaI, sigmaQ -- numpy array or list of uncertainty on I and Q values or None"""

    #Do some initialization
    def __init__(self, name, temp, pwr, freq, I, Q, sigmaI = None, sigmaQ = None):
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
        self.mag = np.abs(self.S21)

        #Whether or not a fit has been run
        self.hasFit = False

        #These won't exist until the lmfit method is called
        self.S21result = None
        self.residualI = None
        self.residualQ = None
        self.resultI = None
        self.resultQ = None
        self.resultMag = None
        self.resultPhase = None

        #If errorbars are not supplied for I and Q, then estimate them based on
        #the tail of the power-spectral densities

        if sigmaI is None:
            f, psdI = sps.welch(self.I)
            epsI = np.mean(np.sqrt(psdI[-150:]))
            self.sigmaI = np.full_like(I, epsI)

        if sigmaQ is None:
            f, psdQ = sps.welch(self.Q)
            epsQ = np.mean(np.sqrt(psdQ[-60:]))
            self.sigmaQ = np.full_like(Q, epsQ)

        #Get index of last datapoint
        findex_end = len(freq)-1


        #Set up lmfit parameters object for fitting later

        #Detrend the mag and phase using first and last 5% of data
        findex_5pc = int(len(freq)*0.05)

        findex_center = np.round(findex_end/2)
        f_midpoint = freq[findex_center]

        magEnds = np.concatenate((self.mag[0:findex_5pc], self.mag[-findex_5pc:-1]))
        freqEnds = np.concatenate((self.freq[0:findex_5pc], self.freq[-findex_5pc:-1]))

        #This fits a second order polynomial
        magBaseCoefs = np.polyfit(freqEnds-f_midpoint, magEnds, 2)

        magBase = np.poly1d(magBaseCoefs)

        #Store the frequency at the magnitude minimum for future use.
        #Pull out the baseline variation first

        findex_min=np.argmin(self.mag-magBase(self.freq-f_midpoint))

        f_at_mag_min = freq[findex_min]
        self.fmin = f_at_mag_min
        self.argfmin = findex_min

        #Update best guess with minimum
        f0_guess = f_at_mag_min

        #Recalculate the baseline relative to the new f0_guess
        magBaseCoefs = np.polyfit(freqEnds-f0_guess, magEnds, 2)
        magBase = np.poly1d(magBaseCoefs)
        self.baseline = magBase(self.freq-f0_guess)

        #Remove any linear variation from the phase (caused by electrical delay)
        phaseEnds = np.concatenate((self.uphase[0:findex_5pc], self.uphase[-findex_5pc:-1]))
        phaseRot = self.uphase[findex_min]-self.phase[findex_min]+np.pi

        phaseBaseCoefs = np.polyfit(freqEnds-f0_guess, phaseEnds+phaseRot, 1)

        #Set some bounds (resonant frequency should not be within 5% of file end)
        f_min = freq[findex_5pc]
        f_max = freq[findex_end-findex_5pc]

        if f_min < f0_guess < f_max:
            pass
        else:
            f0_guess = freq[findex_center]

        #Guess the Q values:
        #1/Q0 = 1/Qc + 1/Qi
        #Q0 = f0/fwhm bandwidth
        #Q0/Qi = min(mag)/max(mag)
        magMax = self.baseline[findex_min]
        magMin = self.mag[findex_min]

        fwhm = np.sqrt((magMax**2 + magMin**2)/2.)
        fwhm_mask = self.mag < fwhm
        bandwidth = self.freq[fwhm_mask][-1]-self.freq[fwhm_mask][0]
        q0_guess = f0_guess/bandwidth

        qi_guess = q0_guess*magMax/magMin

        qc_guess = 1./(1./q0_guess-1./qi_guess)

        #Create a lmfit parameters dictionary for later fitting
        #Set up assymetric lorentzian parameters (Name, starting value, range, vary, etc):
        self.params = lf.Parameters()
        self.params.add('df', value = 0, vary=True)
        self.params.add('f0', value = f0_guess, min = f_min, max = f_max, vary=True)
        self.params.add('qc', value = qc_guess, min = 1, max = 10**8 ,vary=True)
        self.params.add('qi', value = qi_guess, min = 1, max = 10**8, vary=True)

        #Allow for quadratic gain variation
        self.params.add('gain0', value = magBaseCoefs[2], min = 0, vary=True)
        self.params.add('gain1', value = magBaseCoefs[1], vary=True)
        self.params.add('gain2', value = magBaseCoefs[0], vary=True)

        #Allow for linear phase variation
        self.params.add('pgain0', value = phaseBaseCoefs[1], vary=True)
        self.params.add('pgain1', value = phaseBaseCoefs[0], vary=True)

        #Add in complex offset (should not be necessary on a VNA, but might be needed for a mixer)
        self.params.add('Ioffset', value = 0, vary=False)
        self.params.add('Qoffset', value = 0, vary=False)

    def lmfit(self, fitFn, **kwargs):
        """Run lmfit on an existing resonator object and update the results.

        Return value:
        No return value -- operates on resonator object in place

        Arguments:
        res -- Existing resonator object. This object will be modified.
        fitFn -- Any lmfit compatible fit function
            fitFn arguments: lmfit parameter object, [Idata, Qdata], [I error, Q error]
        kwargs -- Use this to override any of the lmfit parameter initial guesses
            example: qi=1e6 is equivalent to calling res.params['qi'].value = 1e6
            example: qi_vary=False will fix the parameter
        """

        #Update any of the default Parameter guesses
        if kwargs is not None:
            for key, val in kwargs.iteritems():
                #Allow for turning on and off parameter variation
                if '_vary' in key:
                    key = key.split('_')[0]
                    if key in self.params.keys():
                        if (val is True) or (val is False):
                            self.params[key].vary = val
                if key in self.params.keys():
                    self.params[key].value = val

                #This was for some debugging, no longer used
                elif key is 'addpi' and val is True:
                    self.params['pgain0'].value += np.pi

        #Make complex vectors of the form cData = [reData, imData]
        cmplxData = np.concatenate((self.I, self.Q), axis=0)

        cmplxSigma = np.concatenate((self.sigmaI, self.sigmaQ), axis=0)

        #Create a lmfit minimizer object
        minObj = lf.Minimizer(fitFn, self.params, fcn_args=(self.freq, cmplxData, cmplxSigma))

        #Call the lmfit minimizer method and minimize the residual
        S21result = minObj.minimize(method = 'leastsq')

        #Add the data back to the final minimized residual to get the final fit
        #Also calculate all relevant curves
        cmplxResult = S21result.residual*cmplxSigma+cmplxData
        cmplxResidual = S21result.residual

        #Split the complex data back up into real and imaginary parts
        residualI, residualQ = np.split(cmplxResidual, 2)
        resultI, resultQ = np.split(cmplxResult, 2)

        resultMag = np.abs(resultI + 1j*resultQ)
        resultPhase = np.arctan2(resultQ,resultI)

        #Set the hasFit flag
        self.hasFit = True

        #Add some results back to the resonator object
        self.S21result = S21result
        self.residualI = residualI
        self.residualQ = residualQ
        self.resultI = resultI
        self.resultQ = resultQ
        self.resultMag = resultMag
        self.resultPhase = resultPhase

#This creates a resonator object from a data dictionary. Optionally performs a fit, and
#adds the fit data back in to the resonator object
def makeResFromData(dataDict, fitFn = None, **kwargs):
    """Create a Resonator object from a data dictionary.

    Return value:
    (res, temp, pwr) or None -- a tuple containing the resonator object and state variables

    Arguments:
    dataDict -- a dict containing frequency, I, and Q data
    fitFn -- if a fit function is passed, an lmfit minimization will be done automatically
    kwargs -- any lmfit Parameter key name and starting value."""
    #Check dataDict for validity
    if dataDict is not None:
        resName = dataDict['name']
        temp = dataDict['temp']
        pwr = dataDict['pwr']
        freqData = dataDict['freq']
        IData = dataDict['I']
        QData = dataDict['Q']

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

        #Run a fit on the resonator if a fit function is specified
        if fitFn is not None:
            res.lmfit(fitFn, **kwargs)

        #Return resonator object plus state variables to make indexing easy
        return (res, res.temp, pwr)
    else:
        return None

#Keep this defined so old code doesn't break. Functionality was moved to
#a Resonator object module called 'lmfit'
def lmfitRes(res, fitFn, **kwargs):
    """Run lmfit on an existing resonator object and update the results.

    Return value:
    No return value -- operates on resonator object in place

    Arguments:
    res -- Existing resonator object. This object will be modified.
    fitFn -- Any lmfit compatible fit function
        fitFn arguments: lmfit parameter object, [Idata, Qdata], [I error, Q error]
    kwargs -- Use this to override any of the lmfit parameter initial guesses
        example: qi=1e6 is equivalent to calling res.params['qi'].value = 1e6
    """

    res.lmfit(fitFn, **kwargs)
