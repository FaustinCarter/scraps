import emcee as mc
import lmfit as lf
import numpy as np

def makeBall(mcVec, powTen, numWalkers):
    """Make a ball of starting values clustered around the starting guess."""
    ball = [mcVec + mcVec*np.random.rand(len(mcVec))*10**powTen for walker in range(numWalkers)]
    return ball

def lfParams2paramsVec(params):
    """Unpack lmfit Parameters object into a vector of parameter values."""
    paramsDict = params.valuesdict()
    paramsVec = [value for value in paramsDict.itervalues()]
    return paramsVec

def lfParams2mcVec(params):
    """Unpack lmfit Parameters object into a vector of only varying parameter values."""
    mcVec = [params[key].value for key in params.keys() if params[key].vary == True]
    return mcVec

def mcVec2paramsVec(mcVec, params):
    """Insert the non-varying parameter values back in to the parameters vector.

    Return value:
    paramsVec -- list of paramter values in same order as params

    Arguments:
    mcVec -- list of varying parameter values
    params -- lmfit Parameters object"""
    paramsVec = mcVec
    for index, key in enumerate(params.keys()):
        if params[key].vary == False:
            paramsVec = np.insert(paramsVec, index, params[key].value)

    return paramsVec

def logProbFn(mcVec, logLikeFn, logPriorFn, fitFn, params, freqs, data, sigmas):
    """Return the log-Probability of the sampling distribution.

    Return value:
    logProb -- float

    Arguments:
    mcVec -- vector of varying model parameters
    logLikeFn -- log-liklihood function
    logPriorFn -- log-Prior function containing Bayesian priors
    fitFn -- the model that is used by logLikeFn
    params -- lmfit Parameters object
    freqs -- vector of frequencies at which data exists
    data -- vector of complex resonator data in the form [I, Q]
    sigmas -- errors on data values in the form [sigmaI, sigmaQ]

    2*len(freqs) == len(data) == len(sigmas)"""
    #Pad the mcVec with the non-varying parameter values in the right locations
    paramsVec = mcVec2paramsVec(mcVec, params)

    #Update the log-liklihood using the fitFn and the new paramsVec
    logLike = logLikeFn(fitFn, paramsVec, freqs, data, sigmas)

    #Update the prior using the parameter bounds and the new paramsVec
    logPrior = logPriorFn(paramsVec, params)

    #Update the log-Probability
    logProb = logLike + logPrior
    return logProb

def logLikeNormal(fitFn, paramsVec, freqs, data, sigmas):
    """Return the log-liklihood of the weighted residual in a normal distribution.

    Return value:
    log-liklihood -- single float

    Arguments:
    fitFn -- function that returns a weighted residual based on a model
    paramsVec -- lmfit Parameters object containing fitFn model parameters
    freqs -- vector of frequencies at which to evaluate model
    data -- vector of data to compare with model
    sigmas -- optional vector of uncertaintaty associated with each data point

    2*len(freqs) == len(data) == len(sigmas)"""
    #calculate the residual, which should already be weighted in the fitFn
    residual = fitFn(paramsVec, freqs, data, sigmas)

    #Return the log-liklihood of a normally distributed residual
    return -0.5*np.sum(np.log(2*np.pi*sigmas**2)+residual**2)

def logPriorFlat(paramsVec, params):
    """Return natural log of prior probability based on params bounds.

    Return-value:
    logProb -- either 0 or -inf

    Arguments:
    paramsVec -- list of parameter values in same order as params
    params -- lmfit Parameters object containing parameter bounds"""
    logPrior = 0 #ln(1) = 0
    if params is None:
        #Maximally flat prior: p=1 always
        pass
    else:
        paramsDict = params.valuesdict()

        #Loop through parameter bounds and update the prior
        for kindex, key in enumerate(paramsDict.keys()):
            if (params[key].min < paramsVec[kindex] < params[key].max):
                pass
            else:
                logPrior = -np.inf #ln(0) = -inf
    return logPrior
