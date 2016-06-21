import numpy as np
import scipy.signal as sps
import scipy.special as spc

def IQcircle(paramsVec, freqs, data=None, eps=None):
    """Return complex S21 resonance model or, if data is specified, a residual.

    Return value:
    model or (model-data) -- a complex vector [I, Q]
        len(I) = len(Q) = len(freqs) = len(data)/2

    Arguments:
    params -- a list or an lmfit Parameters object containing (df, f0, qc, qi, gain0, gain1, gain2, pgain1, pgain2)
    freqs -- a vector of frequency points at which the model is calculated
    data -- a vector of complex data in the form [I, Q]
        len(I) = len(Q) = len(freqs)
    eps -- a vector of errors for each point in data"""
    #Check if the paramsVec looks like a lmfit params object. If so, unpack to list
    if hasattr(paramsVec, 'valuesdict'):
        paramsDict = paramsVec.valuesdict()
        paramsVec = [value for value in paramsDict.itervalues()]

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

    #Voltage offset at mixer output. Not needed for VNA
    Ioffset = paramsVec[9]
    Qoffset = paramsVec[10]

    #Make everything referenced to the shifted, unitless, reduced frequency
    fs = f0+df
    ff = (freqs-fs)/fs

    #Calculate the total Q_0
    q0 = 1./(1./qi+1./qc)

    #Calculate magnitude and phase gain
    gain = gain0 + gain1*(freqs-fs)+ 0.5*gain2*(freqs-fs)**2
    pgain = np.exp(1j*(pgain0 + pgain1*(freqs-fs)))

    #Allow for voltage offset of I and Q
    offset = Ioffset + 1j*Qoffset

    #Calculate model from params at each point in freqs
    modelCmplx = -gain*pgain*(1./qi+1j*2.0*(ff+df/fs))/(1./q0+1j*2.0*ff)+offset

    #Package complex data in 1D vector form
    modelI = np.real(modelCmplx)
    modelQ = np.imag(modelCmplx)
    model = np.concatenate((modelI, modelQ),axis=0)

    #Calculate eps from stdev of data if not supplied
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
