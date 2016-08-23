import matplotlib.pyplot as plt
import numpy as np

def plotResListData(resList, plotTypes=['IQ'], **kwargs):
    pass

def plotResSweepParamsVsTemp(resSweep, keysToPlot=None, keysToIgnore=None, **kwargs):
    """Plot parameter data vs temperature from a ResonatorSweep object."""

    #This will really only work for sure if block is sucessful
    assert resSweep.smartindex == 'block', "index must be 'block' for plotting to work."
    #TODO: fix for other smartindex types

    #set defaults
    fitter = kwargs.pop('fitter', 'lmfit')
    numCols = int(kwargs.pop('numCols', 4))
    powers = list(kwargs.pop('powers', resSweep.pvec))
    assert all(p in resSweep.pvec for p in powers), "Can't plot a power that doesn't exist!"

    #Set up the temperature axis mask
    maxTemp = kwargs.pop('maxTemp', np.max(resSweep.tvec))
    minTemp = kwargs.pop('minTemp', np.min(resSweep.tvec))

    tempMask = (resSweep.tvec >= minTemp) * (resSweep.tvec <= maxTemp)

    if keysToIgnore is None:
        keysToIgnore = ['listIndex',
                        'temps']
    else:
        assert keysToPlot is None, "Either pass keysToPlot or keysToIgnore, not both."
        assert all(key in resSweep.keys() for key in keysToIgnore), "Unknown key"
        keysToIgnore.append('listIndex')
        keysToIgnore.append('temps')


    #Set up the figure
    figS = plt.figure()

    if keysToPlot is None:
        keysToPlot = set(resSweep.keys())-set(keysToIgnore)
    else:
        assert all(key in resSweep.keys() for key in keysToPlot), "Unknown key"

    numKeys = len(keysToPlot)

    #Don't need more columns than plots
    if numKeys < numCols:
        numCols = numKeys

    #Calculate rows for figure size
    numRows = int(np.ceil(numKeys/numCols))

    #Magic numbers!
    figS.set_size_inches(3*numCols,3*numRows)

    #Loop through all the keys in the ResonatorSweep object and plot them
    indexk = 1
    for key in keysToPlot:
        axs = figS.add_subplot(numRows,numCols,indexk)
        for pwr in powers:
            axs.plot(resSweep.tvec[tempMask],resSweep[key][pwr][tempMask],'--',label='Power: '+str(pwr))

        axs.set_xlabel('Temperature (mK)')
        axs.set_ylabel(key)

        #Stick some legends where they won't crowd too much
        if key == 'f0' or key == 'fmin':
            axs.legend(loc='best')

        indexk += 1
    return figS

def plotResSweepParamsVsPwr(resSweep, keysToPlot=None, keysToIgnore=None, **kwargs):
    """Plot parameter data vs power from a ResonatorSweep object."""

    #This will really only work for sure if block is sucessful
    assert resSweep.smartindex == 'block', "index must be 'block' for plotting to work."
    #TODO: fix for other smartindex types

    #set defaults
    fitter = kwargs.pop('fitter', 'lmfit')
    numCols = int(kwargs.pop('numCols', 4))
    temps = list(kwargs.pop('temps', resSweep.tvec))
    assert all(t in resSweep.tvec for t in temps), "Can't plot a temperature that doesn't exist!"

    #Set up the power axis mask
    maxPwr = kwargs.pop('maxPwr', np.max(resSweep.pvec))
    minPwr = kwargs.pop('minPwr', np.min(resSweep.pvec))

    pwrMask = (resSweep.pvec >= minPwr) * (resSweep.pvec <= maxPwr)

    if keysToIgnore is None:
        keysToIgnore = ['listIndex',
                        'temps']
    else:
        assert keysToPlot is None, "Either pass keysToPlot or keysToIgnore, not both."
        assert all(key in resSweep.keys() for key in keysToIgnore), "Unknown key"
        keysToIgnore.append('listIndex')
        keysToIgnore.append('temps')


    #Set up the figure
    figS = plt.figure()

    if keysToPlot is None:
        keysToPlot = set(resSweep.keys())-set(keysToIgnore)
    else:
        assert all(key in resSweep.keys() for key in keysToPlot), "Unknown key"

    numKeys = len(keysToPlot)

    #Don't need more columns than plots
    if numKeys < numCols:
        numCols = numKeys

    #Calculate rows for figure size
    numRows = int(np.ceil(numKeys/numCols))

    #Magic numbers!
    figS.set_size_inches(3*numCols,3*numRows)

    #Loop through all the keys in the ResonatorSweep object and plot them
    indexk = 1
    for key in keysToPlot:
        axs = figS.add_subplot(numRows,numCols,indexk)
        for tmp in temps:
            axs.plot(resSweep.pvec[pwrMask],resSweep[key][pwrMask][tmp],'--',label='Temperature: '+str(tmp))

        axs.set_xlabel('Power (dB)')
        axs.set_ylabel(key)

        #Stick some legends where they won't crowd too much
        if key == 'f0' or key == 'fmin':
            axs.legend(loc='best')

        indexk += 1
    return figS
