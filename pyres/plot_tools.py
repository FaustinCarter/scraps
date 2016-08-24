import matplotlib.pyplot as plt
import matplotlib.gridspec as gs
import matplotlib as mpl
import numpy as np
from .containers import indexResList

def plotResListData(resList, plotTypes=['IQ'], **kwargs):
    #TODO: Add temperature and power masking that makes more sense, like the ability
    #to set a temperature range, or maybe decimate the temperature data. Also
    #need to add the ability to waterfall the mag and phase plots.
    supportedTypes = ['IQ', 'rIQ', 'LogMag', 'LinMag', 'rMag',
                        'Phase', 'rPhase', 'uPhase', 'ruPhase',
                        'I', 'rI', 'Q', 'rQ']

    assert all(pt in supportedTypes for pt in plotTypes), "Unsupported plotType requested!"

    powers = []
    temps = []

    use_itemps = kwargs.pop('use_itemps', False)

    for res in resList:
        powers.append(res.pwr)
        if use_itemps:
            temps.append(res.itemp)
        else:
            temps.append(res.temp)

    powers = np.unique(powers)
    temps = np.unique(temps)

    powers = kwargs.pop('powers', powers)
    temps = kwargs.pop('temps', temps)

    numCols = kwargs.pop('numCols', 1)
    plot_fits = kwargs.pop('plot_fits', False)

    freq_units = kwargs.pop('freq_units', 'GHz')
    assert freq_units in ['Hz', 'kHz', 'MHz', 'GHz', 'THz'], "Unsupported units request!"

    unitsDict = {'Hz':1,
                'kHz':1e3,
                'MHz':1e6,
                'GHz':1e9,
                'THz':1e12}

    fig_size = kwargs.pop('fig_size', 3)

    #Set the colormap: Default to a nice red/blue thing
    color_map = kwargs.pop('color_map', 'coolwarm')
    assert color_map in plt.colormaps(), "Unknown colormap provided"
    color_gen = plt.get_cmap(color_map)

    #Should the temperatures or the powers iterate the colors?
    color_by = kwargs.pop('color_by', 'temps')
    assert color_by in ['temps', 'pwrs'], "color_by must be 'temps' or 'pwrs'."


    #Set up the figure
    figS = plt.figure()

    #Calculate rows for figure size
    numRows = int(np.ceil(1.0*len(plotTypes)/numCols))

    #Set figure size, including some extra spacing for the colorbar
    figS.set_size_inches(fig_size*(numCols+0.1), fig_size*numRows)

    #Initialize the grid for plotting
    plt_grid = gs.GridSpec(numRows, numCols+1, width_ratios=[10]*numCols+[1])

    #Make a dictionary of axes corresponding to plot types
    axDict = {}

    #Set up axes and make labels
    for ix, key in enumerate(plotTypes):

        iRow = int(ix/numCols)
        iCol = ix%numCols

        ax = figS.add_subplot(plt_grid[iRow, iCol])

        if key == 'IQ':
            ax.set_xlabel('I (Volts)')
            ax.set_ylabel('Q (Volts)')

        if key == 'rIQ':
            ax.set_xlabel('Residual of I / $\sigma_\mathrm{I}$')
            ax.set_ylabel('Residual of Q / $\sigma_\mathrm{Q}$')

        if key in ['LogMag', 'LinMag', 'rMag', 'Phase', 'rPhase', 'uPhase',
                    'ruPhase', 'I', 'Q', 'rQ', 'rI']:
            ax.set_xlabel('Frequency (' + freq_units + ')')

        if key == 'LinMag':
            ax.set_ylabel('Magnitude (Volts)')

        if key == 'LogMag':
            ax.set_ylabel('Magnitude (dB)')

        if key == 'rMag':
            ax.set_ylabel('Residual of Magnitude (Volts)')

        if key == 'Phase':
            ax.set_ylabel('Phase (Radians)')

        if key == 'rPhase':
            ax.set_ylabel('Residual of Phase (Radians)')

        if key == 'uPhase':
            ax.set_ylabel('Unwrapped Phase (Radians)')

        if key == 'ruPhase':
            ax.set_ylabel('Residual of unwrapped Phase (Radians)')

        if key == 'I':
            ax.set_ylabel('I (Volts)')

        if key == 'Q':
            ax.set_ylabel('Q (Volts)')

        if key == 'rI':
            ax.set_ylabel('Residual of I / $\sigma_\mathrm{I}$')

        if key == 'rQ':
            ax.set_ylabel('Residual of Q / $\sigma_\mathrm{Q}$')

        axDict[key] = ax

    #Plot the data
    for pwr in powers:
        for temp in temps:
            resIndex = indexResList(resList, temp, pwr, itemp=use_itemps)

            if color_by == 'temps':
                if len(temps) > 1:
                    plt_color = color_gen(temp*1.0/max(temps))
                else:
                    plt_color = color_gen(0)
            elif color_by == 'pwrs':
                if len(powers) > 1:
                    plt_color = color_gen(1-((max(powers)-pwr)*1.0/(max(powers)-min(powers))))
                else:
                    plt_color = color_gen(0)

            if resIndex is not None:
                res = resList[resIndex]
                scaled_freq = res.freq/unitsDict[freq_units]

                for key, ax in axDict.iteritems():
                    if key == 'IQ':
                        ax.plot(res.I, res.Q, color=plt_color)
                        if plot_fits:
                            ax.plot(res.resultI, res.resultQ, 'k--')

                    if key == 'rIQ':
                        ax.plot(res.residualI, res.residualQ, color=plt_color)

                    if key == 'LogMag':
                        ax.plot(scaled_freq, res.logmag, color=plt_color)
                        if plot_fits:
                            ax.plot(scaled_freq, 20*np.log(res.resultMag), 'k--')

                    if key == 'LinMag':
                        ax.plot(scaled_freq, res.mag, color=plt_color)
                        if plot_fits:
                            ax.plot(scaled_freq, res.resultMag, 'k--')

                    if key == 'rMag':
                        ax.plot(scaled_freq, res.resultMag-res.mag, color=plt_color)

                    if key == 'Phase':
                        ax.plot(scaled_freq, res.phase, color=plt_color)
                        if plot_fits:
                            ax.plot(scaled_freq, res.resultPhase, 'k--')

                    if key == 'rPhase':
                        ax.plot(scaled_freq, res.resultPhase-res.phase, color=plt_color)

                    if key == 'uPhase':
                        ax.plot(scaled_freq, res.uphase, color=plt_color)
                        if plot_fits:
                            ax.plot(scaled_freq, np.unwrap(res.resultPhase), 'k--')

                    if key == 'ruPhase':
                        ax.plot(scaled_freq, np.unwrap(res.resultPhase)-res.uphase, color=plt_color)

                    if key == 'I':
                        ax.plot(scaled_freq, res.I, color=plt_color)
                        if plot_fits:
                            ax.plot(scaled_freq, res.resultI, 'k--')

                    if key == 'rI':
                        ax.plot(scaled_freq, res.residualI, color=plt_color)

                    if key == 'Q':
                        ax.plot(scaled_freq, res.Q, color=plt_color)
                        if plot_fits:
                            ax.plot(scaled_freq, res.resultQ, 'k--')

                    if key == 'rQ':
                        ax.plot(scaled_freq, res.residualQ, color=plt_color)

                    ax.set_xticklabels(ax.get_xticks(),rotation=45)

    #Add the colorbar

    #Set limits for the min and max colors
    if color_by == 'temps':
        cbar_norm= mpl.colors.Normalize(vmin=0, vmax=max(temps))
        cbar_units = 'Kelvin'
    elif color_by == 'pwrs':
        cbar_norm = mpl.colors.Normalize(vmin=min(powers), vmax=max(powers))
        cbar_units = 'dB'

    #Make an axis that spans all rows
    cax = figS.add_subplot(plt_grid[:, numCols])

    #Plot and label
    cbar_plot = mpl.colorbar.ColorbarBase(cax, cmap=color_gen, norm=cbar_norm)
    cbar_plot.set_label(cbar_units)

    figS.tight_layout()

    return figS



def plotResSweepParamsVsTemp(resSweep, keysToPlot=None, keysToIgnore=None, **kwargs):
    """Plot parameter data vs temperature from a ResonatorSweep object."""

    #This will really only work for sure if block is sucessful
    assert resSweep.smartindex == 'block', "index must be 'block' for plotting to work."
    #TODO: fix for other smartindex types

    #set defaults

    #Which fit data should be plot? lmfit or emcee?
    fitter = kwargs.pop('fitter', 'lmfit')

    #Number of columns
    numCols = int(kwargs.pop('numCols', 4))

    #Powers to plot
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
    figS.tight_layout()
    return figS

def plotResSweepParamsVsPwr(resSweep, keysToPlot=None, keysToIgnore=None, **kwargs):
    """Plot parameter data vs power from a ResonatorSweep object."""

    #This will really only work for sure if block is sucessful
    assert resSweep.smartindex == 'block', "index must be 'block' for plotting to work."
    #TODO: fix for other smartindex types

    #set defaults:

    #Which fit data should be plot? lmfit or emcee?
    fitter = kwargs.pop('fitter', 'lmfit')

    #Number of columns
    numCols = int(kwargs.pop('numCols', 4))

    #Temperature values to plot
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
    figS.tight_layout()
    return figS
