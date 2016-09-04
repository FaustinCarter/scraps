import matplotlib.pyplot as plt
import matplotlib.gridspec as gs
import matplotlib as mpl
import mpl_toolkits.mplot3d
import numpy as np
import scipy.signal as sps
from .resonator import indexResList

Axes3D = mpl_toolkits.mplot3d.Axes3D

def plotResListData(resList, plot_types=['IQ'], **kwargs):
    r"""Plot resonator data and fits.

    Parameters
    ----------
    resList : list-like
        A list of ``pyres.Resonator`` objects. A single ``Resonator`` object can
        be passed, as long as it is in a list.

    plot_types : list, optional
        A list of plots to create, each one specified by a string. Possible plot
        types are:

        - 'IQ': Plots the real part of the transmission (`I`) vs the imaginary
            part (`Q`). This is the default plot.

        - 'rIQ': Plots the residual of `I` vs the residual of `Q`. This plot is
            only available if the ``do_lmfit`` method of each ``Resonator``
            object has been called. The `I` and `Q` residuals are normalized by
            the uncertainty of the `I` and `Q` data respectively. If this is not
            explicitly supplied, it is calculated by taking the standard
            deviation of the first 10 data points.

        - 'LinMag': Plots the magnitude of the tranmission in Volts vs
            frequency.

        - 'LogMag': Plots the magnitude of the transmission in dB vs frequency.
            ``LogMag = 20*np.log(LinMag)``.

        - 'rMag': Plots the difference of `LinMag` and the best-fit magnitude vs
            frequency. This plot is only available if the ``do_lmfit`` method of
            each ``Resonator`` object has been called.

        - 'Phase': Plots the phase of the transmision vs frequency.
            ``Phase = np.arctan2(Q, I)``.

        - 'rPhase': Plots the difference of `Phase` and the best-fit phase vs
            frequency. This plot is only available if the ``do_lmfit`` method of
            each ``Resonator`` object has been called.

        - 'uPhase': Plots the unwrapped phase vs frequency.
            ``uPhase = np.unwrap(Phase)``.

        - 'ruPhase': Plots the difference of `uPhase` and the unwrapped best-fit
            phase vs frequency. This plot is only available if the ``do_lmfit``
            method of each ``Resonator`` object has been called.

        - 'I': Plots the real part of the transmission vs frequency.

        - 'rI': Plots the residual of `I` vs frequency. The residual is weighted
            by the uncertainty in `I`. This plot is only available if the
            ``do_lmfit`` method of each ``Resonator`` object has been called.

        - 'Q': Plots the imaginary part of the transmission vs frequency.

        - 'rQ': Plots the residual of `Q` vs frequency. The residual is weighted
            by the uncertainty in `Q`. This plot is only available if the
            ``do_lmfit`` method of each ``Resonator`` object has been called.

    Keyword Arguments
    -----------------

    plot_fits : list-like, optional
        A list of boolean flags, one for each plot type specified. Determines
        whether or not to overplot the best fit on the data. This is only
        effective if the ``do_lmfit`` method of each ``Resonator`` object has
        been called. Default is all False.

    powers : list-like, optional
        A list of power values to plot. Default is to plot all of the unique
        powers that exist in the list of ``Resonator`` objects.

    temps : list-like, optional
        A list of temperature values to plot. Default os to plot all of the
        unique temperatures that exist in the list of ``Resonator`` obejcts.

    use_itemps : {False, True}, optional
        If a ``ResonatorSweep`` object has been generated from the resList it
        may have added the ``itemp`` attrubute to each ``ResonatorObject`` in
        the list. Specifying ``use_itemps = True`` will force the plotting
        routine to use those tempeartures.

    freq_units : {'GHz', 'Hz', 'kHz', 'MHz', 'THz'}, optional
        The units for the frequency axis, if it exists. Defaul is 'GHz'.

    detrend_phase : {False, True}, optional
        Whether or not to remove a linear trend from the `Phase` data. A typical
        reason for a steep linear offset in the phase is an uncorrected
        electrical delay due to long transmission lines.

    num_cols : int, optional
        The number of columns to include in the grid of subplots. Default is 1.

    fig_size : int, optional
        The size of an individual subplot in inches. Default is 3.

    force_square : {False, True}, optional
        Whether or not to force each subplot axis to be perfectly square.

    show_colorbar : {True, False}, optional
        Whether or not to add a colorbar to the right edge of the figure. The
        colorbar will correspond to the limits of the colored data. Default is
        True.

    color_by : {'temps', 'pwrs'}, optional
        If multiple temperatures and multiple powers are passed, this selects
        which variable will set the color of the plots. Default is 'temps'.

    color_map : str, optional
        The name of any colormap returned by calling
        ``matplotlib.pyplot.colormaps()`` is a valid option. Default is
        'coolwarm'.

    plot_kwargs : dict, optional
        A dict of keyword arguments to pass through to the individual plots.
        Attempting to set 'color' will result in an error.

    fit_kwargs : dict, optional
        A dict of keyword arguments to pass through to the fit plots. Default is
        a dashed black line.

    Returns
    -------
    figS : ``matplotlib.pyplot.figure``
        A ``matplotlib.pyplot`` figure object.

    """
    #TODO: Add temperature and power masking that makes more sense, like the ability
    #to set a temperature range, or maybe decimate the temperature data. Also
    #need to add the ability to waterfall the mag and phase plots.
    supported_types = ['IQ', 'rIQ', 'LogMag', 'LinMag', 'rMag',
                        'Phase', 'rPhase', 'uPhase', 'ruPhase',
                        'I', 'rI', 'Q', 'rQ']
    assert all(plt_key in supported_types for plt_key in plot_types), "Unsupported plotType requested!"

    #Get a list of unique temps and powers
    powers = []
    temps = []

    #Should we use itemps?
    use_itemps = kwargs.pop('use_itemps', False)
    if use_itemps:
        assert all(hasattr(res, 'itemps') for res in resList), "Could not locate itemp for at least one resonator!"

    for res in resList:
        powers.append(res.pwr)
        if use_itemps:
            temps.append(res.itemp)
        else:
            temps.append(res.temp)

    powers = np.unique(powers)
    temps = np.unique(temps)

    #Optionally override either list
    powers = kwargs.pop('powers', powers)
    temps = kwargs.pop('temps', temps)

    #Should we plot best fits?
    plot_fits = kwargs.pop('plot_fits', [False]*len(plot_types))
    assert len(plot_fits) == len(plot_types), "Must specify a fit bool for each plot type."
    if any(plot_fits):
        assert all(res.hasFit for res in resList), "At least one resonator has not been fit yet."

    #Set the units for the frequency axes
    freq_units = kwargs.pop('freq_units', 'GHz')
    assert freq_units in ['Hz', 'kHz', 'MHz', 'GHz', 'THz'], "Unsupported units request!"

    unitsDict = {'Hz':1,
                'kHz':1e3,
                'MHz':1e6,
                'GHz':1e9,
                'THz':1e12}

    #Remove linear phase variation? Buggy...
    detrend_phase = kwargs.pop('detrend_phase', False)

    #Set some plotting defaults
    num_cols = kwargs.pop('num_cols', 1)
    fig_size = kwargs.pop('fig_size', 3)
    force_square = kwargs.pop('force_square', False)
    show_colorbar = kwargs.pop('show_colorbar', True)


    #Should the temperatures or the powers iterate the colors?
    color_by = kwargs.pop('color_by', 'temps')
    assert color_by in ['temps', 'pwrs'], "color_by must be 'temps' or 'pwrs'."

    #Set the colormap: Default to a nice red/blue thing
    #TODO: Allow for custom colormaps (like from Seaborn, etc)

    color_map = kwargs.pop('color_map', 'coolwarm')
    assert color_map in plt.colormaps(), "Unknown colormap provided"
    color_gen = plt.get_cmap(color_map)

    #Any extra kwargs for plotting
    plot_kwargs = kwargs.pop('plot_kwargs', {})
    fit_kwargs = kwargs.pop('fit_kwargs', {'color' : 'k',
                                            'linestyle' : '--',
                                            'linewidth' : 1.5})

    if kwargs:
        raise NameError("Unknown keyword argument: " + kwargs.keys()[0])


    #Set up the figure
    figS = plt.figure()

    #Calculate rows for figure size
    num_rows = int(np.ceil(1.0*len(plot_types)/num_cols))

    #Set figure size, including some extra spacing for the colorbar
    #0.1 is the extra space for the colorbar.
    #*1.2 is the extra padding for the axis labels
    #15:1 is the ratio of axis width for regular axes to colorbar axis
    if show_colorbar:
        figS.set_size_inches(fig_size*(num_cols+0.1)*1.2, fig_size*num_rows)

        #Initialize the grid for plotting
        plt_grid = gs.GridSpec(num_rows, num_cols+1, width_ratios=[15]*num_cols+[1])
    else:
        figS.set_size_inches(fig_size*(num_cols)*1.2, fig_size*num_rows)

        #Initialize the grid for plotting
        plt_grid = gs.GridSpec(num_rows, num_cols)

    #Initialize a dictionary of axes corresponding to plot types
    axDict = {}

    #Set up axes and make labels
    for ix, key in enumerate(plot_types):

        iRow = int(ix/num_cols)
        iCol = ix%num_cols

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

        #Stuff the axis into the axis dictionary
        axDict[key] = ax

    #Plot the data
    for pwr in powers:
        for temp in temps:
            #Grab the right resonator from the list
            resIndex = indexResList(resList, temp, pwr, itemp=use_itemps)

            #Color magic!
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

            #Not every temp/pwr combo corresponds to a resonator. Ignore missing ones.
            if resIndex is not None:
                res = resList[resIndex]
                scaled_freq = res.freq/unitsDict[freq_units]

                for key, ax in axDict.iteritems():
                    pix = plot_types.index(key)
                    plot_fit = plot_fits[pix]
                    if key == 'IQ':
                        ax.plot(res.I, res.Q, color=plt_color, **plot_kwargs)
                        if plot_fit:
                            ax.plot(res.resultI, res.resultQ, **fit_kwargs)

                    if key == 'rIQ':
                        ax.plot(res.residualI, res.residualQ, color=plt_color, **plot_kwargs)

                    if key == 'LogMag':
                        ax.plot(scaled_freq, res.logmag, color=plt_color, **plot_kwargs)
                        if plot_fit:
                            ax.plot(scaled_freq, 20*np.log(res.resultMag), **fit_kwargs)

                    if key == 'LinMag':
                        ax.plot(scaled_freq, res.mag, color=plt_color, **plot_kwargs)
                        if plot_fit:
                            ax.plot(scaled_freq, res.resultMag, **fit_kwargs)

                    if key == 'rMag':
                        ax.plot(scaled_freq, res.resultMag-res.mag, color=plt_color, **plot_kwargs)

                    if key == 'Phase':
                        if detrend_phase:
                            ax.plot(scaled_freq, sps.detrend(res.phase), color=plt_color, **plot_kwargs)
                            if plot_fit:
                                ax.plot(scaled_freq, sps.detrend(res.resultPhase), **fit_kwargs)
                        else:
                            ax.plot(scaled_freq, res.phase, color=plt_color, **plot_kwargs)
                            if plot_fit:
                                ax.plot(scaled_freq, res.resultPhase, **fit_kwargs)

                    if key == 'rPhase':
                        ax.plot(scaled_freq, res.resultPhase-res.phase, color=plt_color, **plot_kwargs)

                    if key == 'uPhase':
                        if detrend_phase:
                            ax.plot(scaled_freq, sps.detrend(res.uphase), color=plt_color, **plot_kwargs)
                            if plot_fit:
                                ax.plot(scaled_freq, sps.detrend(np.unwrap(res.resultPhase)), **fit_kwargs)
                        else:
                            ax.plot(scaled_freq, res.uphase, color=plt_color, **plot_kwargs)
                            if plot_fit:
                                ax.plot(scaled_freq, np.unwrap(res.resultPhase), **fit_kwargs)

                    if key == 'ruPhase':
                        ax.plot(scaled_freq, np.unwrap(res.resultPhase)-res.uphase, color=plt_color, **plot_kwargs)

                    if key == 'I':
                        ax.plot(scaled_freq, res.I, color=plt_color, **plot_kwargs)
                        if plot_fit:
                            ax.plot(scaled_freq, res.resultI, **fit_kwargs)

                    if key == 'rI':
                        ax.plot(scaled_freq, res.residualI, color=plt_color, **plot_kwargs)

                    if key == 'Q':
                        ax.plot(scaled_freq, res.Q, color=plt_color, **plot_kwargs)
                        if plot_fit:
                            ax.plot(scaled_freq, res.resultQ, **fit_kwargs)

                    if key == 'rQ':
                        ax.plot(scaled_freq, res.residualQ, color=plt_color, **plot_kwargs)

                    ax.set_xticklabels(ax.get_xticks(),rotation=45)

                    if force_square:
                        #Make the plot a square
                        x1, x2 = ax.get_xlim()
                        y1, y2 = ax.get_ylim()

                        #Explicitly passing a float to avoid a warning in matplotlib
                        #when it gets a numpy.float64
                        ax.set_aspect(float((x2-x1)/(y2-y1)))

    #Add the colorbar
    if show_colorbar:
        #Set limits for the min and max colors
        if color_by == 'temps':
            cbar_norm= mpl.colors.Normalize(vmin=0, vmax=max(temps))
            cbar_units = 'Kelvin'
        elif color_by == 'pwrs':
            cbar_norm = mpl.colors.Normalize(vmin=min(powers), vmax=max(powers))
            cbar_units = 'dB'

        #Make an axis that spans all rows for the colorbar
        cax = figS.add_subplot(plt_grid[:, num_cols])

        #Plot and label the colorbar
        cbar_plot = mpl.colorbar.ColorbarBase(cax, cmap=color_gen, norm=cbar_norm)
        cbar_plot.set_label(cbar_units)

    #Clean up the subfigs and make sure nothing overlaps
    figS.tight_layout()

    return figS



def plotResSweepParamsVsTemp(resSweep, plot_keys=None, ignore_keys=None, **kwargs):
    """Plot parameter data vs temperature from a ResonatorSweep object.

    Parameters
    ----------
    resSweep : ``scraps.ResonatorSweep`` object
        The object containing the data you want to look at.

    plot_keys : list-like (optional)
        A list of strings corresponding to avaiable plot data. The available
        keys depend on your parameter definitions and may be found by executing
        ``print resSweep.keys()``. Some keys may point to empty (NaN) objects.
        Default is to plot all of the keys that exist. If you pass plot_keys
        you may not pass ignore_ignore keys.

    ignore_keys : list-like (optional)
        A list of strings corresponding to plots that should not be made. This
        is useful if you want to plot most of the avaialble data, but ignore one
        or two sets of data. Default is ``None``. If you pass ignore_keys you
        may not pass plot_keys.

    Keyword Arguments
    -----------------
    plot_labels : list-like
        A list of strings to use to label the y-axes of the plots. There must be
        one for each plot requested. ``None`` is acceptable for any position in
        the list and will default to using the key as the label. Default is to
        use the key as the label.

    unit_multipliers : list-like
        A list of numbers to multiply against the y-axis data. There must be one
        for each plot requested. ``None`` is acceptable for any position in the
        list and will default to 1. Default is 1.

    fitter : string {'lmfit', 'emcee'}
        Which fit data to use when overlaying best fits. Default is 'lmfit'.

    num_cols : int
        The number of columns to create in the plot grid. Default is 1. The
        number of rows will be calculated based on num_cols and the number of
        requested plots.

    powers : list
        List of powers to plot. Default is to plot all available.

    max_temp : numeric
        Don't plot any temperatures above this value. Default is infinity.

    min_temp : numeric
        Don't plot any temperatures below this value. Default is 0.

    fig_size : numeric
        Size in inches for each plot in the figure.

    color_map : string
        Specifies the colormap to use. Any value in ``matplotlib.pyplot.colormaps()``
        is a valid option.

    force_square : bool
        Whether or not to force each subplot to have perfectly square axes.

    """

    #This will really only work for sure if block is sucessful
    assert resSweep.smartindex == 'block', "index must be 'block' for plotting to work."
    #TODO: fix for other smartindex types

    #set defaults
    plot_labels = kwargs.pop('plot_labels', None)

    unit_multipliers = kwargs.pop('unit_multipliers', None)

    #Which fit data should be plot? lmfit or emcee?
    fitter = kwargs.pop('fitter', 'lmfit')

    #Number of columns
    num_cols = int(kwargs.pop('num_cols', 4))

    #Powers to plot
    powers = list(kwargs.pop('powers', resSweep.pvec))
    assert all(p in resSweep.pvec for p in powers), "Can't plot a power that doesn't exist!"

    #Set up the temperature axis mask
    max_temp = kwargs.pop('max_temp', np.max(resSweep.tvec))
    min_temp = kwargs.pop('min_temp', np.min(resSweep.tvec))

    tempMask = (resSweep.tvec >= min_temp) * (resSweep.tvec <= max_temp)

    if ignore_keys is None:
        ignore_keys = ['listIndex',
                        'temps']
    else:
        assert plot_keys is None, "Either pass plot_keys or ignore_keys, not both."
        assert all(key in resSweep.keys() for key in ignore_keys), "Unknown key"
        ignore_keys.append('listIndex')
        ignore_keys.append('temps')

    fig_size = kwargs.pop('fig_size', 3)

    force_square = kwargs.pop('force_square', False)

    #Set the colormap: Default to a nice red/blue thing
    color_map = kwargs.pop('color_map', 'rainbow')
    assert color_map in plt.colormaps(), "Unknown colormap provided"
    color_gen = plt.get_cmap(color_map)


    #Set up the figure
    figS = plt.figure()

    if plot_keys is None:
        plot_keys = set(resSweep.keys())-set(ignore_keys)
    else:
        assert all(key in resSweep.keys() for key in plot_keys), "Unknown key"

    num_keys = len(plot_keys)

    #Don't need more columns than plots
    if num_keys < num_cols:
        num_cols = num_keys

    #Calculate rows for figure size
    num_rows = int(np.ceil(num_keys/num_cols))

    #Calculate rows for figure size
    num_rows = int(np.ceil(1.0*num_keys/num_cols))

    #Set figure size, including some extra spacing for the colorbar
    figS.set_size_inches(fig_size*(num_cols+0.1)*1.2, fig_size*num_rows)

    #Initialize the grid for plotting
    plt_grid = gs.GridSpec(num_rows, num_cols+1, width_ratios=[15]*num_cols+[1])

    #Loop through all the keys in the ResonatorSweep object and plot them
    for ix, key in enumerate(plot_keys):

        iRow = int(ix/num_cols)
        iCol = ix%num_cols

        axs = figS.add_subplot(plt_grid[iRow, iCol])

        if unit_multipliers is not None:
            mult = unit_multipliers[ix]
        else:
            mult = 1

        for pwr in powers:
            if len(powers) > 1:
                plt_color = color_gen(1-((max(powers)-pwr)*1.0/(max(powers)-min(powers))))
            else:
                plt_color = color_gen(0)

            axs.plot(resSweep.tvec[tempMask],mult*resSweep[key].loc[tempMask, pwr],'--',color=plt_color, label='Power: '+str(pwr))

        axs.set_xlabel('Temperature (mK)')
        if plot_labels is not None:
            axs.set_ylabel(plot_labels[ix])
        else:
            axs.set_ylabel(key)
        axs.set_xticklabels(axs.get_xticks(),rotation=45)

        if force_square:
            #Make the plot a square
            x1, x2 = axs.get_xlim()
            y1, y2 = axs.get_ylim()

            #Explicitly passing a float to avoid an warning in matplotlib
            #when it gets a numpy.float64
            axs.set_aspect(float((x2-x1)/(y2-y1)))

        #Stick some legends where they won't crowd too much
        # if key == 'f0' or key == 'fmin':
        #     axs.legend(loc='best')

    cbar_norm = mpl.colors.Normalize(vmin=min(powers), vmax=max(powers))
    cbar_units = 'dB'

    #Make an axis that spans all rows
    cax = figS.add_subplot(plt_grid[:, num_cols])

    #Plot and label
    cbar_plot = mpl.colorbar.ColorbarBase(cax, cmap=color_gen, norm=cbar_norm)
    cbar_plot.set_label(cbar_units)

    figS.tight_layout()
    return figS

def plotResSweepParamsVsPwr(resSweep, plot_keys=None, ignore_keys=None, **kwargs):
    """Plot parameter data vs power from a ResonatorSweep object.

    Parameters
    ----------
    resSweep : ``scraps.ResonatorSweep`` object
        The object containing the data you want to look at.

    plot_keys : list-like (optional)
        A list of strings corresponding to avaiable plot data. The available
        keys depend on your parameter definitions and may be found by executing
        ``print resSweep.keys()``. Some keys may point to empty (NaN) objects.
        Default is to plot all of the keys that exist. If you pass plot_keys
        you may not pass ignore_ignore keys.

    ignore_keys : list-like (optional)
        A list of strings corresponding to plots that should not be made. This
        is useful if you want to plot most of the avaialble data, but ignore one
        or two sets of data. Default is ``None``. If you pass ignore_keys you
        may not pass plot_keys.

    Keyword Arguments
    -----------------
    plot_labels : list-like
        A list of strings to use to label the y-axes of the plots. There must be
        one for each plot requested. ``None`` is acceptable for any position in
        the list and will default to using the key as the label. Default is to
        use the key as the label.

    unit_multipliers : list-like
        A list of numbers to multiply against the y-axis data. There must be one
        for each plot requested. ``None`` is acceptable for any position in the
        list and will default to 1. Default is 1.

    fitter : string {'lmfit', 'emcee'}
        Which fit data to use when overlaying best fits. Default is 'lmfit'.

    num_cols : int
        The number of columns to create in the plot grid. Default is 1. The
        number of rows will be calculated based on num_cols and the number of
        requested plots.

    temps : list
        List of temperatures to plot. Default is to plot all available.

    max_power : numeric
        Don't plot any powers above this value. Default is infinity.

    min_power : numeric
        Don't plot any powers below this value. Default is -infinity.

    fig_size : numeric
        Size in inches for each plot in the figure.

    color_map : string
        Specifies the colormap to use. Any value in ``matplotlib.pyplot.colormaps()``
        is a valid option.

    force_square : bool
        Whether or not to force each subplot to have perfectly square axes.

    """

    #This will really only work for sure if block is sucessful
    assert resSweep.smartindex == 'block', "index must be 'block' for plotting to work."
    #TODO: fix for other smartindex types

    #set defaults:
    plot_labels = kwargs.pop('plot_labels', None)
    unit_multipliers = kwargs.pop('unit_multipliers', None)

    #Which fit data should be plot? lmfit or emcee?
    fitter = kwargs.pop('fitter', 'lmfit')

    #Number of columns
    num_cols = int(kwargs.pop('num_cols', 4))

    #Temperature values to plot
    temps = list(kwargs.pop('temps', resSweep.tvec))
    assert all(t in resSweep.tvec for t in temps), "Can't plot a temperature that doesn't exist!"

    #Set up the power axis mask
    max_power = kwargs.pop('max_power', np.max(resSweep.pvec))
    min_power = kwargs.pop('min_power', np.min(resSweep.pvec))

    pwrMask = (resSweep.pvec >= min_power) * (resSweep.pvec <= max_power)

    fig_size = kwargs.pop('fig_size', 3)

    force_square = kwargs.pop('force_square', False)

    #Set the colormap: Default to a nice red/blue thing
    color_map = kwargs.pop('color_map', 'coolwarm')
    assert color_map in plt.colormaps(), "Unknown colormap provided"
    color_gen = plt.get_cmap(color_map)

    if ignore_keys is None:
        ignore_keys = ['listIndex',
                        'temps']
    else:
        assert plot_keys is None, "Either pass plot_keys or ignore_keys, not both."
        assert all(key in resSweep.keys() for key in ignore_keys), "Unknown key"
        ignore_keys.append('listIndex')
        ignore_keys.append('temps')


    #Set up the figure
    figS = plt.figure()

    if plot_keys is None:
        plot_keys = set(resSweep.keys())-set(ignore_keys)
    else:
        assert all(key in resSweep.keys() for key in plot_keys), "Unknown key"

    num_keys = len(plot_keys)

    #Don't need more columns than plots
    if num_keys < num_cols:
        num_cols = num_keys

    #Calculate rows for figure size
    num_rows = int(np.ceil(1.0*num_keys/num_cols))

    #Set figure size, including some extra spacing for the colorbar
    figS.set_size_inches(fig_size*(num_cols+0.1)*1.2, fig_size*num_rows)

    #Initialize the grid for plotting
    plt_grid = gs.GridSpec(num_rows, num_cols+1, width_ratios=[15]*num_cols+[1])

    #Loop through all the keys in the ResonatorSweep object and plot them
    for ix, key in enumerate(plot_keys):

        iRow = int(ix/num_cols)
        iCol = ix%num_cols

        axs = figS.add_subplot(plt_grid[iRow, iCol])

        if unit_multipliers is not None:
            mult = unit_multipliers[ix]
        else:
            mult = 1

        for tmp in temps:
            if len(temps) > 1:
                plt_color = color_gen(tmp*1.0/max(temps))
            else:
                plt_color = color_gen(0)

            axs.plot(resSweep.pvec[pwrMask],resSweep[key].loc[tmp,pwrMask],'--', color=plt_color, label='Temperature: '+str(tmp))

        axs.set_xlabel('Power (dB)')
        if plot_labels is not None:
            axs.set_ylabel(plot_labels[ix])
        else:
            axs.set_ylabel(key)
        axs.set_xticklabels(axs.get_xticks(),rotation=45)

        if force_square:
            #Make the plot a square
            x1, x2 = axs.get_xlim()
            y1, y2 = axs.get_ylim()

            #Explicitly passing a float to avoid an warning in matplotlib
            #when it gets a numpy.float64
            axs.set_aspect(float((x2-x1)/(y2-y1)))

        #Stick some legends where they won't crowd too much
        # if key == 'f0' or key == 'fmin':
        #     axs.legend(loc='best')
    #Make an axis that spans all rows
    cax = figS.add_subplot(plt_grid[:, num_cols])

    #Set colorbar limits
    cbar_norm= mpl.colors.Normalize(vmin=0, vmax=max(temps))
    cbar_units = 'mK'

    #Plot and label
    cbar_plot = mpl.colorbar.ColorbarBase(cax, cmap=color_gen, norm=cbar_norm)
    cbar_plot.set_label(cbar_units)

    figS.tight_layout()
    return figS

def plotResSweep3D(resSweep, plot_keys, **kwargs):
    r"""Make 3D surface or mesh plots of any key in the ``ResonatorSweep`` object as functions
    of temperature and power.

    Parameters
    ----------
    resSweep : ``ResonatorSweep`` object
        A ``pyres.ResonatorSweep`` object containing all of the data to be
        plotted.

    plot_keys : list-like
        List of strings where each string is a key corresponding to a plot that
        should be made. For a list of acccetable keys, run ``print
        resSweep.keys()``.

    Keyword arguments
    -----------------

    min_temp : numeric (optional)
        The minimum temperature to plot. Defaults to ``min(temperatures)``.

    max_temp : numeric (optional)
        The maximum temperature to plot. Defaults to ``max(temperatres)``.

    min_pwr : numeric (optional)
        The minimum power to plot. Defaults to ``min(powers)``.

    max_pwr : numeric (optional)
        The maximum power to plot. Defauts to ``max(powers)``.

    unit_multipliers : list (optional)
        Values to scale the z-axis by. Default is 1.
        ``len(unit_multipliers) == len(plot_keys)``.

    plot_labels : list (optional)
        Labels for the z-axis. Default is the plot key.
        ``len(plot_labels) == len(plot_keys)``.

    num_cols : int (optional)
        The number of columns in the resulting plot grid. Default is 1.

    fig_size : numeric (optional)
        The size of an individual subplot in inches. Default is 3.

    plot_lmfits : bool (optional)
        Whether or not to show a fit. The fit must exist! Default is False.

    plot_kwargs : dict (optional)
        A dictionary of keyword arguments to pass the plotting function.
        Default is ``None``.

    Note
    ----
    This function is currently a little buggy, because I can't figure out how
    to intelligently adjust the label positions, sizes, etc to deal with large
    numbers in the ticks. The current workaround is to pick a large fig_size
    (so far anything larger than 5 seems ok) and then scale the plot as needed
    in some other application.

    You can also use something like the following to adjust the ticks on a
    specific axis::

        figX.axes[0].tick_params(axis='z', pad=8)
        figX.axes[0].zaxis.labelpad = 13

    """
    #TODO: Fix the labels issues when tick labels are very long.

    #Some plotting niceties
    plot_labels = kwargs.pop('plot_labels', None)
    if plot_labels is not None:
        assert len(plot_labels) == len(plot_keys), "Number of labels must equal number of plots."

    unit_multipliers = kwargs.pop('unit_multipliers', None)
    if unit_multipliers is not None:
        assert len(unit_multipliers) == len(plot_keys), "One multiplier per plot required."

    #Set some limits
    min_temp = kwargs.pop('min_temp', min(resSweep.tvec))
    max_temp = kwargs.pop('max_temp', max(resSweep.tvec))
    t_filter = (resSweep.tvec >= min_temp) * (resSweep.tvec <= max_temp)

    min_pwr = kwargs.pop('min_pwr', min(resSweep.pvec))
    max_pwr = kwargs.pop('max_pwr', max(resSweep.pvec))
    p_filter = (resSweep.pvec >= min_pwr) * (resSweep.pvec <= max_pwr)

    #Check for fits
    plot_lmfits = kwargs.pop('plot_lmfits', False)
    assert all('lmfit_'+key in resSweep.keys() for key in plot_keys), "No fit to plot for "+key+"."

    #Get all the possible temperature/power combos in two lists
    ts, ps = np.meshgrid(resSweep.tvec[t_filter], resSweep.pvec[p_filter])

    #Set up the figure
    figS = plt.figure()

    fig_size = kwargs.pop('fig_size', 3)

    assert all(key in resSweep.keys() for key in plot_keys), "Unknown key"

    num_keys = len(plot_keys)
    num_cols = kwargs.pop('num_cols', 1)

    #Don't need more columns than plots
    if num_keys < num_cols:
        num_cols = num_keys

    #Calculate rows for figure size
    num_rows = int(np.ceil(1.0*num_keys/num_cols))

    #Set figure size, including some extra spacing for the colorbar
    figS.set_size_inches(fig_size*(num_cols)*1.5, fig_size*num_rows)

    #Initialize the grid for plotting
    plt_grid = gs.GridSpec(num_rows, num_cols)

    #Grab any kwargs for the plotting functions
    plot_kwargs = kwargs.pop('plot_kwargs', {})

    #Set some defaults that can be overwritten
    if 'cstride' not in plot_kwargs.keys():
        plot_kwargs['cstride'] = 1

    if 'rstride' not in plot_kwargs.keys():
        plot_kwargs['rstride'] = 1

    if 'cmap' not in plot_kwargs.keys():
        plot_kwargs['cmap'] = 'viridis'

    if 'alpha' not in plot_kwargs.keys():
        plot_kwargs['alpha'] = 0.2

    #Grab any kwargs for the fits
    fit_kwargs = kwargs.pop('fit_kwargs', {})

    #Set some defaults
    if 'color' not in fit_kwargs.keys():
        fit_kwargs['color'] = 'k'

    if 'linestyle' not in fit_kwargs.keys():
        fit_kwargs['linestyle'] ='--'



    #Loop through all the keys in the ResonatorSweep object and plot them
    for ix, key in enumerate(plot_keys):

        iRow = int(ix/num_cols)
        iCol = ix%num_cols

        axs = figS.add_subplot(plt_grid[iRow, iCol], projection='3d')

        axs.ticklabel_format(useOffset=False)

        plt_data = resSweep[key].loc[t_filter, p_filter].values.T

        if unit_multipliers is not None:
            mult = unit_multipliers[ix]
        else:
            mult = 1

        axs.plot_surface(ts, ps, mult*plt_data, **plot_kwargs)

        if plot_lmfits:
            fit_data = resSweep['lmfit_'+key].loc[t_filter, p_filter].values.T
            axs.plot_wireframe(ts, ps, mult*fit_data, **fit_kwargs)

        if plot_labels is not None:
            axs.set_zlabel(plot_labels[ix])
        else:
            axs.set_zlabel(key)
        axs.set_ylabel('Power (dB)')
        axs.set_xlabel('Temperature (mK)')


    figS.tight_layout()

    return figS
