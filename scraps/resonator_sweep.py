from __future__ import division
import pandas as pd
import numpy as np
import lmfit as lf
from .resonator import makeResFromData, makeResList, indexResList
from .process_file import process_file

#This is a glorified dictionary with a custom initalize method. It takes a list
#of resonator objects that have been fit, and then sets up a
#dict of pandas DataFrame objects, one for each interesting fit parameter
#This adds no new information, but makes accessing the fit data easier
class ResonatorSweep(dict):
    r"""Dictionary object with custom ``__init__`` method.\

    Parameters
    ----------
    resList : list-like
        A list of ``scraps.Resonator`` objects. Each object must have the
        attribute ``hasFit == True``.

    Keyword Arguments
    -----------------
    index : string{'raw', 'round', 'block'} (optional)
        Selects which method to use for indexing.

    roundto : int (optional)
        Number to round temperature index to in mK. Default is 5.

    Attributes
    ----------
    tvec : array-like[nUniqeTemps]
        Index of temperature values, one for each unique temperature.

    pvec : array-like[nUniqePowers]
        Index of powers values, one for each unique power.

    smartindex : string
        Indicates the method of formatting temperature values in tvec. 'raw'
        means to take temperature values as they are, 'round' means to round to
        the neareast `X` mK, where `X` is set by `roundTo`, 'block' means to
        figure out which temperature values are nominally the same and set each
        block of those temperatures to the same value.

    rountTo : float
        The number of mK to round to when ``smartIndex == 'round'``.

    lmfit_results : dict
        A dictionary containing the fit results for different data by key.
        Initially empty, results are added by calling
        ``ResonatorSweep.do_lmfit``.

    emcee_results : dict
        A dictionary containing the MCMC results for different data by key.
        Initially empty, results are added by calling
        ``ResonatorSweep.do_emcee``.

    lmfit_joint_results : dict
        If multiple data sets are fit simultaneously via
        ``ResonatorSweep.do_lmfit``, the results will appear in this attribute
        by key, where ``key = 'key1+key2' == 'key2+key1'.

    emcee_joint_results : dict
        If multiple data sets are fit simultaneously via
        ``ResonatorSweep.do_emcee``, the results will appear in this attribute
        by key, where ``key = 'key1+key2'.

    Note
    ----
    The following keys are added to the self dict:

    'temps' : ``pandas.DataFrame``
        The temperature of each resonator in the sweep.

    'fmin' : ``pandas.DataFrame``
        The minimum value of the magnitude vs frequency curve after subtraction
        of the best-guess baseline.

    'listIndex' : ``pandas.DataFrame``
        The sweep objects are built from a list of ``pyres.Resonator``
        objects; this is the index of the original list for
        each data value in the sweep.

    If the ``do_lmfit`` method has been run, these keys are also added:

    'chisq' : ``pandas.DataFrame``
        The Chi-squared value of each fit in the sweep.

    'redchi' : ``pandas.DataFrame``
        The reduced Chi-squared value of each fit in the sweep.

    'feval' : ``pandas.DataFrame``
        The number of function evaluations for each fit in the sweep.

    paramNames : ``pandas.DataFrame``
        There is a key for each parameter value in the ``Resonator.params``
        attribute.

    paramNames + '_sigma' : ``pandas.DataFrame``
        There is a key for the uncertainty on each paramter value returned by
        the fit. Key name is paramName + '_sigma'

    If the ``do_emcee`` method has been run, these keys are also added:

    paramNames + '_mle' : ``pandas.DataFrame``
        The maximum-liklihood estimate from the MCMC sampling.

    paramNames + '_mc' : ``pandas.DataFrame``
        The 50th percentile value of the posterior distribution from MCMC
        sampling.

    paramNames + '_sigma_plus_mc' : ``pandas.DataFrame``
        The plus errorbar on the MCMC data. Calculated as the difference between
        the 84th percentile value and the 50th percentile value.

    paramNames + '_sigma_minus_mc' : ``pandas.DataFrame``
        The minus errorbar on the MCMC data. Calculated as the difference between
        the 50th percentile value and the 16th percentile value.


    """


    def __init__(self, resList, **kwargs):
        """Formats temp/pwr sweeps into easily parsed pandas DataFrame objects.

        """
        #Call the base class initialization for an empty dict.
        #Not sure this is totally necessary, but don't want to break the dict...
        dict.__init__(self)

        #Create some objects that will be filled in the future:
        self.lmfit_results = {} #Holds fit results for individual quantities
        self.emcee_results = {}
        self.lmfit_joint_results = {} #Holds fit results for joint quantities
        self.emcee_joint_results = {}

        #Build a list of keys that will eventually become the dict keys:

        #Start with the list of fit parameters, want to save all of them
        #Can just use the first resonator's list, as they are all the same.
        #params is NOT an lmfit object.
        params = list(resList[0].params.keys())

        #Add a few more
        #TODO: Right now, only fit information from the most recent fit is stored
        #TODO: Would be good to maybe have keys for each joint fit?
        params.append('temps') #Actual temperature value of measured resonator
        params.append('fmin') #Frequency at magnitude minimum
        params.append('chisq') #Chi-squared value from fit
        params.append('redchi') #Reduced chi-squared value
        params.append('feval') #Number of function evaluations to converge on fit
        params.append('listIndex') #Index in resonator list of resonator

        #If a model explicitly uses qi and qc, then calculate q0
        if all(p in params for p in ['qi', 'qc']):
            params.append('q0') #The total Q = qi*qc/(qi+qc)



        #This flag sets different indexing methods
        self.smartindex = kwargs.pop('index', 'raw')
        assert self.smartindex in ['raw', 'round', 'block'], "index must be 'raw', 'round', or 'block'."

        #Default the rounding option to 5 mK
        self.roundto = kwargs.pop('roundto', 5)

        if kwargs:
            raise ValueError("Unknown keyword: " + kwargs.keys()[0])


        #Loop through the resList and make lists of power and index temperature
        tvals = np.empty(len(resList))
        pvals = np.empty(len(resList))

        if self.smartindex == 'round':
            itvals = np.empty(len(resList))

        for index, res in enumerate(resList):
            tvals[index] = res.temp
            pvals[index] = res.pwr

            if self.smartindex == 'round':
                #itemp is stored in mK
                itmp = np.round(res.temp*1000/self.roundto)*self.roundto
                itvals[index] = itmp


        #Create index vectors containing only the unique values from each list
        tvec = np.sort(np.unique(tvals))
        self.pvec = np.sort(np.unique(pvals))

        #Because of uncertainty and fluctuation in temperature measurements,
        #not every temperature value / power value combination has data.
        #We want to assign index values in a smart way to get rid of empty combinations


        #Check to make sure that there aren't any rogue extra points that will mess this up
        if (self.smartindex == 'block') and (len(resList) % len(self.pvec) == 0) and (len(resList)>0):
            temptvec = [] #Will add to this as we find good index values

            tindex = 0
            setindices = []
            settemps = []
            for temp in tvec:
                for pwr in self.pvec:
                    curindex = indexResList(resList, temp, pwr)
                    if curindex is not None:
                        setindices.append(curindex)
                        settemps.append(temp)

                if len(setindices) % len(self.pvec) == 0:
                    #For now this switches to mK instead of K because of the
                    #stupid way python handles float division (small errors)
                    itemp = np.round(np.mean(np.asarray(settemps))*1000)
                    temptvec.append(itemp)

                    #Set the indexing temperature of the resonator object
                    for index in setindices:
                        resList[index].itemp = itemp

                    setindices = []
                    settemps = []

            self.tvec = np.asarray(temptvec)
        elif self.smartindex == 'raw':
            for res in resList:
                res.itemp = np.round(res.temp*1000)
            self.tvec = np.round(tvec*1000)
        elif self.smartindex == 'round':
            for index, res in enumerate(resList):
                res.itemp = itvals[index]
            self.tvec = np.sort(np.unique(itvals))
        else:
            self.tvec = np.round(tvec*1000)
            self.smartindex = 'raw'
            for res in resList:
                res.itemp = np.round(res.temp*1000)

        #Loop through the parameters list and create a DataFrame for each one
        for pname in params:
            #Start out with a 2D dataframe full of NaN of type float
            #Row and Column indices are temperature and power values
            self[pname] = pd.DataFrame(np.nan, index = self.tvec, columns = self.pvec)


            if pname in resList[0].params.keys():
                #Uncertainty on best fit from least-squares
                self[pname+'_sigma'] = pd.DataFrame(np.nan, index=self.tvec, columns = self.pvec)

                #Maximum liklihood value from MCMC
                self[pname+'_mle'] = pd.DataFrame(np.nan, index = self.tvec, columns = self.pvec)

                #50th percentile value of MCMC chain
                self[pname+'_mc'] = pd.DataFrame(np.nan, index = self.tvec, columns = self.pvec)

                #84th-50th values from MCMC chain
                self[pname+'_sigma_plus_mc'] = pd.DataFrame(np.nan, index = self.tvec, columns = self.pvec)

                #50th-16th values from MCMC chain
                self[pname+'_sigma_minus_mc'] = pd.DataFrame(np.nan, index = self.tvec, columns = self.pvec)


            #Fill it with as much data as exists
            for index, res in enumerate(resList):
                if pname in res.lmfit_result.params.keys():
                    if res.lmfit_result.params[pname].vary is True:
                        #The actual best fit value
                        self[pname][res.pwr][res.itemp] = res.lmfit_result.params[pname].value

                        #Get the right index to find the uncertainty in the covariance matrix
                        cx = res.lmfit_result.var_names.index(pname)

                        #The uncertainty is the sqrt of the autocovariance
                        if res.lmfit_result.covar is not None:
                            self[pname+'_sigma'][res.pwr][res.itemp] = np.sqrt(res.lmfit_result.covar[cx, cx])

                        #Get the maximum liklihood if it exists
                        if res.hasChain is True:

                            #Grab the index of the parameter in question
                            sx = list(res.emcee_result.flatchain.iloc[np.argmax(res.emcee_result.lnprob)].keys()).index(pname)
                            self[pname+'_mle'][res.pwr][res.itemp] = res.mle_vals[pname]
                            self[pname+'_mc'][res.pwr][res.itemp] = res.emcee_result.params[pname].value

                            #Since the plus and minus errorbars can be different,
                            #have to store them separately
                            self[pname+'_sigma_plus_mc'][res.pwr][res.itemp] = res.emcee_sigmas[sx][0]
                            self[pname+'_sigma_minus_mc'][res.pwr][res.itemp] = res.emcee_sigmas[sx][1]
                elif pname == 'temps':
                    #Since we bin the temps by itemp for indexing, store the actual temp here
                    self[pname][res.pwr][res.itemp] = res.temp
                elif pname == 'fmin':
                    self[pname][res.pwr][res.itemp] = res.fmin
                elif pname == 'chisq':
                    self[pname][res.pwr][res.itemp] = res.lmfit_result.chisqr
                elif pname == 'redchi':
                    self[pname][res.pwr][res.itemp] = res.lmfit_result.redchi
                elif pname == 'feval':
                    self[pname][res.pwr][res.itemp] = res.lmfit_result.nfev
                elif pname == 'listIndex':
                    #This is useful for figuring out where in the resList the data you care about is
                    self[pname][res.pwr][res.itemp] = index
                elif pname == 'q0':
                    qi = res.lmfit_result.params['qi'].value
                    qc = res.lmfit_result.params['qc'].value
                    self[pname][res.pwr][res.itemp] = qi*qc/(qi+qc)

    def do_lmfit(self, fit_keys, models_list, params_list, model_kwargs=None, param_kwargs=None, **kwargs):
        r"""Run simulatneous fits on the temp/pwr data for several parameters.
        Results are stored in either the ``lmfit_results`` or
        ``lmfit_joint_results`` attribute depending on whether one or multiple
        keys are passed to `fit_keys`.

        Parameters
        ----------
        fit_keys : list-like
            A list of keys that correspond to existing data. Any combination of
            keys from `self.keys()`` is acceptable, but duplicates are not
            permitted.

        models_list : list-like
            A list of fit functions, one per key in `fit_keys`. Function must
            return a residual of the form: ``residual = (model-data)/sigma``
            where ``residual``, ``model``, and ``data`` are all ``numpy``
            arrays. Function signature is ``model_func(params, temps, powers,
            data=None, sigmas=None)``. If ``data==None`` the functions must
            return the model calculated at ``temps`` and ``powers``. The model
            functions should also gracefully handle ``np.NaN`` or ``None``
            values.

        params_list : list-like
            A list of ``lmfit.Parameters`` objects, one for each key in
            `fit_keys`. Parameters sharing the same name will be merged so that
            the fit is truly joint. Alternately, a list of functions that return
            ``lmfit.Parameters`` objects may be passed. In this case, one should
            use `param_kwargs` to pass any needed options to the functions.

        model_kwargs : list-like (optional)
            A list of ``dict`` objects to pass to the individual model functions
            as kwargs. ``None`` is also an acceptable entry  if there are no
            kwargs to pass to a model function. Default is ``None.``

        param_kwargs : list-like (optional)
            A list of ``dict`` objects to pass to the individual params
            functions as kwargs. ``None`` is also an acceptable entry  if
            there are no kwargs to pass to a model function. Default is
            ``None.``

        lmfit_kwargs : dict (optional)
            Keyword arguments to pass options to the fitter

        kwargs : dict (optional)
            Supported keyword arugments are 'min_temp', 'max_temp', 'min_pwr',
            and 'max_pwr'. These set limits on which data to fit.

        Keyword arguments
        -----------------

        min_temp : numeric
            Lower limit of temperature to fit. Default is 0.

        max_temp : numeric
            Upper limit of temerature to fit. Default is infinity.

        min_pwr : numeric
            Lower limit of temperature to fit. Default is -infinity.

        max_pwr : numeric
            Upper limit of temperature to fit. Default is infinity.

        raw_data : string {'lmfit', 'emcee', 'mle'}
            Whether to use the values returned by lmfit, or the values returned
            by the emcee fitter (either the 50th percentile or the maximum
            liklihood). This also chooses which set of errorbars to use: either
            those from the lmfit covariance matrix, or those from the 16th and
            84th percentiles of the posterior probablility distribution. Default
            is 'lmfit'.

        Note
        ----
        If the fits are succesful, the resulting fit data (ie the best fit
        surface) will be added to the self dict in the form of a
        ``pandas.DataFrame`` under the following keys:

        For a joint fit (``len(fit_keys) > 1``)::

            'lmfit_joint_'+joint_key+'_'+key for each key in fit_keys

        For a single fit (``len(fit_keys) == 1``)::

            'lmfit_'+key

        """



        #Set some limits
        min_temp = kwargs.pop('min_temp', min(self.tvec))
        max_temp = kwargs.pop('max_temp', max(self.tvec))
        t_filter = (self.tvec >= min_temp) * (self.tvec <= max_temp)

        min_pwr = kwargs.pop('min_pwr', min(self.pvec))
        max_pwr = kwargs.pop('max_pwr', max(self.pvec))
        p_filter = (self.pvec >= min_pwr) * (self.pvec <= max_pwr)

        #Process the final kwarg:
        raw_data = kwargs.pop('raw_data', 'lmfit')
        assert raw_data in ['lmfit', 'emcee', 'mle'], "raw_data must be 'lmfit' or 'emcee'."



        assert len(fit_keys) == len(models_list) == len(params_list), "Make sure argument lists match in number."

        #Make some empty dictionaries just in case
        if model_kwargs is None:
            model_kwargs = [{}]*len(fit_keys)

        if param_kwargs is None:
            params_kwargs = [{}]*len(fit_keys)


        #Check to see if this should go in the joint_fits dict, and build a key if needed.
        if len(fit_keys) > 1:
            joint_key = '+'.join(fit_keys)
        else:
            joint_key = None

        #Check if params looks like a lmfit.Parameters object.
        #If not, assume is function and try to set params by calling it
        for px, p in enumerate(params_list):
            if not hasattr(p, 'valuesdict'):
                assert params_kwargs[px] is not None, "If passing functions to params, must specfify params_kwargs."
                params_list[px] = p(**param_kwargs[px])

        #Combine the different params objects into one large list
        #Only the first of any duplicates will be transferred
        merged_params = lf.Parameters()
        if len(params_list) > 1:
            for p in params_list:
                for key in p.keys():
                    if key not in merged_params.keys():
                        merged_params[key] = p[key]
        else:
            merged_params = params_list[0]

        #Get all the possible temperature/power combos into two grids
        ts, ps = np.meshgrid(self.tvec[t_filter], self.pvec[p_filter])

        #Create grids to hold the fit data and the sigmas
        fit_data_list = []
        fit_sigmas_list = []

        #Get the data that corresponds to each temperature power combo and
        #flatten it to match the ts/ps combinations
        #Transposing is important because numpy matrices are transposed from
        #Pandas DataFrames
        for key in fit_keys:

            if raw_data == 'emcee':
                key = key + '_mc'
            elif raw_data == 'mle':
                key = key + '_mle'

            if raw_data in ['emcee', 'mle']:
                err_bars = (self[key+'_sigma_plus_mc'].loc[t_filter, p_filter].values.T+
                            self[key+'_sigma_minus_mc'].loc[t_filter, p_filter].values.T)
            else:
                err_bars = self[key+'_sigma'].loc[t_filter, p_filter].values.T

            fit_data_list.append(self[key].loc[t_filter, p_filter].values.T)
            fit_sigmas_list.append(err_bars)

        #Create a new model function that will be passed to the minimizer.
        #Basically this runs each fit and passes all the residuals back out
        def model_func(params, models, ts, ps, data, sigmas, kwargs):
            residuals = []
            for ix, key in enumerate(fit_keys):
                residuals.append(models[ix](params, ts, ps, data[ix], sigmas[ix], **kwargs[ix]))

            return np.asarray(residuals).flatten()


        #Create a lmfit minimizer object
        minObj = lf.Minimizer(model_func, merged_params, fcn_args=(models_list, ts, ps, fit_data_list, fit_sigmas_list, model_kwargs))

        #Call the lmfit minimizer method and minimize the residual
        lmfit_result = minObj.minimize(method = 'leastsq')

        #Put the result in the appropriate dictionary
        if joint_key is not None:
            self.lmfit_joint_results[joint_key] = lmfit_result
        else:
            self.lmfit_results[fit_keys[0]] = lmfit_result

        #Calculate the best-fit model from the params returned
        #And put it into a pandas DF with the appropriate key.
        #The appropriate key format is: 'lmfit_joint_'+joint_key+'_'+key
        #or, for a single fit: 'lmfit_'+key
        for ix, key in enumerate(fit_keys):
            #Call the fit model without data to have it return the model
            returned_model = models_list[ix](lmfit_result.params, ts, ps)

            #Build the appropriate key
            if joint_key is not None:
                new_key = 'lmfit_joint_'+joint_key+'_'+key
            else:
                new_key = 'lmfit_'+key

            #Make a new dict entry to the self dictioary with the right key.
            #Have to transpose the matrix to turn it back into a DF
            self[new_key] = pd.DataFrame(np.nan, index=self.tvec, columns=self.pvec)
            self[new_key].loc[self.tvec[t_filter], self.pvec[p_filter]] = returned_model.T




    def do_emcee(self, fit_keys, models_list, params_list=None, model_kwargs=None, param_kwargs=None, emcee_kwargs=None, **kwargs):
        r"""Run simulatneous MCMC sampling on the temp/pwr data for several
        parameters. Results are stored in either the ``emcee_results`` or
        ``emcee_joint_results`` attribute depending on whether one or multiple
        keys are passed to `fit_keys`.

        Parameters
        ----------
        fit_keys : list-like
            A list of keys that correspond to existing data. Any combination of
            keys from `self.keys()`` is acceptable, but duplicates are not
            permitted.

        models_list : list-like
            A list of fit functions, one per key in `fit_keys`. Function must
            return a residual of the form: ``residual = (model-data)/sigma``
            where ``residual``, ``model``, and ``data`` are all ``numpy``
            arrays. Function signature is ``model_func(params, temps, powers,
            data=None, sigmas=None)``. If ``data==None`` the functions must
            return the model calculated at ``temps`` and ``powers``. The model
            functions should also gracefully handle ``np.NaN`` or ``None``
            values.

        params_list : list-like
            A list of ``lmfit.Parameters`` objects, one for each key in
            `fit_keys`. Parameters sharing the same name will be merged so that
            the fit is truly joint. Alternately, a list of functions that return
            ``lmfit.Parameters`` objects may be passed. In this case, one should
            use `param_kwargs` to pass any needed options to the functions.
            Default is ``None`` and is equivalent to setting ``use_lmfit_params =
            True``.

        model_kwargs : list-like (optional)
            A list of ``dict`` objects to pass to the individual model functions
            as kwargs. ``None`` is also an acceptable entry  if there are no
            kwargs to pass to a model function. Default is ``None.``

        param_kwargs : list-like (optional)
            A list of ``dict`` objects to pass to the individual params
            functions as kwargs. ``None`` is also an acceptable entry  if
            there are no kwargs to pass to a model function. Default is
            ``None.``

        emcee_kwargs : dict (optional)
            Keyword arguments to pass options to the fitter

        Keyword Arguments
        -----------------
        min_temp : numeric
            Lower limit of temperature to fit. Default is 0.

        max_temp : numeric
            Upper limit of temerature to fit. Default is infinity.

        min_pwr : numeric
            Lower limit of temperature to fit. Default is -infinity.

        max_pwr : numeric
            Upper limit of temperature to fit. Default is infinity.

        use_lmfit_params : bool
            Whether or not to use the resulting best-fit ``lmfit.Paramters``
            object that resulted from calling ``ResonatorSweep.do_lmfit()`` as
            the starting value for the MCMC sampler. Default is True.

        raw_data : string {'lmfit', 'emcee', 'mle'}
            Whether to use the values returned by lmfit, or the values returned
            by the emcee fitter (either the 50th percentile or the maximum
            liklihood). This also chooses which set of errorbars to use: either
            those from the lmfit covariance matrix, or those from the 16th and
            84th percentiles of the posterior probablility distribution. Default
            is 'lmfit'.

        Note
        ----
        If the fits are succesful, the resulting fit data (ie the best fit
        surface) will be added to the self dict in the form of a
        ``pandas.DataFrame`` under the following keys:

        For a joint fit (``len(fit_keys) > 1``)::

            'emcee_joint_'+joint_key+'_'+key for each key in fit_keys

        For a single fit (``len(fit_keys) == 1``)::

            'emcee_'+key

        """

        #Figure out which data to fit
        raw_data = kwargs.pop('raw_data', 'lmfit')
        assert raw_data in ['lmfit', 'emcee', 'mle'], "raw_data must be 'lmfit' or 'emcee'."



        #Set some limits
        min_temp = kwargs.pop('min_temp', min(self.tvec))
        max_temp = kwargs.pop('max_temp', max(self.tvec))
        t_filter = (self.tvec >= min_temp) * (self.tvec <= max_temp)

        min_pwr = kwargs.pop('min_pwr', min(self.pvec))
        max_pwr = kwargs.pop('max_pwr', max(self.pvec))
        p_filter = (self.pvec >= min_pwr) * (self.pvec <= max_pwr)



        if params_list is not None:
            assert len(fit_keys) == len(models_list) == len(params_list), "Make sure argument lists match in number."
        else:
            assert len(fit_keys) == len(models_list), "Make sure argument lists match in number."

        #Make some empty dictionaries just in case so we don't break functions
        #by passing None as a kwargs
        if model_kwargs is None:
            model_kwargs = [{}]*len(fit_keys)

        if param_kwargs is None:
            params_kwargs = [{}]*len(fit_keys)

        if emcee_kwargs is None:
            emcee_kwargs = {}


        #Check to see if this should go in the joint_fits dict, and build a key if needed.
        if len(fit_keys) > 1:
            joint_key = '+'.join(fit_keys)
        else:
            joint_key = None


        #If possible (and desired) then we should use the existing best fit as a starting point
        #For the MCMC sampling. If not, build params from whatever is passed in.
        use_lmfit_params = kwargs.pop('use_lmfit_params', True)

        if (params_list is not None) and (use_lmfit_params == False):

            #Check if params looks like a lmfit.Parameters object.
            #If not, assume is function and try to set params by calling it
            for px, p in enumerate(params_list):
                if not hasattr(p, 'valuesdict'):
                    assert params_kwargs[px] is not None, "If passing functions to params, must specfify params_kwargs."
                    params_list[px] = p(**param_kwargs[px])

            #Combine the different params objects into one large list
            #Only the first of any duplicates will be transferred
            merged_params = lf.Parameters()
            if len(params_list) > 1:
                for p in params_list:
                    for key in p.keys():
                        if key not in merged_params.keys():
                            merged_params[key] = p[key]
            else:
                merged_params = params_list[0]

        else:
            if joint_key is not None:
                assert joint_key in self.lmfit_joint_results.keys(), "Can't use lmfit params. They don't exist."
                merged_params = self.lmfit_joint_results[joint_key].params
            else:
                assert fit_keys[0] in self.lmfit_results.keys(), "Can't use lmfit params. They don't exist."
                merged_params = self.lmfit_results[fit_keys[0]].params


        #Get all the possible temperature/power combos into two grids
        ts, ps = np.meshgrid(self.tvec[t_filter], self.pvec[p_filter])

        #Create grids to hold the fit data and the sigmas
        fit_data_list = []
        fit_sigmas_list = []

        #Get the data that corresponds to each temperature power combo and
        #flatten it to match the ts/ps combinations
        #Transposing is important because numpy matrices are transposed from
        #Pandas DataFrames
        for key in fit_keys:

            if raw_data == 'emcee':
                key = key + '_mc'
            elif raw_data == 'mle':
                key = key + '_mle'

            if raw_data in ['emcee', 'mle']:
                err_bars = (self[key+'_sigma_plus_mc'].loc[t_filter, p_filter].values.T+
                            self[key+'_sigma_minus_mc'].loc[t_filter, p_filter].values.T)
            else:
                err_bars = self[key+'_sigma'].loc[t_filter, p_filter].values.T

            fit_data_list.append(self[key].loc[t_filter, p_filter].values.T)
            fit_sigmas_list.append(err_bars)

        #Create a new model function that will be passed to the minimizer.
        #Basically this runs each fit and passes all the residuals back out
        def model_func(params, models, ts, ps, data, sigmas, kwargs):
            residuals = []
            for ix, key in enumerate(fit_keys):
                residuals.append(models[ix](params, ts, ps, data[ix], sigmas[ix], **kwargs[ix]))

            return np.asarray(residuals).flatten()


        #Create a lmfit minimizer object
        minObj = lf.Minimizer(model_func, merged_params, fcn_args=(models_list, ts, ps, fit_data_list, fit_sigmas_list, model_kwargs))

        #Call the lmfit minimizer method and minimize the residual
        emcee_result = minObj.emcee(**emcee_kwargs)

        #Put the result in the appropriate dictionary
        if joint_key is not None:
            self.emcee_joint_results[joint_key] = emcee_result
        else:
            self.emcee_results[fit_keys[0]] = emcee_result

        #Calculate the best-fit model from the params returned
        #And put it into a pandas DF with the appropriate key.
        #The appropriate key format is: 'lmfit_joint_'+joint_key+'_'+key
        #or, for a single fit: 'lmfit_'+key
        for ix, key in enumerate(fit_keys):
            #Call the fit model without data to have it return the model
            returned_model = models_list[ix](emcee_result.params, ts, ps)

            #Build the appropriate key
            if joint_key is not None:
                new_key = 'emcee_joint_'+joint_key+'_'+key
            else:
                new_key = 'emcee_'+key

            #Make a new dict entry to the self dictioary with the right key.
            #Have to transpose the matrix to turn it back into a DF
            self[new_key] = pd.DataFrame(np.nan, index=self.tvec, columns=self.pvec)
            self[new_key].loc[self.tvec[t_filter], self.pvec[p_filter]] = returned_model.T

    def info():
        """Print out some information on all the keys that are stored in the object."""

        #For now, this just spits out all the keys. Could be more useful.
        print(sorted(self.keys()))
