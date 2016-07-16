from __future__ import division
import pandas as pd
import numpy as np
import glob
import matplotlib.pyplot as plt
from .pyRes import makeResFromData
from .process_file import process_file

#This is a glorified dictionary with a custom initalize method. It takes a list
#of resonator objects that have been fit, and then sets up a
#dict of pandas DataFrame objects, one for each interesting fit parameter
#This adds no new information, but makes accessing the fit data easier
class ResonatorSweep(dict):
    """Dictionary object with custom __init__ method"""


    def __init__(self, resList, **kwargs):
        """Formats various scalar quantities into easily parsed pandas DataFrame objects.

        Arguments:
        self -- reference to self, required for all Class methods
        resList -- a list of Resonator objects

        Created quantities:
        self.tvec -- index of temperature values
        self.pvec -- index of power values

        Note: Temperature data is binned into 5 mK spaced bins for compactness.
        Actual temperature value is stored in the 'temps' field."""

        #Start with the list of fit parameters, want to save all of them
        #Can just use the first resonator's list, as they are all the same.
        #params is NOT an lmfit object.
        params = resList[0].params.keys()

        #Add a few more
        params.append('temps') #Actual temperature value of measured resonator
        params.append('fmin') #Frequency at magnitude minimum
        params.append('chisq') #Chi-squared value from fit
        params.append('redchi') #Reduced chi-squared value
        params.append('feval') #Number of function evaluations to converge on fit
        params.append('listIndex') #Index in resonator list of resonator

        #This flag sets different indexing methods
        self.smartindex = 'raw'

        #Default the rounding option to 5 mK
        self.roundto = 5

        #Possible values are 'raw', 'round', 'block'
        #If round, also must pass 'roundto'
        if kwargs is not None:
            for key, val in kwargs.iteritems():
                if key == 'index':
                    self.smartindex = val

                if key == 'roundto':
                    #Value in mK to round temperature index to
                    self.roundto = val

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

            #Fill it with as much data as exists
            for index, res in enumerate(resList):
                if pname in res.lmfit_result.params.keys():
                    self[pname][res.pwr][res.itemp] = res.lmfit_result.params[pname].value
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
                    self[pname][res.pwr][res.itemp] = index

    def plotParamsVsTemp(self, **kwargs):
        #This will really only work for sure if block is sucessful
        if self.smartindex == 'block':
        #To do: fix for other smartindex types

            #set defaults
            numCols = 4
            powers = self.pvec
            keysToIgnore = ['Ioffset',
                            'Qoffset',
                            'listIndex',
                            'temps',
                            'chisq']

            if kwargs is not None:
                for key, val in kwargs.iteritems():
                    if key == 'numCols':
                        numCols = int(val)
                    elif key == 'powers':
                        assert all([p in powers for p in val]), "Can't plot a power that doesn't exist!"
                        powers = list(val)
                    elif key == 'keysToIgnore':
                        assert all([key in self.keys() for key in val]), "Key does not exist!"
                        keysToIgnore = list(val)

            #Set up the figure
            figS = plt.figure()

            numKeys = len(self.keys())-len(keysToIgnore)
            numRows = int(np.ceil(numKeys/numCols))

            #Magic numbers!
            figS.set_size_inches(6*numCols,6*numRows)

            #Loop through all the keys in the ResonatorSweep object and plot them
            indexk = 1
            for key in set(self.keys())-set(keysToIgnore):
                axs = figS.add_subplot(numRows,numCols,indexk)
                for pwr in powers:
                    axs.plot(self.tvec,self[key][pwr],'--',label='Power: '+str(pwr))

                axs.set_xlabel('Temperature (mK)')
                axs.set_ylabel(key)

                #Stick some legends where they won't crowd too much
                if key == 'f0' or key == 'fmin':
                    axs.legend(loc='best')

                indexk += 1
            return figS
        else:
            return None

def makeResList(fileFunc, dataPath, resName):
    """Create a list of resonator objects from a directory of dataDict

    Returns:
    resList -- a list of Resonator objects

    Arguments:
    fileFunc -- the function that converts files into a data dictionary
    dataPath -- path to the directory holding the data
    resName -- the name of the resonator you want to pull data from"""
    #Find the files that match the resonator you care about
    fileList = glob.glob(dataPath + '*' + resName + '_*' + '*')

    #loop through files and process all the data
    fileDataDicts = map(fileFunc, fileList)

    #Strip out any bad files
    fileDataDicts = [fileDataDict for fileDataDict in fileDataDicts if (fileDataDict is not None)]

    #Create resonator objects from the data
    #makeResFromData returns a tuple of (res, temp, pwr),
    #but only care about the first one
    resList = [makeResFromData(fileDataDict)[0] for fileDataDict in fileDataDicts]

    return resList

#Index a list of resonator objects easily
def indexResList(resList, temp, pwr, **kwargs):
    """Index resList by temp and pwr.

    Returns:
    index -- an int corresponding to the location of the Resonator specified by the Arguments

    Arguments:
    resList -- a list of Resonator objects
    temp -- the temperature of a single Resonator object
    pwr -- the power of a single Resonator object

    Keyword Args:
    itemp -- boolean switch to determine whether lookup uses temp or itemp (rounded value of temp)

    Note:
    The combination of temp and pwr must be unique. indexResList does not check for duplicates."""
    itemp = False
    if kwargs is not None:
        for key, val in kwargs.iteritems():
            if key == 'itemp':
                itemp = val


    for index, res in enumerate(resList):
        if itemp is True:
            if res.itemp == temp and res.pwr == pwr:
                return index
        else:
            if res.temp == temp and res.pwr == pwr:
                return index

    return None
