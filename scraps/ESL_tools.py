from .resonator import Resonator, makeResFromData
import numpy as np

def load_one_ESL(dataFile, sweepType='rough', resNum = 0, **kwargs):
    """Load a single resonator file from ESL into a data dict for later processing.

    Return value:
        dataDict -- a dict object with six fields
            freq, I, Q, temp, pwr, name

    Arguments:
        dataFile -- path to data
        sweepType -- 'rough', 'gain', or 'fine'. Default is 'rough'
        resNum -- the number of the resonator in the file you want to load
            indexing starts at zero

    Keyword arguments:
        'temp' -- float
        'units' -- 'mK' or 'K' depending on how you enter 'temp'
        'name' -- string. Defaults to 'RES-X'
        'legacy' -- bool swaps I and Q values if you have an old file"""
    #Set some default values
    unitString = kwargs.pop('units', 'mK')
    if unitString == 'mK':
        units = 0.001
    elif unitString == 'K':
        units = 1
    else:
        raise ValueError("Unknown unit type. Acceptable values are 'mK' or 'K'.")
    temp = kwargs.pop('temp', np.NAN)
    name = kwargs.pop('name', 'RES-X')
    legacy = kwargs.pop('legacy', False)

    if kwargs:
        raise ValueError("Unknown keyword argument passed: " + kwargs.keys()[0])

    #If you change your default sweep types, update this dict here!
    sweepDict = {'gain':201,
                'rough':401,
                'fine':1601}

    #Open file and read power from header data
    with open(dataFile) as fp:
        for i, line in enumerate(fp):
            if i == 2:
                pwr = int(line[line.find(':')+1:line.find('.')])
            elif i > 2:
                break

    #Load in the all I,Q,freq data
    cdata = np.loadtxt(dataFile, delimiter=',',skiprows=6)

    #Slice out the data from the resonator you want to look at
    numPts = sweepDict[sweepType]
    startData = resNum*numPts
    endData = (resNum+1)*numPts - 1

    #Dump everything into the dataDict
    dataDict = {}
    dataDict['freq'] = cdata[startData:endData,0]

    #Load in the IQ data
    #Old VNA programs reversed I and Q, so call legacy if trying to load those
    if legacy:
        dataDict['Q'] = cdata[startData:endData,1]
        dataDict['I'] = cdata[startData:endData,2]
    else:
        dataDict['I'] = cdata[startData:endData,1]
        dataDict['Q'] = cdata[startData:endData,2]
    ###

    dataDict['temp'] = temp*units
    dataDict['pwr'] = pwr
    dataDict['name'] = name

    return dataDict #Suitable for use with pyres.makeResFromData()



def load_sweep_ESL(dataFolder, resNames, tvals, pwrs, **kwargs):
    """Load a set of ESL resonator data corresponding to a power/temperature sweep.

    Return value:
        resList -- a dict of lists of Resonator objects, indexed by resonator name.

    Arguments:
        resNames -- a list of string identifiers. len(resNames) == number of individual resonators you scanned over
        dataFolder -- path to folder containing data, including final slash
        tvals -- a list of temperature values in the same order as the sweepXX folders
            Example: Two different temperatures
                dataFolder/
                    sweep00/ <--- 2K
                    sweep01/ <--- 100 mK

                so tval = [2., 0.1] and units='K' OR tval = [2000., 100] and units = 'mK'
        pwrs -- a list of power values in the same order as the VNAPnXX identifier in the filenames

    Keyword arguments:
        'units' -- 'mK' or 'K' depending on how you enter 'temp'
        'sweep' -- 'gain', 'rough', or 'fine'. Default is 'rough'."""
    #Set up some default values

    unitString = kwargs.pop('units', 'mK')
    if unitString == 'mK':
        units = 0.001
    elif unitString == 'K':
        units = 1
    else:
        raise ValueError("Unknown unit type. Acceptable values are 'mK' or 'K'.")


    #Change this if you change the types of sweep that exist in the program
    sweepDict = {'gain':201,
                'rough':401,
                'fine':1601}

    sweepType = kwargs.pop('sweep', 'rough')
    assert sweepType in sweepDict.keys(), "Invalid sweep type. Try 'gain', 'rough', or 'fine'."

    #If loading old data, swaps I and Q
    legacy = kwargs.pop('legacy', False)

    #Make sure no unknown keywords arguments
    if kwargs:
        raise ValueError("Unknown keyword argument passed: " + kwargs.keys()[0])

    #Declare some empty containers
    dataDicts = {}
    resLists = {}
    for resName in resNames:
        dataDicts[resName] = []
        resLists[resName] = []

    #Loop through all the directories and pull out the data
    for indexp, pwr in enumerate(pwrs):
        seriesNum = str(indexp).zfill(2)

        for indext, tval in enumerate(tvals):
            folderNum = str(indext+1).zfill(2)
            cdata = np.loadtxt(dataFolder+'sweep'+folderNum+'/'+sweepType+'VNAPn'+seriesNum+'set000.txt', delimiter=',',skiprows=6)

            for indexr, resName in enumerate(resNames):
                numPts = sweepDict[sweepType]
                startData = indexr*numPts
                endData = (indexr+1)*numPts - 1

                dataDict = {}
                dataDict['freq'] = cdata[startData:endData,0]

                if legacy:
                    dataDict['Q'] = cdata[startData:endData,1]
                    dataDict['I'] = cdata[startData:endData,2]
                else:
                    dataDict['I'] = cdata[startData:endData,1]
                    dataDict['Q'] = cdata[startData:endData,2]
                ###

                dataDict['temp'] = tval*units
                dataDict['pwr'] = pwr
                dataDict['name'] = resName

                dataDicts[resName].append(dataDict)

    #Convert the dataDicts into Resonator objects and return a dict of lists of Resonators
    for resName in resNames:
        resLists[resName] = [makeResFromData(dataDict) for dataDict in dataDicts[resName]]

    return resLists #Suitable for passing to ResonatorSweep
