import numpy as np

def process_file(fileName):
    """Load Keysight PNA file data into dict.

    Parameters
    ----------
    fileName : string
        Path to data.

    Returns
    -------
    dataDict : dict or ``None``
        Dictionary contains the following keys: 'name', 'temp', 'pwr', 'freq',
        'I', 'Q'. If fileName does not exist, then returns ``None``.

    """
    #Find the temperature, power, and name locations from the filename
    tempLoc = fileName.find('TEMP') + 5
    pwrLoc = fileName.find('DBM') - 4
    resNameLoc = fileName.find('RES-')

    #Read the temp, pwr, and resName from the filename
    if(fileName[tempLoc + 6] != '.'):
        temp = np.float(fileName[tempLoc:tempLoc+5])

        if fileName[pwrLoc] == '_':
            pwr = np.float(fileName[pwrLoc+1:pwrLoc+3])
        else:
            pwr = np.float(fileName[pwrLoc:pwrLoc+3])

        resName = fileName[resNameLoc:resNameLoc+5]

        #Grab frequency, I, and Q
        fileData = np.loadtxt(fileName, skiprows=1)
        freqData = fileData[:,0]
        IData = fileData[:,1]
        QData = fileData[:,2]

        dataDict = {'name':resName,'temp':temp,'pwr':pwr,'freq':freqData,'I':IData,'Q':QData}
        return dataDict
    else:
        return None
