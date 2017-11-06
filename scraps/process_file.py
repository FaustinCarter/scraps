import numpy as np

def process_file(fileName, **loadtxt_kwargs):
    """Load Keysight PNA file data into dict.

    Parameters
    ----------
    fileName : string
        Path to data. Actual name of the file should be in the format:
        ``name + '_' + pwr + '_DBM_TEMP_' + temp + '.S2P'`` where:

        - name: Must be 'RES-N' where N is any single character you like.

        - pwr: The power the resonator sees in dBm, three characters Example: '-25'

        - temp: The temperature of the resonator in K, five characters Example: '0.150'

    loadtxt_kwargs : dict
        This is a pass-through to numpy.loadtxt

    Returns
    -------
    dataDict : dict or ``None``
        Dictionary contains the following keys: 'name', 'temp', 'pwr', 'freq',
        'I', 'Q'. If fileName does not exist, then returns ``None``.

    Note
    ----
    This assumes the data in the file is in three columns in the order frequency, I, Q.

    This is also a terribly written function, and you really should write your own!
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
        fileData = np.loadtxt(fileName, **loadtxt_kwargs)
        freqData = fileData[:,0]
        IData = fileData[:,1]
        QData = fileData[:,2]

        dataDict = {'name':resName,'temp':temp,'pwr':pwr,'freq':freqData,'I':IData,'Q':QData}
        return dataDict
    else:
        return None
