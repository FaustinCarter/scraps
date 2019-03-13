import os

import numpy as np
import pandas as pd
import glob

try: import simplejson as json
except ImportError: import json

from .utils import find_nearest


def process_file(fileName, mask = None, meta_only=False, **loadtxt_kwargs):
    """Load Keysight PNA file data into dict.

    Parameters
    ----------
    fileName : string
        Path to data. Actual name of the file should be in the format:
        ``name + '_' + pwr + '_DBM_TEMP_' + temp + '.S2P'`` where:

        - name: Must be 'RES-N' where N is any single character you like.

        - pwr: The power the resonator sees in dBm, three characters Example: '-25'

        - temp: The temperature of the resonator in K, five characters Example: '0.150'

    mask : slice (optional)
        You can pass a slice to cut out some region of data. Example 1: mask = slice(10,-10) will cut out the first 10 and last 10 datapoints. Example 2: maks = slice(None,None,10) will resample your data and take every 10th point. Example 3: mask = slice(None, -50) will cut the last 50 points off your data.

    meta_only : bool
        Return just the metadata from the filename (True) or all the data (False).
        Default is False.
    
    loadtxt_kwargs : dict (optional)
        This is a pass-through to numpy.loadtxt

    Returns
    -------
    dict or ``None``
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

    if mask is None:
        mask = slice(None, None)
    else:
        assert type(mask) == slice, "mask must be of type slice."

    #Read the temp, pwr, and resName from the filename
    if(fileName[tempLoc + 1] == '.'):
        temp = np.float(fileName[tempLoc:tempLoc+5])

        if fileName[pwrLoc] == '_':
            pwr = np.float(fileName[pwrLoc+1:pwrLoc+3])
        else:
            pwr = np.float(fileName[pwrLoc:pwrLoc+3])

        resName = fileName[resNameLoc:resNameLoc+5]

        metaDict = {'name':resName,'temp':temp,'pwr':pwr}

        if meta_only:
            dataDict = {}
        else:
            #Grab frequency, I, and Q
            fileData = np.loadtxt(fileName, **loadtxt_kwargs)
            freqData = fileData[:,0][mask]
            IData = fileData[:,1][mask]
            QData = fileData[:,2][mask]

            dataDict = {'freq':freqData,'I':IData,'Q':QData}

        retVal = {}
        retVal.update(metaDict)
        retVal.update(dataDict)
        
        return retVal
    else:
        
        assert False, "Bad file? " + fileName

def build_metadf(basepath, meta_style='json'):
    """Build a metadata DataFrame from a directory full of resonator files.

    Paramaters
    ----------
    basepath: string
        Path to the directory containing the data

    meta_style: string
        Type of metadata. Can be 'filename' or 'json'
    
    Return
    ------
    temperature_index : numpy.array
        An index of unique temperature values."""

    metadf = pd.DataFrame(columns=['Temperature', 'Power', 'Name'])

    for ix, filename in enumerate(glob.glob(os.path.join(basepath, '*.S2P'))):
        
        if meta_style == 'json':
            with open(filename, 'r') as f:
                metadata = json.loads(f.readline())
            
            power = float(metadata['input_power'])-float(metadata['internal_atten'])-float(metadata['var_atten'])
            temperature = metadata['temp']
            name = metadata['res_name']

        elif meta_style == 'filename':
            metadata = process_file(filename, meta_only=True)
            power = metadata['pwr']
            temperature = metadata['temp']
            name = metadata['name']
        
        metadf.loc[ix] = [temperature, power, name]

    return metadf

def pivot_metadf_temp(metadf):
    """Pivots the metadf to put temperature as the value. Useful for debugging."""
    grp_metadf = metadf.groupby(['Name', 'Power'])
    length = max(len(v) for k, v in grp_metadf)
    newdf = pd.DataFrame()
    for k, v in metadf.groupby(['Name', 'Power']):
        newdf[k] = np.pad(v.sort_values(by='Temperature')['Temperature'].values,
                          (0,length-len(v)),
                          mode='constant',
                          constant_values=np.NaN)

    return newdf

def get_temperature_index(metadf):
    """Get a temperature index from metadata DataFrame returned by
    build_metadf."""

    num_powers = len(metadf['Power'].unique())
    num_res = len(metadf['Name'].unique())

    #Non-identical values, so have to infer
    num_temps = int(len(metadf)/(num_powers*num_res))

    datacube_shape = (num_powers,
                    num_temps,
                    num_res)

    temperature_index = (np.reshape(metadf.sort_values(by=['Power', 'Temperature'])['Temperature'].values, datacube_shape)
                    .mean(axis=0) #Average along powers
                    .mean(axis=1) #Average along resonators
                    *1000).astype(int)/1000 #Round to nearest mK

    return temperature_index