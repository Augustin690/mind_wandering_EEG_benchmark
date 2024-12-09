import numpy as np
import os
import mne
from mne.time_frequency import tfr_morlet
from mne.stats import permutation_cluster_test
from scipy.signal import hilbert
from scipy.stats import ttest_ind

def computeEEGwave_group(subs, tasks, states, contentsets, chans, measure, type, f_marker, averageOn):
    # Tasks: cell of 'sart', 'vs' or 'merge'
    # Contentsets: cell of contentsets(e.g., 1i, 1c)ordered by the specified STATE order; [] - use defaul state definition
    # Chans: channels used to average data
    # Measure: e.g., 'p3', 'alpha'
    # Type: 'raw' - data from raw matrix; 'marker' - data as features in modelling
    # F_marker: folders containing measurements of markers 
    # AverageOn: 1(default) - to return group mean and se-within, other values - to return individual mean
    # Return is the data for wavegraph with length of npnt
    # Author: Christina Jin (christina.mik109@gmail.com)
    # Last edit: Nov 21, 2018

    # Load parameters
    os.system('ls')  # Assuming 'ls' command is used to navigate to the directory containing 'pars.mat'
    times = np.load('pars.npy', allow_pickle=True)['times']
    
    if not contentsets:
        content_code = np.load('pars.npy', allow_pickle=True)['content_code']
    
    if not averageOn:
        averageOn = 1
    
    if measure.lower() in ['alpha', 'theta']:
        bands = np.load('pars_marker.mat.npy', allow_pickle=True)['bands']
        id = np.where(np.array([band['name'] for band in bands]) == measure)[0][0]
        bandrange = bands[id]['range']
    
    data = []
    for ti, task in enumerate(tasks):
        for si, state in enumerate(states):
            # Get raw data
            if type.lower() == 'raw':
                # Get state definition
                if contentsets:
                    contentset = contentsets[si]
                    print(f'State is {state}. Contents are: {contentset}')
                
                # Loop over subs to get data 
                for subi, sub in enumerate(subs):
                    # Load data
                    if not contentsets:
                        temp = get_data(sub, task, state)
                    else:
                        temp = select_trials2(sub, {'content': contentset})
                    
                    # Compute mean
                    if measure in ['p1', 'n1', 'p3']:
                        temp = np.mean(temp, axis=2) 
                    elif measure in ['theta', 'alpha']: 
                        dataFilt = hilbertFilter(temp, srate, bandrange, 0)
                        power = compute_power(dataFilt) 
                        temp = np.mean(power, axis=2)
                    else:
                        raise ValueError('Invalid measure!')
                    
                    # Register
                    if subi == 0:
                        data.append(np.zeros((temp.shape[0], temp.shape[1], len(subs))))
                    data[-1][:,:,subi] = temp
                    
                # Average over channels
                if type.lower() == 'raw':
                    data[-1] = np.mean(data[-1][chans,:,:], axis=0)
                
                # Prepare for output
                if averageOn:
                    data2plot = np.mean(data[-1], axis=1)
                    if si == 0:
                        data_raw = np.zeros((len(times), len(subs), len(states)))
                    data_raw[:,:,si] = data[-1]
                    cellOutput[(ti-1)*2+si]['data'] = data2plot
                else:
                    cellOutput[(ti-1)*2+si]['data'] = data[-1]
                
                # Generate output
                cellOutput[(ti-1)*2+si]['task'] = task
                cellOutput[(ti-1)*2+si]['state'] = state
                cellOutput[(ti-1)*2+si]['contentset'] = contentset
                cellOutput[(ti-1)*2+si]['subs'] = subs
                cellOutput[(ti-1)*2+si]['chans'] = chans
                cellOutput[(ti-1)*2+si]['measure'] = measure
                cellOutput[(ti-1)*2+si]['type'] = type
                
            # Get recovered data based on marker
            elif type.lower() == 'marker':
                # State definition
                if contentsets:
                    contentset = contentsets[si]
                    print(f'State is {state}. Contents are: {contentset}')
                
                for subi, sub in enumerate(subs):
                    if measure not in ['p1', 'n1', 'p3']:
                        raise ValueError('type = marker can only be set when the specified measure is an ERP!')
                    
                    # Load measures and recover the signal
                    temp = np.zeros((len(times), len(chans)))
                    for chani, chan in enumerate(chans):
                        if not contentsets:
                            temp2 = np.load([f_marker, '\\', measure.lower(), ' chan', str(chan), '\\', str(sub), '.npy'], allow_pickle=True)
                        else:
                            temp2 = np.zeros((len(times), len(contentset)))
                            for ci, content in enumerate(contentset):
                                if not task.lower() == 'merge':
                                    temp2[:,ci] = np.load([f_marker, '\\', measure.lower(), ' chan', str(chan), '\\', str(sub), '.npy'], allow_pickle=True)[content]
                                else:
                                    temp2[:,ci] = np.load([f_marker, '\\', measure.lower(), ' chan', str(chan), '\\', str(sub), '.npy'], allow_pickle=True)[content]
                        temp += temp2
                    
                    # Compute idv mean 
                    temp = np.mean(temp, axis=1)
                    
                    # Register
                    if subi == 0:
                        data.append(np.zeros((len(times), len(subs))))
                    data[-1][:,subi] = temp
                                    
                # Prepare for output
                if averageOn:
                    data2plot = np.mean(data[-1], axis=1)
                    if si == 0:
                        data_raw = np.zeros((len(times), len(subs), len(states)))
                    data_raw[:,:,si] = data[-1]
                    cellOutput[(ti-1)*2+si]['data'] = data2plot
                else:
                    cellOutput[(ti-1)*2+si]['data'] = data[-1]
                
                # Generate output
                cellOutput[(ti-1)*2+si]['task'] = task
                cellOutput[(ti-1)*2+si]['state'] = state
                cellOutput[(ti-1)*2+si]['contentset'] = contentset
                cellOutput[(ti-1)*2+si]['subs'] = subs
                cellOutput[(ti-1)*2+si]['chans'] = chans
                cellOutput[(ti-1)*2+si]['measure'] = measure
                cellOutput[(ti-1)*2+si]['type'] = type
                
            else:
                raise ValueError('Invalid measure!')
    
    if averageOn:
        # Compute for se
        data_corr = data_raw + np.mean(data_raw, axis=1)[:,np.newaxis,:] - np.mean(data_raw, axis=2)[np.newaxis,:,:]
        for ti in range(len(tasks)):
            for si in range(len(states)):
                cellOutput[(ti-1)*2+si]['sewithin'] = np.std(data_corr[:, :, (ti-1)*2+si], axis=1) / np.sqrt(len(subs)-1)
    return cellOutput
