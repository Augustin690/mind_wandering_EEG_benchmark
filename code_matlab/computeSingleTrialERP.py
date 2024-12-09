import numpy as np
import os
import mne
from mne.io import read_raw_edf
from mne.time_frequency import tfr_morlet
from mne.stats import permutation_cluster_test
from mne.viz import plot_evoked
import matplotlib.pyplot as plt
import pandas as pd
import scipy

def compute_single_trial_erp(erp, subs, plot_on, condset, f_output):
    # Compute W matrix and do local peak search to find the single trial ERP of interest. 
    # Outputs are based on invididuals x conditions x tasks. 
    # Note: the W in output file is integrated in the unit of sampling points to save computing time; the real W should be adjusted by W * 1000/srate.
    # Author: Christina Jin (christina.mik109@gmail.com)

    # Load necessary data
    os.system('ls')  # Assuming the data is in the current directory
    times = scipy.io.loadmat('pars.mat')['times']
    srate = scipy.io.loadmat('pars.mat')['srate']
    erps = scipy.io.loadmat('pars_marker.mat')['erps']

    # Set default output file
    if f_output is None or f_output == '':
        f_output = 'measure_matfile'

    # Load condition set if provided
    if condset is None:
        print('Computing based on default conditions...')
        triggersets = pd.read_csv('pars.mat', usecols=['triggersets'])
        tasks = pd.read_csv('pars.mat', usecols=['tasks'])
        states = pd.read_csv('pars.mat', usecols=['states'])
        default_on = 1
        factors = ['state']
        levels = [states]
    else:
        default_on = 0
        triggersets = condset['triggersets']
        tasks = condset['tasks']
        factors = condset['factors']
        levels = condset['levels'] 

    # Get the ERP definition
    id = np.where(erps['name'] == erp)[0][0]
    measure = erps['name'][id]
    x2search_lower = erps['x2searchLower'][id] 
    x2search_upper = erps['x2searchUpper'][id] 
    y_rng2search = erps['yRng2search'][id]
    sign = erps['sign'][id]
    chans = erps['chans'][id]
    boundary_type = erps['boundaryType'][id]  # 'float': [xLower RT]; 'fixed': [xLower xUpper]; 'adjusted': [xLower RT-->xUpper]

    # Calculate level counts
    nlevel = [len(level) for level in levels]  # vector of level counts

    # Get level combinations
    if len(nlevel) > 1:
        combos = np.zeros((np.prod(nlevel), len(nlevel)))  # row = combo, column = factor, val = id in each factor
        i = len(nlevel)
        while i > 0:
            if i == len(nlevel):
                combos[:,i-1] = np.tile(np.arange(1, nlevel[i-1]+1), len(combos)/nlevel[i-1])
            else:
                temp = np.tile(np.arange(1, nlevel[i-1]+1), np.prod(nlevel[i:]))
                combos[:,i-1] = np.repeat(temp, len(combos)/len(temp))
            i -= 1
    else:
        combos = np.arange(1, nlevel[0]+1)[:, np.newaxis]

    # Initialize progress bar
    n = len(chans) * len(combos) * len(tasks) * len(subs)
    i = 0
    print('Computing single-trial ERP...')

    for chani, chan in enumerate(chans):
        for taski, task in enumerate(tasks):
            for comboi, combo in enumerate(combos):
                if not default_on:
                    slicebase = f'{{taski}}' 
                    triggerset = {}
                    cond = ''
                    for fi, factor in enumerate(factors):
                        exec(f'triggerset[fi] = triggersets{slicebase}{{fi}}{{combo[fi]}}')
                        if fi < len(factors)-1:
                            cond += f'{levels[fi][combo[fi]]}_'
                        else:
                            cond += f'{levels[fi][combo[fi]]}'
                else:
                    triggers = triggersets[taski][comboi]
                    cond = states[comboi]

                for sub in subs:
                    # Load data
                    if default_on:
                        data, idx, rts = select_trials(sub, triggers)
                    else:
                        args = {}
                        for fi, factor in enumerate(factors):
                            args[factor] = triggerset[fi]
                        data, idx, rts = select_trials2(sub, args)

                    # Adjust response times if necessary
                    if boundary_type != 'fixed':
                        rts[rts==0] = x2search_upper  # adjust for SART: T
                        rts[rts>800] = x2search_upper  # adjust for longer response time

                    # Initialize matrix to store results
                    mat = np.zeros((data.shape[2], 5))  # colnames: s1Idx, s2Idx, val, t, s
                    mat[:, :2] = idx

                    for ni in range(data.shape[2]):  # loop over trial
                        # Find session and trial indices
                        id, session = np.argmax(idx[ni, :2])  

                        # Extract trial data
                        trial = data[chan, :, ni]

                        # Determine search range
                        if boundary_type == 'fixed':
                            x_rng2search = [x2search_lower, x2search_upper]
                        else:
                            x_rng2search = [x2search_lower, rts[ni]]

                        # Compute W matrix
                        if ni == 0:
                            W, x_rng, y_rng = compute_Wst(trial, times, srate, 0)
                        else:
                            W = compute_Wst(trial, times, srate, 0)

                        # Find local peak
                        mat[ni, 2:5] = local_peak(W, x_rng, y_rng, x_rng2search, y_rng2search, sign, 0)

                        # Adjust search range for 'adjusted' boundary type
                        if boundary_type == 'adjusted' and mat[ni, 2] == 0:
                            x_rng2search = [x2search_lower, x2search_upper]
                            mat[ni, 2:5] = local_peak(W, x_rng, y_rng, x_rng2search, y_rng2search, sign, 0)

                        # Plot if required
                        if plot_on:
                            plt.figure()
                            plt.imshow(W, extent=(x_rng[0], x_rng[-1], y_rng[0], y_rng[-1]), aspect='auto', cmap='RdBu_r')
                            plt.colorbar()
                            plt.plot([x_rng2search[0], x_rng2search[-1], x_rng2search[-1], x_rng2search[0], x_rng2search[0]], 
                                     [y_rng2search[0], y_rng2search[0], y_rng2search[-1], y_rng2search[-1], y_rng2search[0]], 
                                     'k:', linewidth=2)
                            if mat[ni, 2] != 0:
                                plt.plot(mat[ni, 3], mat[ni, 4], '--ks', markersize=5, markerfacecolor='k')
                                plt.text(mat[ni, 3]+10, mat[ni, 5], str(mat[ni, 2]))
                            plt.title(f'Session {session}, trial {id}')
                            plt.show()

                    # Output
                    var_name = f'{measure}_{task}_{cond}'
                    file = f'{f_output}\\{measure} chan{chan}\\{sub}.mat'
                    exec(f'{var_name} = mat')
                    if os.path.exists(file):
                        np.save(file, {var_name: mat}, allow_pickle=True)
                    else:
                        np.save(file, {var_name: mat}, allow_pickle=True)

                    i += 1
                    print(f'Progress: {i/n*100:.2f}%')

    print('Computation complete.')

