import os
import numpy as np
from mne.io import read_raw_edf
from mne.channels import make_standard_montage
from mne.filter import filter_data

def get_data(sub, task, state, trialIdx=None):
    """
    Get data to plot based on the given task_state combination or the trial index.
    Parameters:
    - sub: Subject ID
    - task: Task name
    - state: State name
    - trialIdx: Trial indices to select. If None, all trials are selected.
    
    Returns:
    - data: A 3D matrix of raw data with size of nchan x npnt x ntrial
    """
    # Set default path
    data_path = 'c:\\topic_mind wandering\\3data'
    os.system('cd ' + data_path)
    
    # Load parameters
    if task:
        # Assuming 'pars' is a Python module with 'tasks', 'states', 'triggersets' as attributes
        from pars import tasks, states, triggersets
        
        # Find state index
        si = next((i for i, s in enumerate(states) if s == state), None)
        if si is None:
            raise ValueError('Invalid state!')
        
        # Find task index
        ti = next((i for i, t in enumerate(tasks) if t == task), None)
        if ti is None:
            raise ValueError('Invalid task!')
        
    # Load data
    if trialIdx is None or isinstance(trialIdx, list):  # Load data by task_state combination
        data = select_trials(sub, triggersets[ti][si])
    else:  # Load data by trialIdx (x task_state)
        if not task:  # If task is not specified, use a default trigger matrix
            triggerbase = np.arange(10, 61, 10)
            triggermat = np.add.outer(triggerbase, np.arange(7))
            data, rawIdx, _ = select_trials(sub, triggermat.reshape(1, -1))
        else:
            data, rawIdx, _ = select_trials(sub, triggersets[ti][si])
        
        # Select trials based on trialIdx
        trial2select = np.isin(rawIdx, trialIdx, axis=1)
        data = data[:, :, trial2select]
    
    return data