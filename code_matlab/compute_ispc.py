import numpy as np
from mne.time_frequency import tfr_array_morlet

def compute_ispc(data, freq=None):
    """
    Compute inter-site phase clustering across time between rows (chans)
    Note: If 2 chans has high ISPC & phase-lag nearly == 0, the coherence
          might be caused by volumn conductance. To exclude volumn
          conduction, use phase-lag coherence instead. Moreover, for the 
          directional phase-lag coherence, check the data2psiX.m
    data: nChan x nPnt x nTrial complex
    ispc / lag: nChan x nChan x nTrial
    phase-lag can be further used to do gv-test
    """

    phase = np.angle(data)
    nChan = data.shape[0]

    # initialize
    ispc = np.zeros((nChan, nChan, data.shape[2]))
    lag = ispc

    for chani in range(nChan-1):
        for chanj in range(chani + 1, nChan):
            phase_sig1 = phase[chani, :, :]
            phase_sig2 = phase[chanj, :, :]
            phase_diffs = phase_sig1 - phase_sig2
            meanDiffVec = np.mean(np.exp(1j * phase_diffs), axis=1)  # averaging the diff vector space across pnts
            ispc[chani, chanj, :] = np.abs(meanDiffVec)
            lag[chani, chanj, :] = np.angle(meanDiffVec)
            ispc[chanj, chani, :] = ispc[chani, chanj, :]  # flip along diagnol
            lag[chanj, chani, :] = lag[chani, chanj, :]

    if freq is not None:
        lag = 1000 * lag / (2 * np.pi * freq)

    return ispc, lag