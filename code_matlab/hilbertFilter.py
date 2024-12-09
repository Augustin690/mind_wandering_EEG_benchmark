import numpy as np
from scipy.signal import hilbert, filtfilt, firls
from scipy.fftpack import fft, ifft
import matplotlib.pyplot as plt
from mne.filter import filter_data
from mne.io import read_raw_edf

def hilbert_filter(data, srate, bandrange, check_on=False):
    """
    Apply Hilbert filter to EEG data.

    Parameters
    ----------
    data : array, shape (n_chan, n_pnt, n_trial)
        EEG data.
    srate : float
        Sampling rate of the EEG data.
    bandrange : list or tuple of two floats
        Lower and upper bounds of the frequency band to filter.
    check_on : bool, optional
        If True, plots the filter kernel and its frequency response.

    Returns
    -------
    data_filt : array, shape (n_chan, n_pnt, n_trial)
        Filtered EEG data.
    """
    nyquist = srate / 2
    lower_filter_bound = min(bandrange)  # Hz
    upper_filter_bound = max(bandrange)  # Hz
    transition_width = 0.2  # used to distortion in time-domain
    filter_order = round(3 * srate / lower_filter_bound)  # contain >= 3 cycles of the lowest freq, in unit of pnt

    # Build filter
    ffrequencies = np.array([0, (1 - transition_width) * lower_filter_bound, lower_filter_bound, upper_filter_bound, (1 + transition_width) * upper_filter_bound, nyquist]) / nyquist
    ideal_response = np.array([0, 0, 1, 1, 0, 0])
    filter_weights = firls(filter_order, ffrequencies, ideal_response)  # create finite-impulse response filters via least squares

    # Check kernel
    if check_on:
        filter_weights_norm = (filter_weights - np.mean(filter_weights)) / np.std(filter_weights)
        hz_filtkern = np.linspace(0, nyquist, filter_order // 2 + 1)

        plt.figure()
        plt.plot(ffrequencies * nyquist, ideal_response, 'r')
        plt.hold(True)

        fft_filtkern = np.abs(fft(filter_weights_norm))
        fft_filtkern = fft_filtkern / np.max(fft_filtkern)  # normalized to 1.0 for visual comparison ease
        plt.plot(hz_filtkern, fft_filtkern[:filter_order // 2 + 1], 'b')

        plt.gca().set_ylim(-0.1, 1.1)
        plt.gca().set_xlim(0, nyquist)
        plt.legend(['ideal', 'best fit'])

        freqsidx = np.digitize(hz_filtkern, ffrequencies * nyquist)
        plt.title(['SSE: ', str(np.sum((ideal_response - fft_filtkern[freqsidx]) ** 2))])
        # the closer SSE to zero, the better; don't use filter with SSE > 1

    # Filter data
    rawsize = data.shape

    # In case data only has one trial (size(data,3) == 1)
    if data.shape[2] == 1 or len(rawsize) == 2:
        rawsize = (rawsize[0], rawsize[1], 1)

    if rawsize[1] <= filter_order * 3:  # zero-padding if pnt number isn't enough
        data = np.pad(data, ((0, 0), (0, filter_order * 3 - rawsize[1] + 1), (0, 0)), 'constant')

    data_filt = np.zeros_like(data)
    for ti in range(data.shape[2]):
        for chani in range(data.shape[0]):
            data_filt[chani, :, ti] = filtfilt(filter_weights, 1, data[chani, :, ti])

    data = data[:, :rawsize[1], :]
    data_filt = data_filt[:, :rawsize[1], :]

    # Transform to analytical signal
    for ti in range(data_filt.shape[2]):
        data_filt[:, :, ti] = hilbert(data_filt[:, :, ti], axis=1)

    # Check transform by examining phase
    if check_on:
        chan = np.random.randint(1, data_filt.shape[0] + 1)
        ni = np.random.randint(1, data_filt.shape[2] + 1)  # pick a trial randomly
        times = np.arange(0, data_filt.shape[1] * 1000 / srate, 1000 / srate)
        plt.figure()
        plt.suptitle(['Channel ', str(chan)])
        plt.subplot(1, 2, 1)
        plt.plot(times, data[chan, :, ni], 'b', times, np.real(data_filt[chan, :, ni]), 'r--')
        plt.xlabel('Time (ms)'), plt.ylabel('Amp. (\muV)')
        plt.legend(['Original', 'Filtered'])
        plt.subplot(1, 2, 2)
        plt.plot(times, np.angle(data_filt[chan, :, ni]), 'b');
        plt.xlabel('Time (ms)'), plt.ylabel('Phase angle (rad.)')

    return data_filt