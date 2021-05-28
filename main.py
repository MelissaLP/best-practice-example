import re
import itertools
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from pycbc.filter import matched_filter
from gwpy.timeseries import TimeSeries
from gwpy.signal import filter_design
from gwpy.signal.filter_design import bandpass
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

#####################
#     Functions     #
#####################


def WhiteningAndPSDComputing(signal, ASD, deltaf):

    """
        This function whitens the raw signal and computes the PSD. 
        It returns a Time Series that matches the size of the template
        and the PSD for matched filtering technique.

    Input
    ______

       Signal: Time Series to filter and whiten
       ASD: Frequency Series necessary to whiten
       deltaf: Parameter to match the template

    Output
    ______

       cleaned_signal: Time Series containing cleaned signal
       PSD: Frequency Series containing signal PSD

    """
    # Filtering: Band-pass in [35, 250], notches 60Hz harmonics
    bp = bandpass(float(35), float(250), signal.sample_rate) 
    notches = [filter_design.notch(line, signal.sample_rate) for line in [60, 120, 180]]
    zpk = filter_design.concatenate_zpks(bp, *notches)
   
    filter_signal = signal.filter(zpk, filtfilt=True)  
    whiten_signal = filter_signal.whiten(3, 1.5, window='hann', asd=ASD)
    cut_signal = whiten_signal[int(3627):int(len(whiten_signal)-3627)] # TODO: generalize

    array_signal = np.array(cut_signal).reshape(-1, 1)
    rescaled_signal = MinMaxScaler(feature_range=(-1, 1)).fit_transform(array_signal).flatten()
    cleaned_signal = TimeSeries(rescaled_signal, t0=-1/(deltaf*2), sample_rate=4096, name="L1")

    PSD = cleaned_signal.psd(deltaf, deltaf/2)

    return cleaned_signal, PSD
    
#####################
#        Data       #
#####################

# Read GPS times of real blips
times = open("./data/raw/gps_times.txt", "r")
read_times = re.findall(r"[+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?", times.read())
gps = np.asarray(read_times, dtype=float)

# Compute the ASD of the beginning of O2 to whiten
start = TimeSeries.fetch_open_data('L1', 1164601507.0, 1164601507.0+20)
ASD = start.asd(10, window='hann')

#####################
#     MF method     #
#####################

for i, j in itertools.product([0, 1, 10, 100, 101], np.arange(len(gps)),):

    # Call the input signal and the template to convolve 
    raw_signal = TimeSeries.fetch_open_data('L1', (gps[j])-1.0, (gps[j])+1.0)
    template = np.asarray(np.load("./data/processed/fake_glitch_"+str(
        i)+'.npy')[0], dtype=float)

    #To match the template and the signal
    deltaf = len(template)/4096 

    #t0 represents the initial time, where the middle of the template is at t=0
    template = TimeSeries(template, t0=-1/(deltaf*2), sample_rate=4096, name="L1")

    signal, PSD = WhiteningAndPSDComputing(raw_signal, ASD, deltaf)

    snrs = matched_filter(template.to_pycbc(),
                          signal.to_pycbc(),
                          psd=PSD.to_pycbc())


    fig = plt.figure(figsize=(12, 5))
    plt.plot(np.array(snrs))
    plt.ylabel('Signal-to-noise ratio (SNR)')
    plt.xlabel('Data points')
    plt.savefig('./results/figures/snr_'+str(i)+'_'+str(j)+'.png')
    plt.close()
