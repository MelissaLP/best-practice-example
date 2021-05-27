from gwpy.timeseries import TimeSeries
from gwpy.signal import filter_design
from gwpy.plot import Plot
from gwpy.signal.filter_design import bandpass
import matplotlib.pyplot as plt
import matplotlib; matplotlib.use('Agg')
from pycbc.waveform import get_fd_waveform
from pycbc.filter import matched_filter, sigmasq, match
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import re
from numpy import f2py

#Path to good enough project

#Read GPS times of real blips
f=open("./data/raw/gps_times.txt", "r")
gps=np.array(re.findall(r"[+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?", f.read())).astype(float)
print(gps[0])
print('potato')

path_template="./data/processed/"; 

#Compute the ASD of the beginning of O2 to whiten
start = TimeSeries.fetch_open_data('L1', 1164601507.0, 1164601507.0+20); 
asd_s= start.asd(10, window='hann')



for j in range(len(gps)):
    for i in [0,1,10,100,101]:

        blip = TimeSeries.fetch_open_data('L1', float(gps[j])-1.0, float(gps[j])+1.0); 
        
        # Band-pass filter in [35, 250]
        bp = bandpass(float(35), float(250), blip.sample_rate)
        #Notches for the 1st three harminics of the 60 Hz AC
        notches = [filter_design.notch(line, blip.sample_rate) for line in [60,120,180]]
        #Concatenate both filters
        zpk = filter_design.concatenate_zpks(bp, *notches)
        #Call GPS of blips and return a GPS array

        #Whiten and band-pass filter
        #white = data.whiten(0.5,float(0.5/2), window='hann') #whiten the data
        white_down = blip.filter(zpk, filtfilt=True) #downsample to 2048Hz
        white_s = white_down.whiten(3,float(3/2), window='hann', asd=asd_s)[int(3627):int(len(white_down)-3627)]
        white_s= (MinMaxScaler(feature_range=(-1, 1)).fit_transform(np.array(white_s).reshape(-1,1)).flatten())

        deltaf=len(white_s)/4096
        white_s = TimeSeries(white_s,t0=-1/(deltaf*2), sample_rate=4096,name = "L1")


        #Plot version with and without notches
        fig=plt.figure(figsize=(12,5)); 
        plt.plot((np.array(white_s)), label='Original', alpha=0.7) 
        plt.ylabel('Amplitude'); plt.xlabel('Data points')
        plt.title('LIGO-Livingston strain data whitened, band-passed in range ['+str(35)+', '+str(250)+'] $Hz$')
        plt.legend(); plt.show()

       
        template= TimeSeries(np.asarray(np.load(path_template+"fake_glitch_"+str(i)+'.npy')[0], dtype=float),t0=-1/(deltaf*2), sample_rate=4096,name = "L1")

        PSD = white_s.psd(deltaf,float(deltaf/2))
        snrs=matched_filter(template.to_pycbc(),white_s.to_pycbc(), psd=PSD.to_pycbc())
        print(max(np.array(snrs)[100:800]))
        fig=plt.figure(figsize=(12,5)); 
        plt.plot(np.array(snrs)); plt.ylabel('Signal-to-noise ratio (SNR)'); plt.xlabel('Data points')
        plt.savefig('./results/figures/snr_'+str(i)+'_'+str(j)+'.png');plt.close()
