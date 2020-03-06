import pyaudio
import wave
import numpy as np
from matplotlib import pyplot as plt
#plt.switch_backend('QT4Agg')
from scipy import signal as sig
from scipy.signal import find_peaks
from scipy.signal import butter, lfilter
from scipy.signal import correlate
from scipy.io import wavfile as wfile

def envelope(signal):
    '''
    Fast envelop detection of real signal using thescipy hilbert transform
    Essentially adds the original signal to the 90d signal phase shifted version
    similar to I&Q
    '''
    signal = sig.hilbert(signal)
    envelope = abs(signal)# np.sqrt(signal.imag**2 + signal.real**2)
    return envelope

def Smoothing(DataArray, Window=101, PolyOrder=3):
    DataArray = sig.savgol_filter(DataArray, Window, PolyOrder)
    return DataArray

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

# Import audio for cross correlation
samplerate, ChirpKernel = wfile.read('C:\\Users\\jense\\ChirpTrain.wav', mmap=False)
print(ChirpKernel)


#Number of observations
N = 100
# PRI length in Seconds
PRI=0.236
CHUNKSIZE = int(44100*PRI) # fixed chunk size
rate=44100
# initialize portaudio
p = pyaudio.PyAudio()
stream = p.open(format=pyaudio.paInt16, channels=1, rate=44100, input=True, frames_per_buffer=CHUNKSIZE)

data = []

# listen
print('Listening...')
x = 0
while x < N:
    values = stream.read(CHUNKSIZE)
    data.append(values)
    x = x+1

# record
data = np.array(data)
MicData = np.frombuffer(data, dtype=np.int16)
stream.stop_stream()
stream.close()
p.terminate()

print('Done.')

#Process Data
# Envelope Hilbert Transform
#MicData = MicData*(1000/max(MicData))
MicData = butter_bandpass_filter(MicData, 4e3, 5e3, fs=44100, order=1)
MicData = envelope(MicData)
#Reshape
MicData = MicData.reshape(N,int(x*CHUNKSIZE/N))
print('dimensions of MicData',MicData.shape)

# Store Envelope without Cross Correlation for later Comparison
GatedEnvelope=np.sum(MicData,axis=0)
max_value = max(GatedEnvelope)
max_index = np.argmax(GatedEnvelope)
GatedEnvelope = np.roll(GatedEnvelope,-1*max_index)

# Cross Correlation
# Setup Kernel to same length as PRI
ChirpKernel = ChirpKernel[0:len(MicData[0,:])]
#Normalize Chirpkernel for a better correlation
ChirpKernel = ChirpKernel*(np.amax(MicData)/np.amax(ChirpKernel))
#Test Point
plt.plot(ChirpKernel)
# Find Envelope of ChirpKernel
ChirpKernel = envelope(ChirpKernel)

# Compares ChirpKernel preenvolope vs envelope vs MicData
plt.plot(ChirpKernel)
plt.plot(MicData[5,:])
# Cross Correlate Signals row by row
x=0
while x < N:
    MicData[x,:] = sig.correlate(MicData[x,:],ChirpKernel,mode='same')
    x=x+1







#Gate Data
GatedData=np.sum(MicData,axis=0)

# Find Max
max_value = max(GatedData)
max_index = np.argmax(GatedData)
print('this is the max index', max_index)

GatedData = np.roll(GatedData,-1*max_index)
MicData = np.roll(MicData,-1*max_index)

#Index Peaks
peaks, _ = find_peaks(GatedData, height=np.mean(GatedData), distance=200)





dx=343/rate
peakdistance=dx*peaks

print('The Number of target is', (len(peakdistance)))
print("The distance to targets in meters is", peakdistance/2)

# plot data
fig, axs = plt.subplots(6)
axs[0].plot(MicData[0,:])
axs[1].plot(MicData[1,:])
axs[2].plot(MicData[2,:])
axs[3].plot(MicData[3,:])
axs[4].plot(MicData[4,:])
axs[5].semilogy(GatedData)
axs[5].semilogy(peaks,GatedData[peaks], 'x')
axs[5].semilogy(GatedEnvelope)
plt.show()
# close stream
