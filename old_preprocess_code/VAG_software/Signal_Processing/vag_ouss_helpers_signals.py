#!/usr/bin/env python
#fft
import scipy
import numpy
import os

def vag_ouss_rfft_of_signal(signal,fs):
    #fs [sample/sec]
    n = len(signal)
    d = 1./fs
    window = scipy.signal.hamming(n)
    signal_window = signal*window
    #Padding can happen here
    signal_window_zeropadding = signal_window
    rfft_signal = 2.* numpy.fft.rfft((signal_window_zeropadding))/n
    #freq = np.linspace(0.0, fs/2.0, n/2)
    freq = numpy.fft.rfftfreq(n, d)
    return (freq, abs(rfft_signal))

def vag_ouss_resample(signal, destinationSamplesCount):
    resampledSignal = scipy.signal.resample(signal, destinationSamplesCount);
    return resampledSignal
    
def vag_ouss_resample_signals(signals,destinationSamplesCount):
    resampledSignals = []
    for signal in signals:
        resampledSignal = vag_ouss_resample(signal, destinationSamplesCount)
        resampledSignals.append(resampledSignal)
    return resampledSignals
    
def vag_ouss_resample_ffts(FFTs, destinationSamplesCount, fs):
    resampledFFTs = []
    for FFT in FFTs:
#        print numpy.shape(FFT)        
#        sourceSpacing = float(fs)/float(len(FFT[1]))
        resampledSignal = vag_ouss_resample(FFT[1], destinationSamplesCount)
        freqs = numpy.linspace(0.0, 0.5*fs, destinationSamplesCount)
        #print freqs
        resampledFFT = [freqs, resampledSignal]
        resampledFFTs.append(resampledFFT)
    #Build the freqencies
    #Build the resampledFFTs
    return resampledFFTs

def vag_ouss_fft_of_signals(signals):
    FFTs = []
    for signal in signals:
        FFTs.append(vag_ouss_fft_of_signal(signal))
    return FFTs
    
def vag_ouss_rfft_of_signals(signals,fs):
    FFTs = []
    for signal in signals:
        FFTs.append(vag_ouss_rfft_of_signal(signal,fs))
    return FFTs
    
def vag_ouss_normalize_raw_signal(signal):
    signalNumpy = numpy.asarray(signal, numpy.float32)
    maxS = numpy.amax(signalNumpy)
    signalNumpy /= maxS
    return signalNumpy

def vag_nam_zero_crossing_rate(data, threshold):
    # See http://www.ifs.tuwien.ac.at/~schindler/lectures/MIR_Feature_Extraction.html
    zeroCrossings = 0
    NumberOfSamples = len(data)
    for i in range(1, NumberOfSamples):
        if (data[i-1]<threshold and data[i]>threshold) or \
           (data[i-1]>threshold and data[i]<threshold) or \
           (data[i-1]!=threshold and data[i]==threshold):
                zeroCrossings += 1
    zeroCrossingRate = zeroCrossings/float(NumberOfSamples-1)
    return zeroCrossingRate

def vag_nam_root_mean_square(data):
    rms = numpy.sqrt(numpy.mean(numpy.power(data,2)))
    return rms

def vag_ouss_sum_of_fft_of_signals(FFTs):
    sumOfFFTs = [0 for i in range(len(FFTs[0][1]))]
    for FFT in FFTs:
        sumOfFFTs = [x+abs(y) for x,y in zip(sumOfFFTs , FFT[1])]
    return sumOfFFTs

def vag_ouss_fft_of_signal(signal):
    sampleRate=16000 #sample/sec
    numberOfSignalPoints = len(signal)  # number of signal points
    #samplesPeriod = 1.0/sampleRate #sample period
    # Compute FFT
    #window = numpy.ones(N) # no windowing
    window = scipy.signal.hanning(numberOfSignalPoints) # windowing the signal helps mitigate spectral leakage (e.g. Hanning, Hamming, Blackman)
    SW = scipy.fft(signal*window)    
    frequencies = numpy.linspace(0.0, sampleRate/2.0, numberOfSignalPoints/2)
    # sig is real-valued, t83he spectrum is symmetric; only take the FFT corresponding to the positive frequencies
    return (frequencies.tolist(),(2.0/numberOfSignalPoints)*abs(SW[0:numberOfSignalPoints/2]))
    
def vag_ouss_padd_to_same_length(signals):
    longest = max([len(x) for x in signals])
    for signal in signals:
        signal = vag_ouss_signal_zero_padding(signal,longest)
    return signals

def vag_jacqui_high_pass_filter(filterInput, fs=44100.0, fg=10.0):
    fs = float(fs)
    fg = float(fg)
    b,a = scipy.signal.iirdesign(fg/fs,(fg*0.05)/fs,1.0,40)
    #print "generating a ({:d},{:d}) tap HPF IIR".format(len(b),len(a))
    return scipy.signal.lfilter(b,a,filterInput)

def vag_ouss_signal_zero_padding(signal, finalLength):
    data = signal
    data.extend([0 for i in range(finalLength-len(signal))])
    return data

def vag_ouss_concatinate_segments(vibrationRawData, segments):
    #go over segments
    data = []
    for segment in segments:
        #append the right parts of rawData to data
        segmentData = vag_ouss_get_segment_signal(vibrationRawData, segment)
        data = data + segmentData.tolist()
    return data

def vag_ouss_get_segment_signal(rawSignal, segment):
    return rawSignal[segment[0]:segment[1]+1]

def vag_ouss_arrange_segments_signals_in_a_list(rawSignal,segments):
    signals = []
    for segment in segments:
        signals.append(vag_ouss_get_segment_signal(rawSignal, segment))
    return signals

#Reading .wav files
from scipy.io import wavfile as SciPyIOWavFile

def vag_ouss_get_signal_raw_data(signalWavFileDialogFilename,filterInput = True):
    if not os.path.isfile(signalWavFileDialogFilename):
        print "can't find: "+ os.path.basename(signalWavFileDialogFilename)+"\n"
        return
    try:
        #Load the wave file values
        # Import VAG file
        (fs, data) = SciPyIOWavFile.read(signalWavFileDialogFilename)
        if filterInput :
            data[:,0] = vag_jacqui_high_pass_filter(data[:,0])
    except:
        print "Could not read the file :" + signalWavFileDialogFilename
        raise
    #return (vibrationData, angularData)
    return (data[:,0], data[:,1])
