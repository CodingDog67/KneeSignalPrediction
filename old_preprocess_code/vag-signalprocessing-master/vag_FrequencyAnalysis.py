#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Plot vibration signal, angular signal, PSD heatmap, frequency spectrum, auto-correlation
Author: Tuan Nam Le
Last modified: 02 July 2014
"""

from scipy.io import wavfile
from scipy import log10
from scipy.signal import blackman, periodogram, correlate
import matplotlib.pyplot as plt
import numpy
import os
from PySide2.QtGui import QFileDialog, QApplication

# Some another helpers
import vaghelpers
import segmentation


def AutoCorr(x,fs):
    
    x = x[:fs*2]
    ac = correlate(x,x)
    n = len(ac)
    ac = 2*ac/n
    time = numpy.linspace(0.0,numpy.divide(float(len(x)),fs),len(x))
    dt = (time[1]-time[0])
    dur = n*dt/2
    d = numpy.linspace(-dur,dur,n)
    
    idx = numpy.argmax(ac) 
    if(d[idx]<dt/4):
        d[idx]=0
        
    print ("Autocorrelation  Max: Delay=%6.3g sec   Amp=%7.4g " %(d[idx],ac[idx]))
    
    fig = plt.figure(u"Auto correlation")
    plt.plot(d,ac,color='r')
    plt.xlabel(u"Time (s)")
    plt.title(u"Auto correlation")
    plt.grid()


if __name__ == '__main__':

    try:
        QApplication([])
    except RuntimeError as e:
        pass
    vagFile = QFileDialog.getOpenFileName(caption ="Open WAV File", filter = "*.wav")
    vagFile = vagFile[0]

    # Import VAG file
    (fs, samples) = wavfile.read(vagFile)
    # scipy.io.wavread does not support 32-bit float files
    vagsamples = vaghelpers.vag2float(samples, numpy.float32)

    # Separate signal and angular values for segmentation
    signal = vagsamples[:,0]
    angles = vagsamples[:,1]
    time = numpy.linspace(0,numpy.divide(float(len(signal)),fs),len(signal))

    # Get segment indices, if associated XML file with node 'segmentation' exist,
    # import the segment indices, if not use the FEN's segmentation algorithm 
    (realname, extension) = os.path.splitext(vagFile)
    xmlFile = realname+".xml"
    if os.path.isfile(xmlFile):
        seg_ind = vaghelpers.seg_importxml(xmlFile)
    else:
        seg_ind = segmentation.segmentation_jhu(fs, angles)
    
    
    #### Plot Vibration and Angular ####
    fig = plt.figure(u"Vibration and Angular")

    ax1 = plt.subplot(2,1,1)
    plt.plot(time, signal, color='b')
    plt.xlabel(u"Time (s)")
    plt.ylabel(u"Amplitude (a.u.)")
    plt.title(u"Vibration")

    for idx in range(0, len(seg_ind)):
        if not idx%2:
            ax1.axvline(seg_ind[idx]/fs, color='k', linewidth=1)
        else:
            ax1.axvline(seg_ind[idx]/fs, color='k', linewidth=1)
            ax1.axvspan(seg_ind[idx-1]/fs, seg_ind[idx]/fs, facecolor='b', alpha=0.1)

    ax1.set_xlim([0, max(time)])
    ax1.set_ylim([-1.0, 1.0])
    plt.grid()

    ax2 = plt.subplot(2,1,2)
    plt.plot(time, angles, color='r')
    plt.xlabel(u"Time (s)")
    plt.ylabel(u"Amplitude (a.u.)")
    plt.title(u"Angular")
    ax2.set_xlim([0, max(time)])
    ax2.set_ylim([-1.0, 1.0])
    plt.grid()
    plt.tight_layout()

    #### Plot Power Spectral Density Segments Heatmap ####
    fig = plt.figure(u"PSD Heatmap")
        
    segments = numpy.reshape(seg_ind, (-1,2))
    psdmap = vaghelpers.psdheatmap(signal, angles, seg_ind, fs)
    
    cax = plt.imshow(log10(psdmap),vmin=-17,vmax=-5,extent=[0,fs/2,len(segments),0], aspect='auto',interpolation="nearest")
    cbar = plt.colorbar(cax,ticks=[-17,-14,-11,-8,-5])
    plt.xlabel(u"Frequency (Hz)")
    plt.ylabel(u"Measurement segment")
    cbar.ax.set_yticklabels([-17,-14,-11,-8,-5])
    cbar.set_label('log10(PSD)')
    
    
    #### Plot Frequency Spectrum #####
    fig = plt.figure(u"Frequency Spectrum")
    (f, mags) = vaghelpers.freqspec(signal,fs)
    plt.plot(f,mags, color='m')
    plt.xlabel(u"Frequency (Hz)")
    plt.ylabel(u"Magnitude (dB)")
    plt.grid()
    
    #### Plot Auto Correlation ####
    AutoCorr(signal,fs)
    
    plt.show()
