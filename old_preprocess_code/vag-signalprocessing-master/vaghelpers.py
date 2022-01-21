#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
-------------------------------
Vibroarthrography (VAG) Project
-------------------------------
Helper functions for signal analysis

Author: Tuan Nam Le
Last modified: 26/08/2014
"""

import numpy
from scipy.signal import periodogram
from scipy.signal.windows import blackman
from scipy.fftpack import rfft
from numpy import log10
from xml.etree.ElementTree import parse

def vag2float(sig, dtype=numpy.float32):
    """
    Convert VAG signal to floating point with a range from -1 to 1
    dtype=numpy.float32 for single precision
    
    Parameters
    ----------
    sig : array_like
            Input array, must have (signed) integral type
    dtype : data-type, optional
            Desired (floating point) data type

    Returns
    -------
    ndarray
            normalized floating point data
    """
    # Allow unsigned (e.g. 8-bit) data
    sig = numpy.asarray(sig)  # make sure it's a NumPy array
    assert sig.dtype.kind == 'i', "'sig' must be an array of signed integers!"
    dtype = numpy.dtype(dtype)  # allow string input (e.g. 'f')

    # Note that 'min' has a greater (by 1) absolute value than 'max'!
    # Therefore, we use 'min' here to avoid clipping.
    return sig.astype(dtype) / dtype.type(-numpy.iinfo(sig.dtype).min)


def psdheatmap(sig, angles, seg_ind, fs=16000):
    """
    Power spectral density (PSD) of all segments using Welch’s method
    PSD describes how the power of the signal is distributed over the different frequencies
    
    Parameters
    ----------
    sig : VAG signal from knees, array_like
    angles : Angular signal at potentiometer from orthosis, array_like
    seg_ind : Segment indices (the beginning and the end of each segment), array_like
    fs : Sample rate of sig
    
    Returns
    -------
    Matrix of PSD Heatmap 
            column vector: Frequency
            row vector: Measurement segment    
    """
    segments = numpy.reshape(seg_ind, (-1,2))
    N = 1024
    w = blackman(N) # window
    psdmap = []
    for i in range(0, len(segments)):
        segment = sig[segments[i][0]:segments[i][1]]
        [f_seg, Pxx_seg] = periodogram(segment, fs, w, N)
        psdmap.append(Pxx_seg)

    psdmap = numpy.reshape(psdmap,(len(psdmap),N/2+1))
    
    return psdmap
    

def seg_importxml(xmlFile):
    """
    Import all segment indices from the associated XML of VAG file 
    Using toolkit ElementTree. Download at http://effbot.org/downloads#elementtree
    
    Parameters
    ----------
    xmlFile : Filename of the XML file containing node 'segmentation'
    
    Returns
    -------
    seg_ind : Segment indices (the beginning and the end of each segment)
            
    """
    seg_ind = []
    # Parse the XML file into XML element tree
    xml_Tree = parse(xmlFile)
    xml_Root = xml_Tree.getroot()
    # Find the element node 'segmentation' of the XML element tree
    xml_Segmentation = xml_Root.find('segmentation')
    if xml_Segmentation:
        for segment in xml_Segmentation.findall('segment'):
            begin = segment.find('begin').text
            seg_ind.append(int(begin))
            end = segment.find('end').text
            seg_ind.append(int(end))
    else:
        print ("Segment(s) not found")
    return seg_ind
    
    
def freqspec(sig,fs):
    """
    Frequency spectrum generated via Fourier transform using windows
    
    Parameters
    ----------
    sig : VAG signal from knees, array_like
    fs : Sample rate of sig
    
    Returns
    -------
    f : Frequency bin, array_like
    mags : Magnitude values, array_like    
            
    """
    N = len(sig)  # number of signal points
    T = 1.0/fs
    f = numpy.linspace(0.0, 1.0/(2.0*T), N)
    # Compute fft
    w = blackman(N) # window
    mags = abs(rfft(sig*w))
    # Convert to dB
    mags = 20*log10(mags)
    # Normalise to 0 dB max
    mags -= max(mags)
      
    return (f,mags)


def calculateR(sig,fs):
    """
    Calculate the Ratio Index R of a time serie to characterize the power spectrum property
    See: "Analysis of Vibroarthrographic Signals for Knee Osteoarthtitis Diagnosis"
    (2012 - Lee, Lin , Wu, Wang)
         
    Parameters
    ----------
    sig : Time serie, array_like
    fs : Sample rate of sig
    
    Returns
    -------
    Ratio Index R
                            
    """
    N = len(sig)  # number of signal points
    T = 1.0/fs
    f = numpy.linspace(0.0, 1.0/(2.0*T), N)
    # Compute fft
    w = blackman(N) # window
    mags = abs(rfft(sig*w))
    return mags