#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
-------------------------------
Vibroarthrography (VAG) Project
-------------------------------
Helper functions for signal analysis

Author: Tuan Nam Le
Last modified: 02/06/2015 Walther Schulze
"""
import os
from os import path
import numpy
from scipy.io import wavfile
from scipy.signal import hanning, periodogram, iirdesign, lfilter
from scipy.fftpack import fft
from numpy import log10
import matplotlib.pyplot as plt
from matplotlib import cm
from math import ceil, log, sqrt
import matplotlib
import matplotlib.patches as patches
import csv
import pywt

import xml.etree.ElementTree as XMLETreeElementTree

from numpy import interp
from sklearn import svm
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report


def import_vagfile(vagFile):
    """
    Read VAG file
    Return vibration sensor signal and potentiometer signal and sampling rate of the signals

    Parameters
    ----------
    vagFile : Filename of the RAW WAV file

    Returns
    -------
    vibration : array_like
            Vibration sensor signal
    angular : array_like
            Potentiometer signal
    time : array_like
            Seconds
    fs : sampling rate

    """
    if not path.isfile(vagFile):
        print("could not find the file: " + path.basename(vagFile))
        return
    try:
        # Import VAG file
        (fs, samples) = wavfile.read(vagFile)
        # scipy.io.wavread does not support 32-bit float files
        vagsamples = vag2float(samples, numpy.float32)
        # Separate signal and angular values for segmentation
        vibration = vagsamples[:, 0]
        angular = vagsamples[:, 1]
        time = numpy.linspace(0, numpy.divide(float(len(vibration)), fs), len(vibration))
    except:
        print("Could not read the file: " + vagFile)
        raise
    return (vibration, angular, time, fs)


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


def normalize_raw_signal(signal):
    signalNumpy = numpy.asarray(signal, numpy.float32)
    maxS = numpy.amax(signalNumpy)
    signalNumpy /= maxS
    return signalNumpy


def apply_matrix_zeros(lst, dtype=numpy.float32):
    """
    Function for zero-padding
    See more: http://stackoverflow.com/questions/19878250/zero-padding-numpy-array

    Parameters
    ----------
    lst :   array_like
            Input array
    dtype : data-type, optional
            Desired (floating point) data type

    Returns
    -------
    result
            array with zero-padding
    """
    inner_max_len = max(map(len, lst))
    result = numpy.zeros([len(lst), inner_max_len], dtype)
    for i, row in enumerate(lst):
        for j, val in enumerate(row):
            result[i][j] = val
    return result


def get_max_length_segments(signal, segments):
    max_length = 0
    for segment in segments:
        segmentData = signal[segment[0]:segment[1] + 1]
        if len(segmentData) > max_length:
            max_length = len(segmentData)
    return max_length


def latexify(fig_width=None, fig_height=None, columns=1):
    """Set up matplotlib's RC params for LaTeX plotting.
    Call this before plotting a figure.

    Parameters
    ----------
    fig_width : float, optional, inches
    fig_height : float,  optional, inches
    columns : {1, 2}
    """

    # code adapted from http://www.scipy.org/Cookbook/Matplotlib/LaTeX_Examples

    # Width and max height in inches for IEEE journals taken from
    # computer.org/cms/Computer.org/Journal%20templates/transactions_art_guide.pdf

    assert (columns in [1, 2])

    if fig_width is None:
        fig_width = 3.39 if columns == 1 else 6.9  # width in inches

    if fig_height is None:
        golden_mean = (sqrt(5) - 1.0) / 2.0  # Aesthetic ratio
        fig_height = fig_width * golden_mean  # height in inches

    MAX_HEIGHT_INCHES = 8.0
    if fig_height > MAX_HEIGHT_INCHES:
        print("WARNING: fig_height too large:" + fig_height +
              "so will reduce to" + MAX_HEIGHT_INCHES + "inches.")
        fig_height = MAX_HEIGHT_INCHES

    params = {'backend': 'ps',
              'text.latex.preamble': ['/usepackage{gensymb}'],
              'axes.labelsize': 8,  # fontsize for x and y labels (was 10)
              'axes.titlesize': 8,
              'text.fontsize': 8,  # was 10
              'legend.fontsize': 8,  # was 10
              'xtick.labelsize': 8,
              'ytick.labelsize': 8,
              'text.usetex': True,
              'figure.figsize': [fig_width, fig_height],
              'font.family': 'serif'
              }

    matplotlib.rcParams.update(params)


def format_axes(ax):
    SPINE_COLOR = 'gray'
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)

    for spine in ['left', 'bottom']:
        ax.spines[spine].set_color(SPINE_COLOR)
        ax.spines[spine].set_linewidth(0.5)

    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')

    for axis in [ax.xaxis, ax.yaxis]:
        axis.set_tick_params(direction='out', color=SPINE_COLOR)

    return ax


def plt_vibration_angular_with_segments(vibration, angular, time, fs, seg_ind):
    """
    Plot vibration signal and potentiometer signal

    """
    figure, (ax1, ax2) = plt.subplots(2, sharex=True)
    setFont = 12
    ax1.plot(time, vibration, color='b')
    ax1.set_xlabel("Time (s)", fontsize=setFont)
    ax1.set_ylabel("Amplitude (a.u.)", fontsize=setFont)
    ax1.set_title("Vibration sensor signal", fontsize=setFont)

    for idx in range(0, len(seg_ind)):
        if not idx % 2:
            ax1.axvline(seg_ind[idx] / fs, color='k', linewidth=1)
        else:
            ax1.axvline(seg_ind[idx] / fs, color='k', linewidth=1)
            ax1.axvspan(seg_ind[idx - 1] / fs, seg_ind[idx] / fs, facecolor='#009440',
                        alpha=0.1)  # Beautiful LMU color RGB=0.148.64
            # Maxima
            angular = map(lambda x: (x), angular);
            angular_seg = angular[seg_ind[idx - 1]:seg_ind[idx]]
            max_seg = max(angular_seg)
            max_seg_idx = angular_seg.index(max_seg) + seg_ind[idx - 1]
            ax1.axvline(max_seg_idx / fs, color='r', linewidth=1)

    ax1.set_xlim([0, max(time)])
    ax1.set_ylim([-0.5, 0.5])
    ax1.grid(True)

    ax2.plot(time, angular, color='r')
    ax2.set_xlabel("Time (s)", fontsize=setFont)
    ax2.set_ylabel("Amplitude (a.u.)", fontsize=setFont)
    ax2.set_title("Potentiometer signal", fontsize=setFont)
    ax2.set_xlim([0, max(time)])
    ax2.set_ylim([-1.0, 1.0])
    ax2.grid(True)


def plt_vibration_latex(vibration, time, fs, seg_ind):
    latexify()
    fig = plt.figure(1)
    ax = fig.add_subplot(111)
    ax.plot(time, vibration, linewidth=0.05)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Normalized amplitude (a.u.)")
    # ax1.set_title("Vibration sensor signal")
    ax.set_xlim([0, max(time)])
    ax.set_ylim([-1.0, 1.0])
    ax.grid(True)
    plt.tight_layout()
    format_axes(ax)
    return ax


def plt_angular_latex(angular, time, fs, seg_ind):
    latexify()
    maxAng = max(angular)
    minAng = min(angular)
    fig = plt.figure(1)
    ax = fig.add_subplot(111)
    ax.plot(time, angular, linewidth=0.05)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Normalized amplitude (a.u.)")
    # ax.set_title("Angular sensor signal")
    ax.set_xlim([0, max(time)])
    ax.set_ylim([1.05 * minAng, 1.05 * maxAng])
    ax.grid(True)
    plt.tight_layout()
    format_axes(ax)
    return ax


def plt_vibration_angular_latex(vibration, angular, time, seg_ind):
    latexify(fig_height=3.5, columns=2)
    figure, (ax1, ax2) = plt.subplots(2, sharex=True)
    ax1.plot(time, vibration, linewidth=0.1, c="k")
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Amplitude (a.u.)")
    # ax1.set_title("Vibration sensor signal")
    ax1.set_xlim([0, max(time)])
    ax1.set_ylim([-0.55, 0.55])
    ax1.set_yticks([-0.5, -0.25, 0, 0.25, 0.5])
    ax1.grid(True)
    plt.tight_layout()
    format_axes(ax1)

    maxAng = max(angular)
    minAng = min(angular)
    ax2.plot(time, angular, linewidth=0.1, c="k")
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Amplitude (a.u.)")
    # ax.set_title("Angular sensor signal")
    ax2.set_xlim([0, max(time)])
    # ax2.set_ylim([0.99*minAng, 1.01*maxAng])
    ax2.set_ylim([0.84, 1.01])
    ax2.set_yticks([0.85, 0.90, 0.95, 1.0])
    ax2.grid(True)
    plt.tight_layout()
    format_axes(ax2)
    return figure


def plt_segments_in_figure_latex(figure, vibration, angular, time, segments, mycolor, mylinestyle, myhatch):
    (ax1, ax2) = figure.axes
    for segment in segments:
        ax1.axvspan(time[segment[0]], time[segment[1] + 1], linewidth=0.25, color=mycolor, fill=False,
                    linestyle=mylinestyle, hatch=myhatch)
        ax2.axvspan(time[segment[0]], time[segment[1] + 1], linewidth=0.25, color=mycolor, fill=False,
                    linestyle=mylinestyle, hatch=myhatch)
    return figure


def plt_extrema_in_figure_latex(figure, vibration, angular, time, segments, mycolor):
    (ax1, ax2) = figure.axes
    for segment in segments:
        ax2.scatter(time[segment[0]], angular[segment[0]], s=3, color=mycolor)
        # ax2.text(time[segment[0]], 1.015, r'$t$', fontsize=8)
        ax2.scatter(time[segment[1]], angular[segment[1]], s=3, color=mycolor)
    return figure


def plt_add_tmax_in_figure_latex(figure, vibration, angular, time, segmentsExtension, segmentNummer):
    (ax1, ax2) = figure.axes
    i = 0
    for segment in segmentsExtension:
        i += 1
        if i == segmentNummer:
            ax2.text(time[segment[1]] - 2, 1.015, r'$t_{max,%i}$' % i, fontsize=8)
    return figure


def plt_add_ts_te_in_figure_latex(figure, vibration, angular, time, segments, segmentNummer):
    (ax1, ax2) = figure.axes
    i = 0
    for segment in segments:
        i += 1
        if i == segmentNummer:
            ax2.text(time[segment[0]] - 1, 0.822, r'$t_{s,%i}$' % i, fontsize=8)
            ax2.text(time[segment[1]] - 1, 0.822, r'$t_{e,%i}$' % i, fontsize=8)
    return figure


def plt_marking_segment_in_figure_latex(figure, vibration, angular, time, segments, segmentNummer):
    (ax1, ax2) = figure.axes
    i = 0
    for segment in segments:
        i += 1
        if i == segmentNummer:
            ax2.axvspan(time[segment[0]], time[segment[1] + 1], linewidth=0.2, color='grey', alpha=0.25)
    return figure


def plt_segments_axis(axis, segments, color, linestyle, hatch, scaling=1.0):
    # Interate over segments
    for segment in segments:
        scaledSegment = [x / scaling for x in segment]
        plt_segment_axis(axis, scaledSegment, color, linestyle, hatch)
    plt.show()
    return axis


def plt_segments_figure(figure, segments, color, linestyle, hatch, scaling=1.0):
    # Interate over segments
    for segment in segments:
        scaledSegment = [x / scaling for x in segment]
        plt_segment_figure(figure, scaledSegment, color, linestyle, hatch)
    plt.show()
    return figure


def plt_segment_axis(ax, segment, color, linestyle, hatch):
    axYAxisRange = ax.yaxis.get_view_interval()
    axHeight = axYAxisRange[1] - axYAxisRange[0]
    segmentWidth = segment[1] - segment[0]
    axRect = patches.Rectangle((segment[0], axYAxisRange[0]), segmentWidth, axHeight, fill=False, linewidth=0.2)
    axRect.set_linestyle(linestyle)
    axRect.set_color(color)
    axRect.set_hatch(hatch)
    ax.add_patch(axRect)


def plt_segment_figure(figure, segment, color, linestyle, hatch):
    (ax1, ax2) = figure.axes
    ax1YAxisRange = ax1.yaxis.get_view_interval()
    ax2YAxisRange = ax2.yaxis.get_view_interval()
    ax1Height = ax1YAxisRange[1] - ax1YAxisRange[0]
    ax2Height = ax2YAxisRange[1] - ax2YAxisRange[0]
    segmentWidth = segment[1] - segment[0]
    ax1Rect = patches.Rectangle((segment[0], ax1YAxisRange[0]), segmentWidth, ax1Height, fill=False, linewidth=0.2)
    ax1Rect.set_linestyle(linestyle)
    ax1Rect.set_color(color)
    ax1Rect.set_hatch(hatch)
    ax1.add_patch(ax1Rect)

    ax2Rect = patches.Rectangle((segment[0], ax2YAxisRange[0]), segmentWidth, ax2Height, fill=False, linewidth=0.2)
    ax2Rect.set_linestyle(linestyle)
    ax2Rect.set_color(color)
    ax2Rect.set_hatch(hatch)
    ax2.add_patch(ax2Rect)
    return figure


def plt_3d_time_segment_amplitude(signal, segments):
    matrx = []
    max_length = 0
    for segment in segments:
        segmentData = signal[segment[0]:segment[1] + 1]
        matrx.append(segmentData)
        if len(segmentData) > max_length:
            max_length = len(segmentData)
    matrx = apply_matrix_zeros(matrx)

    fig = plt.figure("Plot 3D Time-Segment-Amplitude")
    ax = fig.add_subplot(111, projection='3d')
    color = iter(cm.rainbow(numpy.linspace(0, 1, len(segments))))
    for i in reversed(range(0, len(segments))):
        c = next(color)
        ax.plot(range(0, max_length), i * numpy.ones(max_length), matrx[i], c=c)

    ax.set_xlabel("Time")
    ax.set_ylabel('Segment')
    ax.set_zlabel('Amplitude')
    ax.set_title(u"3D Time-Segment-Amplitude")
    # ax.view_init(70, 30) # good views
    # ax.view_init(30, 45)


def plt_3d_frequency_segment_amplitude(signal, fs, segments, max_length):
    matrx_fq = []
    n = int(pow(2, ceil(log(max_length) / log(2))))  # Find nearest power of 2
    for segment in segments:
        segmentData = signal[segment[0]:segment[1] + 1]
        fn, mags = calculate_frequency_spectrum_length_n(segmentData, n, fs)
        matrx_fq.append(mags)

    fig = plt.figure(" Plot 3D Frequency-Segment-Amplitude using FFT")
    ax = fig.add_subplot(111, projection='3d')
    color = iter(cm.rainbow(numpy.linspace(0, 1, len(segments))))
    for i in reversed(range(0, len(segments))):
        c = next(color)
        ax.plot(fn, i * numpy.ones(len(fn)), 20 * log10(matrx_fq[i]), c=c)

    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Segment")
    ax.set_zlabel("Amplitude in dB")
    ax.set_zlim(-200, 0)
    ax.set_title("3D Frequency-Segment-Amplitude using FFT")


def plt_psdheatmap(signal, angles, segments, fs=16000):
    """
    Power spectral density (PSD) of all segments using Welch’s method
    PSD describes how the power of the signal is distributed over the different frequencies

    Parameters
    ----------
    signal : Vibration signal, array_like
    angles : Angular signal at potentiometer from orthosis, array_like
    seg_ind : Segment indices (the beginning and the end of each segment), array_like
    fs : Sample rate of sig

    Returns
    -------
    Matrix of PSD Heatmap
            column vector: Frequency
            row vector: Measurement segment
    """
    N = 1024
    w = hanning(N)  # window
    psdmap = []
    for segment in segments:
        segmentData = signal[segment[0]:segment[1] + 1]
        [f_seg, Pxx_seg] = periodogram(segmentData, fs, w, N)
        psdmap.append(Pxx_seg)

    psdmap = numpy.reshape(psdmap, (len(psdmap), N / 2 + 1))

    return psdmap


def get_segment_indices_from_xml(xmlFile, segmentationVersion):
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
    # Prepare an empty list that will contain the SegmentsIndices
    segmentsIndices = []
    # Parse the XML file into XML element tree
    xmlElementTree = XMLETreeElementTree.parse(xmlFile)
    xmlRoot = xmlElementTree.getroot()
    # Find the element node 'segmentation' of the XML element tree with the right version
    xmlSegmentationNodeFound = False
    for xmlSegmentationNode in xmlRoot.iter('segmentation'):
        if xmlSegmentationNode is not None:
            if xmlSegmentationNode.attrib['version'] == segmentationVersion:
                xmlSegmentationNodeFound = True
                xmlSegmentationNodeRequestedVersion = xmlSegmentationNode
                break
    if not xmlSegmentationNodeFound:
        print("Could not find Segmentation Node of version: " + segmentationVersion + " for: " + path.basename(xmlFile))
        return
    # print xmlSegmentationNodeRequestedVersion
    for segment in xmlSegmentationNodeRequestedVersion.findall('segment'):
        segmentBegin = segment.find('begin').text
        segmentsIndices.append(int(segmentBegin))
        segmentEnd = segment.find('end').text
        segmentsIndices.append(int(segmentEnd))
    return segmentsIndices


def get_segment_indices_from_vagfile(vagFile, segmentationVersion):
    """
    Import all segment indices from VAG file

    Parameters
    ----------
    vagFile : Filename of RAW WAV file

    Returns
    -------
    seg_ind : Segment indices (the beginning and the end of each segment)

    """
    seg_ind = []
    (realname, extension) = path.splitext(vagFile)
    xmlFile = realname + ".xml"
    if path.isfile(xmlFile):
        seg_ind = get_segment_indices_from_xml(xmlFile, segmentationVersion)
    return seg_ind


def segment_indices_to_segments(seg_ind):
    segments = numpy.reshape(seg_ind, (-1, 2))
    return segments


def concatinate_segments(signal, segments):
    """
    Concatinate all segments

    Parameters
    ----------
    signal : Time serie, array_like
    segments : Segment indices matrix

    Returns
    -------
    Concatinate signal

    """
    concatinated_signal = []
    for segment in segments:
        segmentData = signal[segment[0]:segment[1] + 1]
        concatinated_signal.extend(segmentData)
    return concatinated_signal


def calculate_frequency_spectrum(signal, fs=16000):
    """
    Frequency spectrum generated via Fourier transform using windows

    Parameters
    ----------
    signal : array_like
    fs : Sample rate of sig

    Returns
    -------
    f : Frequency bin, array_like
    amp : Amplitude values, array_like

    """
    N = len(signal)  # number of signal points
    T = 1.0 / fs
    # Compute FFT
    # w = numpy.ones(N) # no windowing
    w = hanning(N)  # windowing the signal helps mitigate spectral leakage (e.g. Hanning, Hamming, Blackman)
    ywf = fft(signal * w)
    xf = numpy.linspace(0.0, 1.0 / (2.0 * T), N / 2)
    # sig is real-valued, the spectrum is symmetric; only take the FFT corresponding to the positive frequencies
    return (xf, 2.0 / N * abs(ywf[0:N / 2]))


def plt_frequency_spectrum(xf, yf):
    """
    Plot frequency spectrum generated via Fourier transform using windows

    Parameters
    ----------
    xf : Frequency, array_like
    yf : Amplitude values, array_like

    Returns
    -------
    Plot

    """
    fig = plt.figure("Plot Frequency Spectrum using FFT")
    ax = fig.add_subplot(111)
    ax.plot(xf, 20 * log10(yf), color='#009440')
    ax.set_xlabel("Frequency (Hz)")
    ax.set_xscale('log')
    ax.set_xlim(10 ** (0), 10 ** (4))
    ax.set_ylabel("Amplitude (dB)")
    ax.set_ylim(-200, 0)
    ax.set_title(u"Frequency Spectrum using FFT")
    ax.grid(True)


def calculate_frequency_spectrum_length_n(signal, n, fs=16000):
    """
    Frequency spectrum generated via Fourier transform (defined length n) using windows

    Parameters
    ----------
    signal : array_like
    n : Length of the Fourier transform, if n > len(sig), sig is zero-padded
    fs : Sample rate of sig

    Returns
    -------
    f : Frequency bin, array_like
    amp : Amplitude values, array_like

    """
    N = len(signal)  # number of signal points
    T = 1.0 / fs
    # Compute FFT
    # w = numpy.ones(N) # no windowing
    w = hanning(N)  # windowing the signal helps mitigate spectral leakage (e.g. Hanning, Hamming, Blackman)
    ywf = fft(signal * w, n)
    xf = numpy.linspace(0.0, 1.0 / (2.0 * T), n / 2)
    # sig is real-valued, the spectrum is symmetric; only take the FFT corresponding to the positive frequencies
    return (xf, 2.0 / n * abs(ywf[0:n / 2]))


def calculate_R(sig, fs, li, ui, l, u):
    """
    Calculate the Ratio Index R of a time serie to characterize the power spectrum property
    See: "Analysis of Vibroarthrographic Signals for Knee Osteoarthtitis Diagnosis"
    (2012 - Lee, Lin , Wu, Wang)

    Parameters
    ----------
    sig : Time serie, array_like
    fs : Sample rate of sig
    li : lower frequency limit of numerator
    ui : upper frequency limit of numerator
    l : lower frequency limit of denominator
    u : upper frequency limit of denominator

    Returns
    -------
    Ratio Index R

    """
    # Frequency spectrum
    (f, yf) = calculate_frequency_spectrum(sig, fs)
    # Define frequency range of numerator, denominator
    f_num = f[(f >= li) & (f <= ui)]
    f_den = f[(f >= l) & (f <= u)]
    # Find the corresponding indexes
    li_ind = numpy.where(f == min(f_num))
    ui_ind = numpy.where(f == max(f_num))
    l_ind = numpy.where(f == min(f_den))
    u_ind = numpy.where(f == max(f_den))
    # Integrate of X(f)
    X_num = yf[li_ind[0][0]:ui_ind[0][0]]  # li to ui
    X_den = yf[l_ind[0][0]:u_ind[0][0]]  # l to u
    numerator = numpy.sum(X_num ** 2)
    denominator = numpy.sum(X_den ** 2)
    R = numerator / denominator
    return (f, yf, R)


def read_certain_columns_in_csv(csvFilename, NumberOfColumns):
    """
    Read only certain colummns in CSV file

    Parameters
    ----------
    csvFilename : string
    NumberOfColumns : array_like

    Returns
    -------
    Output

    """
    output = []
    with open(csvFilename, "rU") as f:  # open the file in read universal mode
        next(f)  # skip header line
        for line in f:
            cells = line.split(",")
            for i in NumberOfColumns:
                output.append((cells[i]).rstrip())
        np_output = numpy.reshape(output, (-1, len(NumberOfColumns))).T  # transpose
    f.close()
    return np_output


def convert_to_libsvm_file(featureTable, label, outputFilename):
    """
    Convert the feature table to libsvm format file,
    The format of training and testing data file is:
    <label> <index1>:<value1> <index2>:<value2> ...
    Each line contains an instance and is ended by a '\n' character,
    For classification, <label> is an integer indicating the class label (multi-class is supported),
    For regression, <label> is the target value which can be any real number

    Parameters
    ----------
    featureTable : numpy array with dimension (No. of features, No. of objects)
    label : string

    Returns
    -------
    Output text file

    """
    (NumberOfFeatures, NumberOfObjects) = featureTable.shape
    with open(outputFilename, 'w') as f:
        for obj in range(NumberOfObjects):
            towrite = label
            towrite += " "
            for feature in range(NumberOfFeatures):
                new = featureTable[feature][obj]
                towrite += str(feature + 1) + ":" + str(new)
                towrite += " "
            towrite += "\n"
            f.write(towrite)
            f.flush
    f.close


def export_certain_columns_to_dataset(csvFilename, NumberOfColumns):
    """
    Export certain colummns in CSV file to a dataset for sklearn

    Parameters
    ----------
    csvFilename : string
    NumberOfColumns : array_like

    Returns
    -------
    Output

    """
    output = []
    with open(csvFilename, 'rU') as f:  # open the file in read universal mode
        next(f)  # skip header line
        for line in f:
            cells = line.split(",")
            for i in NumberOfColumns:
                output.append((cells[i]))
        np_output = numpy.reshape(output, (-1, len(NumberOfColumns)))
        np_output = np_output.astype(numpy.float)  # convert to float
    f.close()
    return np_output


def identify_session_rl_sensorpos_in_dataset(csvFilename):
    """
    Identify session number, RL, positon of sensor in first colummn of dataset (in CSV file)

    Parameters
    ----------
    csvFilename : string

    Returns
    -------
    session, rl, sensorpos

    """
    session = []
    rl = []
    sensorpos = []
    with open(csvFilename, 'rU') as f:  # open the file in read universal mode
        next(f)  # skip header line
        for line in f:
            cells = line.split(",")
            SPLITfilename = str(cells[0])

            SPLITtemp = SPLITfilename.split("Session-")
            SPLITtemp = SPLITtemp[0]
            SPLITelem = SPLITtemp[-1]  # last element before "Session-", should be / or \
            FIS = SPLITfilename.split(SPLITelem)

            FIS_WAVfile = FIS[-1]  # name of *.wav file
            FIS_Sessiontemp = FIS[-2]  # name of Session
            FIS_Session = FIS_Sessiontemp.split("Session-")
            FIS_WAVfiletemp = FIS_WAVfile.split("-")

            if FIS_WAVfiletemp[1] == 'R':
                rl.append("R")
            elif FIS_WAVfiletemp[1] == 'L':
                rl.append("L")
            else:
                raise Exception(
                    'File name in features list does not contain R or L at the expected position. File name: ' + SPLITfilename)

            if FIS_WAVfile.find("Patella") > -1:
                sensorpos.append(0)
            elif FIS_WAVfile.find("TibiaplateauMedial") > -1:
                sensorpos.append(1)
            elif FIS_WAVfile.find("TibiaplateauLateral") > -1:
                sensorpos.append(2)
            else:
                raise Exception(
                    'File name in features list does not contain expected sensor position File name: ' + SPLITfilename)

            session.append(FIS_Session[-1])
        np_session = numpy.reshape(session, (-1, 1))
        np_session = np_session.astype(numpy.float)  # convert to float
        np_rl = numpy.reshape(rl, (-1, 1))
        np_sensorpos = numpy.reshape(sensorpos, (-1, 1))
    f.close()
    return (np_session, np_rl, np_sensorpos)


def create_target_vector_with_label(length, LabelAsNumber):
    """
    Create target array with label for sklearn

    Parameters
    ----------
    length
    LabelAsNumber

    Returns
    -------
    array-like, shape (n_samples,)

    """
    one_vector = numpy.ones((length,), dtype=numpy.int)
    target_vector = numpy.multiply(LabelAsNumber, one_vector)
    return target_vector


def concatinate_datasets(dataset1, dataset2):
    concatinate_dataset = numpy.concatenate((dataset1, dataset2), axis=0)
    return concatinate_dataset


def plt_mean_roc_with_cv(X_data, y_target, classifier, nfold):
    """
    Plot the Receiver Operating Characteristic (ROC) metric to evaluate
    classifier output quality using cross-validation
    See: http://scikit-learn.org/stable/auto_examples/plot_roc_crossval.html

    Parameters
    ----------
    X_data : array-like, shape = [n_samples, n_features]
    y_target : array-like, shape (n_samples,)
    kernelfunction : string, e.g. 'linear', 'rbf' etc.
    nfold

    Returns
    -------
    ROC curve

    """
    # Run classifier with cross-validation and plot ROC curves
    stratified_folds = StratifiedKFold(y_target, n_folds=nfold)

    mean_TruePositiveRate = 0.0
    mean_FalsePositiveRate = numpy.linspace(0, 1, len(y_target))
    color = iter(cm.rainbow(numpy.linspace(0, 1, 10)))

    for i, (train, test) in enumerate(stratified_folds):
        probas_ = classifier.fit(X_data[train], y_target[train]).predict_proba(X_data[test])
        # Compute ROC curve and area the curve
        FalsePositiveRate, TruePositiveRate, thresholds = roc_curve(y_target[test], probas_[:, 1])
        mean_TruePositiveRate += interp(mean_FalsePositiveRate, FalsePositiveRate, TruePositiveRate)
        mean_TruePositiveRate[0] = 0.0
        roc_auc = auc(FalsePositiveRate, TruePositiveRate)
        c = next(color)
        plt.plot(FalsePositiveRate, TruePositiveRate, lw=1, c=c, label='ROC fold %d (area = %0.2f)' % (i, roc_auc))
    plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='Luck')
    mean_TruePositiveRate /= len(stratified_folds)
    mean_TruePositiveRate[-1] = 1.0
    mean_auc = auc(mean_FalsePositiveRate, mean_TruePositiveRate)
    plt.plot(mean_FalsePositiveRate, mean_TruePositiveRate, color='#009440', label='Mean ROC (area = %0.2f)' % mean_auc)
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    plt.show()
    return stratified_folds


def calculate_dwt(data, wavelet='db1'):
    """
    Calculate single level Discrete Wavelet Transform (DWT)

    Parameters
    ----------
    data : input signal
    wavelet : wavelet to use

    Returns
    -------
    Approximation (cA) and detail (cD) coefficients

    """
    cA, cD = pywt.dwt(data, wavelet)
    return cA, cD


def wavelet_decomposition(signal, wavelet='db1'):
    # Perform 1D multilevel Discrete Wavelet Transform decomposition of given signal
    # Return ordered list of coefficients arrays in the form [cA_n, cD_n, cD_n-1, ..., cD2, cD1]
    tree = pywt.wavedec(signal, wavelet)
    # cA_n, cD = tree
    return tree


def select_file_from_parent_folder(parentfolderFullpath, endswithPhrase, containingPhrase):
    listOfSelectedFiles = []
    for session in os.listdir(parentfolderFullpath)[1:]:
        sessionFullpath = os.path.join(parentfolderFullpath, session)
        listOfSelectedFilesInOneSession = select_file_from_folder(sessionFullpath, endswithPhrase, containingPhrase)
        for filename in listOfSelectedFilesInOneSession:
            newElement = os.path.join(session, filename)
            listOfSelectedFiles.append(newElement)
    return listOfSelectedFiles


def select_file_from_folder(sessionFullpath, endswithPhrase, containingPhrase):
    listOfSelectedFilesInOneSession = []
    for vagfile in os.listdir(sessionFullpath):
        if (vagfile.endswith(endswithPhrase)) & (containingPhrase in vagfile):
            # f.write(os.path.join(session,vagfile)+"\n")
            listOfSelectedFilesInOneSession.append(vagfile)
    return listOfSelectedFilesInOneSession


def write_list_to_textfile(outputPathname, outputTXTFilename, openmode, listOfSelectedFiles):
    fout = open(os.path.join(outputPathname, outputTXTFilename), openmode)
    for item in listOfSelectedFiles:
        fout.write("%s\n" % item)
    fout.close()


def return_list_from_textfile(filenameList, absolutePathSessions):
    filesListObj = open(filenameList, 'r')
    filesList = []
    for filename in filesListObj:
        filesList.append(os.path.join(absolutePathSessions, filename.rstrip()))
    filesListObj.close()
    return filesList


def calculate_features_in_list_td(filesList, outputFilename, segmentationVersion, loadIgnoreSegmentsNode=True):
    itemsCount = len(filesList)
    counter = 1
    fout = open(outputFilename, 'w+')
    fout.write("filename,mean,variance,std,rms,zcr_0,zcr_std\n")
    for item in filesList:
        percent = 100.0 * counter / float(itemsCount)
        print(str(percent) + "% - processing: " + item)
        (mean, var, std, rms, zcr_0, zcr_std) = my_time_domain_feature_selection(item, segmentationVersion,
                                                                                 loadIgnoreSegmentsNode)
        features = item + ","
        features += "%.9f,%.9f,%.9f,%.9f,%.9f,%.9f" % (mean, var, std, rms, zcr_0, zcr_std)
        features += "\n"
        fout.write(features)
        fout.flush()
        counter += 1
    fout.close()


def calculate_features_in_list_fd(filesList, outputFilename, segmentationVersion, loadIgnoreSegmentsNode=True):
    itemsCount = len(filesList)
    counter = 1
    fout = open(outputFilename, 'w+')
    fout.write("filename,r_10_50,r_25_320,r_40_140,r_50_500,r_300_600,r_500_8k,r_10_100,r_3k_5k,r_6k_8k\n")
    for item in filesList:
        percent = 100.0 * counter / float(itemsCount)
        print(str(percent) + "% - processing: " + item)
        (r_10_50, r_25_320, r_40_140, r_50_500, r_300_600, r_500_8k, r_10_100, r_3k_5k,
         r_6k_8k) = my_frequency_domain_feature_selection(item, segmentationVersion, loadIgnoreSegmentsNode)
        features = item + ","
        features += "%.9f,%.9f,%.9f,%.9f,%.9f,%.9f,%.9f,%.9f,%.9f" % (
        r_10_50, r_25_320, r_40_140, r_50_500, r_300_600, r_500_8k, r_10_100, r_3k_5k, r_6k_8k)
        features += "\n"
        fout.write(features)
        fout.flush()
        counter += 1
    fout.close()


def my_time_domain_feature_selection(signalWAVFilename, segmentationVersion, loadIgnoreSegmentsNode=True):
    # (vibrationData, angularData, time, fs) = import_vagfile(signalWAVFilename)
    # Get segments
    # seg_ind = get_segment_indices_from_vagfile(signalWAVFilename, segmentationVersion)
    # segments = segment_indices_to_segments(seg_ind)
    # concatinatedSignal = concatinate_segments(vibrationData, segments)

    # Use Ouss's function to filter bad segments
    (vibrationData, angularData) = vag_ouss_get_signal_raw_data(signalWAVFilename)
    # vibrationData = vibrationData.tolist()
    signalXMLFilename = vag_ouss_get_signal_xml_filename(signalWAVFilename)
    segments = vag_ouss_get_signal_xml_segments(signalXMLFilename, segmentationVersion, loadIgnoreSegmentsNode)
    # Concatinate segments
    concatenatedSignal = vag_ouss_concatenate_segments(vibrationData, segments)
    # Normalize signal
    normalizedConcatenatedSignal = vag_ouss_normalize_raw_signal(concatenatedSignal)
    # Calculate mean
    mean = numpy.mean(normalizedConcatenatedSignal)
    # Calculate var
    var = numpy.var(normalizedConcatenatedSignal)
    # Calculate standard deviation
    std = numpy.std(normalizedConcatenatedSignal)
    # Calculate root mean square value - equivalent average energy
    rms = root_mean_square(normalizedConcatenatedSignal)
    # Calculate zero-crossing rate, threshold = 0
    zcr_0 = zero_crossing_rate(normalizedConcatenatedSignal, threshold=0)
    # Calculate zero-crossing rate, threshold = 0.5*std
    # http://www.ncbi.nlm.nih.gov/pubmed/19015987
    zcr_std = zero_crossing_rate(normalizedConcatenatedSignal, threshold=0.5 * std)
    return (mean, var, std, rms, zcr_0, zcr_std)


def my_frequency_domain_feature_selection(signalWAVFilename, segmentationVersion, loadIgnoreSegmentsNode=True):
    # Use Ouss's function to filter bad segments
    (vibrationData, angularData) = vag_ouss_get_signal_raw_data(signalWAVFilename)
    # vibrationData = vibrationData.tolist()
    signalXMLFilename = vag_ouss_get_signal_xml_filename(signalWAVFilename)
    segments = vag_ouss_get_signal_xml_segments(signalXMLFilename, segmentationVersion, loadIgnoreSegmentsNode)
    # Concatinate segments
    concatinatedSignal = vag_ouss_concatenate_segments(vibrationData, segments)
    # Normalize signal
    normalizedConcatinatedSignal = vag_ouss_normalize_raw_signal(concatinatedSignal)
    # Calculate fft
    (frequencies, amplitudes) = vag_ouss_fft_of_signal(normalizedConcatinatedSignal)
    # Calculate sum of frequency bereich 3D
    d = vag_calculate_sum_of_squares_in_band(frequencies, amplitudes, 10, 8000)

    s_10_50 = vag_calculate_sum_of_squares_in_band(frequencies, amplitudes, 10, 50)  # 1. poster
    s_25_320 = vag_calculate_sum_of_squares_in_band(frequencies, amplitudes, 25, 320)  # meniscal acoustics
    s_40_140 = vag_calculate_sum_of_squares_in_band(frequencies, amplitudes, 40, 140)  # patellar clicks
    s_50_500 = vag_calculate_sum_of_squares_in_band(frequencies, amplitudes, 50, 500)  # 1. poster
    s_300_600 = vag_calculate_sum_of_squares_in_band(frequencies, amplitudes, 300, 600)  # McCoy's observation
    s_500_8k = vag_calculate_sum_of_squares_in_band(frequencies, amplitudes, 500, 8000)  # 1. poster
    s_10_100 = vag_calculate_sum_of_squares_in_band(frequencies, amplitudes, 10, 100)  # nam's observation
    s_3k_5k = vag_calculate_sum_of_squares_in_band(frequencies, amplitudes, 3000, 5000)  # nam's observation
    s_6k_8k = vag_calculate_sum_of_squares_in_band(frequencies, amplitudes, 6000, 8000)  # nam's observation

    r_10_50 = s_10_50 / d
    r_25_320 = s_25_320 / d
    r_40_140 = s_40_140 / d
    r_50_500 = s_50_500 / d
    r_300_600 = s_300_600 / d
    r_500_8k = s_500_8k / d
    r_10_100 = s_10_100 / d
    r_3k_5k = s_3k_5k / d
    r_6k_8k = s_6k_8k / d
    return (r_10_50, r_25_320, r_40_140, r_50_500, r_300_600, r_500_8k, r_10_100, r_3k_5k, r_6k_8k)


def root_mean_square(data):
    rms = numpy.sqrt(numpy.mean(numpy.power(data, 2)))
    return rms


def zero_crossing_rate(data, threshold):
    # See http://www.ifs.tuwien.ac.at/~schindler/lectures/MIR_Feature_Extraction.html
    zero_crossings = 0
    NumberOfSamples = len(data)
    for i in range(1, NumberOfSamples):
        if (data[i - 1] < threshold and data[i] > threshold) or \
                (data[i - 1] > threshold and data[i] < threshold) or \
                (data[i - 1] != threshold and data[i] == threshold):
            zero_crossings += 1
    zero_crossing_rate = zero_crossings / float(NumberOfSamples - 1)
    return zero_crossing_rate


def export_features_list_to_csv(filesList, csvFilename):
    fout = open(csvFilename, 'w+')
    csvwriter = csv.writer(fout, )
    for item in filesList:
        csvwriter.writerow([item, ])


def svm_classification_gridsearch_roc(X, y, svmClassifier, score, savefigFilename):
    plt.clf()
    mean_tpr = 0.0
    mean_fpr = numpy.linspace(0, 1, 100)
    ##### PARAMETER ESTIMATION #####
    # Set SVM classifier
    svmClassifier = svm.SVC()
    # Set the parameters by cross-validation
    #    tuned_Parameters = [{'kernel': ['rbf'], 'gamma': numpy.power(10.0,numpy.arange(-5,2)), 'C': numpy.power(10.0,numpy.arange(-1,4))},
    #                        {'kernel': ['poly'], 'degree': [2, 3, 4], 'coef0': numpy.power(10.0,numpy.arange(-2,1)), 'C': numpy.power(10.0,numpy.arange(-1,4))},
    #                        {'kernel': ['linear'], 'C': numpy.power(10.0,numpy.arange(1,4))}
    #                        ]
    # Short list
    tuned_Parameters = [{'kernel': ['rbf'], 'gamma': [1e-2, 1e-3, 1e-4], 'C': [0.1, 1, 10, 100]},
                        {'kernel': ['poly'], 'degree': [2, 3], 'coef0': [1e-1, 1e-2], 'C': [0.1, 1, 10, 100]},
                        {'kernel': ['linear'], 'C': [0.1, 1, 10, 100]}
                        ]
    # Set model evaluation
    # score = 'accuracy' # ‘accuracy’, ‘average_precision’, ‘f1’, ‘precision’, ‘recall’, ‘roc_auc’
    # Set stratified folds to split the dataset in train and test sets.
    stratifiedKfolds = StratifiedKFold(y, n_folds=4)  # 3 folds, 66% training, 33% test
    for i, (train, test) in enumerate(stratifiedKfolds):
        X_train, X_test, y_train, y_test = X[train], X[test], y[train], y[test]
        print("I'm doing GridSearchCV. It can take a long time ...")
        clf = GridSearchCV(estimator=svmClassifier, param_grid=tuned_Parameters, scoring=score)
        print("SIZE X_train")
        print(X_train.shape)
        print("SIZE y_train")
        print(y_train.shape)
        clf.fit(X_train, y_train)
        print("Finish!")
        output = "Best score: " + str(clf.best_score_) + "\n"
        output += "Best parameters set found on development set:\n"
        output += str(clf.best_estimator_) + "\n"
        output += "Grid scores on development set:\n"
        for params, mean_score, scores in clf.grid_scores_:
            output += str(mean_score) + ", " + str(scores.std() / 2) + ", " + str(params) + "\n"
        y_true, y_pred = y_test, clf.predict(X_test)
        output += classification_report(y_true, y_pred)

        # Compute ROC curve and ROC area for each class
        FalsePositiveRate = dict()
        TruePositiveRate = dict()
        roc_auc = dict()
        clf = clf.best_estimator_.set_params(probability=True)
        probas_ = clf.fit(X_train, y_train).predict_proba(X_test)
        FalsePositiveRate, TruePositiveRate, thresholds = roc_curve(y_true, probas_[:, 1])
        mean_tpr += interp(mean_fpr, FalsePositiveRate, TruePositiveRate)
        mean_tpr[0] = 0.0
        roc_auc = auc(FalsePositiveRate, TruePositiveRate)

        plt.plot(FalsePositiveRate, TruePositiveRate, lw=1, label='ROC fold %d (area = %0.2f)' % (i, roc_auc))

    mean_tpr /= len(stratifiedKfolds)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    output += "Mean ROC_AUC: " + str(mean_auc) + "\n"
    plt.plot(mean_fpr, mean_tpr, 'k--', label='Mean ROC (area = %0.2f)' % mean_auc, lw=2)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    plt.savefig(savefigFilename)
    sensitivity = mean_fpr
    specificity = 1 - mean_tpr
    return output, sensitivity, specificity, mean_auc


def shift_list(l, shift, empty=0):
    src_index = max(-shift, 0)
    dst_index = max(shift, 0)
    length = max(len(l) - abs(shift), 0)
    new_l = [empty] * len(l)
    new_l[dst_index:dst_index + length] = l[src_index:src_index + length]
    return new_l


################################################################################
################################################################################
################################################################################
################################################################################
##############################OUSS's FUNCTIONS##################################


def vag_ouss_get_signal_xml_segments(signalXmlFileDialogFilename, segmentationVersion="ONN",
                                     loadIgnoreSegmentsNode=True):
    if not path.isfile(signalXmlFileDialogFilename):
        print("can't find: " + path.basename(signalXmlFileDialogFilename) + "\n")
        return
    # print segmentationVersion
    # open the xml file and get the indices of the segmentation

    # Prepare an empty list that will contain the SegmentsIndices
    segmentsIndices = []

    # Parse the XML file into XML element tree
    xmlElementTree = XMLETreeElementTree.parse(signalXmlFileDialogFilename)
    xmlRoot = xmlElementTree.getroot()
    # Find the element node 'segmentation' of the XML element tree with the right version
    # xmlSegmentationNode = xmlRoot.find('segmentation')
    # print "Segmentation Version:"+xmlSegmentationNode.get('version')
    xmlSegmentationNodeFound = False
    for xmlSegmentationNode in xmlRoot.iter('segmentation'):
        if xmlSegmentationNode is not None:
            if xmlSegmentationNode.attrib['version'] == segmentationVersion:
                xmlSegmentationNodeFound = True
                xmlSegmentationNodeRequestedVersion = xmlSegmentationNode
                break
    if not xmlSegmentationNodeFound:
        print("Could not find Segmentation Node of version: " + segmentationVersion + " for: " + path.basename(
            signalXmlFileDialogFilename))
        return
    # print xmlSegmentationNodeRequestedVersion
    for segment in xmlSegmentationNodeRequestedVersion.findall('segment'):
        segmentBegin = segment.find('begin').text
        segmentEnd = segment.find('end').text
        segmentsIndices.append((int(segmentBegin), int(segmentEnd)))

    if not loadIgnoreSegmentsNode:
        return segmentsIndices
    # ignore segments
    # Read the segments to be ignored from the xml file
    xmlIgnoreNodeFound = False
    for xmlIgnoreNode in xmlRoot.iter('ignore_segments'):
        if xmlIgnoreNode is not None:
            xmlIgnoreNodeFound = True
            xmlIgnoreNodeRequestedVersion = xmlIgnoreNode
            break

    if not xmlIgnoreNodeFound:
        return segmentsIndices

    ignoredSegments = []
    for segmentNode in xmlIgnoreNodeRequestedVersion.findall('segment'):
        index = segmentNode.find('index').text
        ignoredSegments.append(int(index) - 1)

    #    print ignoredSegments

    # remove the ignoredIndices from segmentsIndices and return the latter
    ignoredSegments.sort()
    ignoredSegments.reverse()
    for index in ignoredSegments:
        del segmentsIndices[index]

    #    print segmentsIndices
    return segmentsIndices


def vag_ouss_get_signal_xml_filename(signalWavFileDialogFilename):
    signalWavFileRealname, signalWavFileExtension = path.splitext(signalWavFileDialogFilename)
    signalXmlFileDialogFilename = signalWavFileRealname + ".xml"
    return signalXmlFileDialogFilename


def vag_ouss_concatenate_segments(vibrationRawData, segments):
    # go over segments
    data = []
    for segment in segments:
        # append the right parts of rawData to data
        segmentData = vag_ouss_get_segment_signal(vibrationRawData, segment)
        data = data + segmentData.tolist()
    return data


def vag_ouss_get_segment_signal(rawSignal, segment):
    return rawSignal[segment[0]:segment[1] + 1]


def vag_ouss_get_signal_raw_data(signalWavFileDialogFilename, filterInput=True):
    if not os.path.isfile(signalWavFileDialogFilename):
        print("can't find: " + os.path.basename(signalWavFileDialogFilename) + "\n")
        return
    try:
        # Load the wave file values
        # Import VAG file
        (fs, data) = wavfile.read(signalWavFileDialogFilename)
        if filterInput:
            data[:, 0] = vag_jacqui_high_pass_filter(data[:, 0])
    except:
        print("Could not read the file :" + signalWavFileDialogFilename)
        raise
    # return (vibrationData, angularData)
    return (data[:, 0], data[:, 1])


def vag_jacqui_high_pass_filter(filterInput, fs=16000.0, fg=10.0):
    fs = float(fs)
    fg = float(fg)
    b, a = iirdesign(fg / fs, (fg * 0.05) / fs, 1.0, 40)
    # print "generating a ({:d},{:d}) tap HPF IIR".format(len(b),len(a))
    return lfilter(b, a, filterInput)


def vag_ouss_normalize_raw_signal(signal):
    signalNumpy = numpy.asarray(signal, numpy.float32)
    maxS = numpy.amax(signalNumpy)
    signalNumpy /= maxS
    return signalNumpy


def vag_calculate_sum_of_squares_in_band(frequencies, amplitudes, f_s, f_e):
    theSum = 0
    # Define frequency range of numerator, denominator
    # Find the corresponding indexes
    # Find the corresponding first index, by getting the first index where frequency[i] is bigger that f_s and the last index where frequencies[j] is smaller that f_e
    # Taking advantage that frequencies is sorted in ascending order => user binary search. Complexity of order O(log n)
    # Taking advantage that frequencies is a sorted list with equidistant values delta_f. Complexity of order O(1) SO.... WHO'S THE BOSS ;-) ????
    delta_f = frequencies[1] - frequencies[0]
    indexStart = int(numpy.ceil((f_s - frequencies[0]) / delta_f))
    indexEnd = int(numpy.floor((f_e - frequencies[0]) / delta_f))
    # Integrate of X(f)
    amplitudesSelected = amplitudes[indexStart:indexEnd]  # l to u
    theSum = numpy.sum(amplitudesSelected ** 2)
    return theSum


def vag_ouss_fft_of_signal(signal):
    sampleRate = 16000  # sample/sec
    numberOfSignalPoints = len(signal)  # number of signal points
    # samplesPeriod = 1.0/sampleRate #sample period
    # Compute FFT
    # window = numpy.ones(N) # no windowing
    window = hanning(
        numberOfSignalPoints)  # windowing the signal helps mitigate spectral leakage (e.g. Hanning, Hamming, Blackman)
    SW = fft(signal * window)
    frequencies = numpy.linspace(0.0, sampleRate / 2.0, numberOfSignalPoints / 2)
    # sig is real-valued, t83he spectrum is symmetric; only take the FFT corresponding to the positive frequencies
    return (frequencies.tolist(), (2.0 / numberOfSignalPoints) * abs(SW[0:numberOfSignalPoints / 2]))
