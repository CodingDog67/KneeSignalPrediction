#!/usr/bin/env python
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import scipy

import vag_ouss_helpers as oussHelps
import vag_ouss_helpers_signals as oussSignals
import vag_ouss_helpers_segments as oussSegments

import os.path

def vag_ouss_close_plots():
    plt.close('all')
    
def vag_ouss_prepare_plots():
    plt.ion()
    
def vag_ouss_plot_fft_log(frequencies, amplitudes, offset=0, color="blue", alpha=0.5):
    plt.plot(frequencies[offset:],20*scipy.log10(amplitudes[offset:]))
    plt.show()
    return

def vag_ouss_plot_time_signal(figure, signal):
    plt.plot(signal)
    plt.show()
    return figure
    
def vag_ouss_plot_segments(figure, segments, color='red', alpha=0.5, scaling=1.0):
    #Interate over segments
    for segment in segments:
#       print segment
        scaledSegment = [x/scaling for x in segment]
        vag_ouss_plot_segment(figure, scaledSegment, color, alpha)
    plt.show()
    return figure
    

def vag_ouss_plot_segment(figure, segment, color, alpha=0.5):
    (ax1, ax2) = figure.axes
    ax1YAxisRange = ax1.yaxis.get_view_interval()
    ax2YAxisRange = ax2.yaxis.get_view_interval()
#    xlim = (ax1.dataLim.bounds[0], ax1.dataLim.bounds[2])
#    ax1.set_xlim(xlim)
#    ax2.set_xlim(xlim)
    ax1Height = ax1YAxisRange[1]-ax1YAxisRange[0]
    ax2Height = ax2YAxisRange[1]-ax2YAxisRange[0]
    segmentWidth = segment[1]-segment[0]
    ax1Rect = patches.Rectangle((segment[0],ax1YAxisRange[0]), segmentWidth, ax1Height)
    ax1Rect.set_alpha(alpha)
    ax1Rect.set_color(color)
    ax1.add_patch(ax1Rect)

    ax2Rect = patches.Rectangle((segment[0],ax2YAxisRange[0]), segmentWidth, ax2Height)
    ax2Rect.set_alpha(0.5)
    ax2Rect.set_color(color)
    ax2.add_patch(ax2Rect)
    return figure
    
def vag_ouss_plot_signal(vibrationData, angularData, title=""):
    #(vibrationData, angularData) = vag_ouss_get_signal_raw_data(signalWavFileDialogFilename)
    # Two subplots, the axes array is 1-d
    figure, (ax1, ax2) = plt.subplots(2, sharex=True)
    figure.suptitle(title, fontsize=9)
#    figure.set_title(OsPath.basename(signalWavFileDialogFilename))
    ax1.plot(vibrationData)
    ax1.set_title('Vibration sensor signal')
    ax1.grid(True)

    ax2.plot(angularData)
    ax2.set_title('Potentiometer output')
    ax2.grid(True)

    xlim = (ax1.dataLim.bounds[0], ax1.dataLim.bounds[2])
    ax1.set_xlim(xlim)
    ax2.set_xlim(xlim)
    plt.ion()
    plt.show()
    return figure
    
