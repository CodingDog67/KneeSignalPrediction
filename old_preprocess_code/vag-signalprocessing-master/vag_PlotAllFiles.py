#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Plot all files in selected folder, save the figures to PNG files
Author: Tuan Nam Le
Last modified: 21 August 2014
"""

from scipy.io import wavfile
from PySide2.QtGui import QFileDialog, QApplication
import matplotlib.pyplot as plt
import numpy
import os
import vaghelpers

if __name__ == '__main__':
    
    try:
        QApplication([])
    except RuntimeError as e:
        pass
    vagDir = QFileDialog.getExistingDirectory(caption ="Select session directory")
    
    fcounter = 0
    for filename in os.listdir(vagDir):
        if filename.endswith(".wav"):
            fcounter = fcounter+1
            print filename
            (realname, extension) = os.path.splitext(filename)
            # Import VAG file
            (fs, samples) = wavfile.read(os.path.join(vagDir, filename))
            # scipy.io.wavread does not support 32-bit float files
            vagsamples = vaghelpers.vag2float(samples, numpy.float32)
            # Separate signal and angular values for segmentation
            signal = vagsamples[:,0]
            angles = vagsamples[:,1]
            time = numpy.linspace(0,numpy.divide(float(len(signal)),fs),len(signal))

            fig = plt.figure(u"Vibration and Angular")
            fig.suptitle(filename, fontsize=10)
    
            ax1 = plt.subplot(2,1,1)
            plt.plot(time, signal, color='b')
            plt.xlabel(u"Time (s)")
            plt.ylabel(u"Vibration (a.u.)")
            ax1.set_xlim([0, max(time)])
            ax1.set_ylim([-1.0, 1.0])
            plt.grid()

            ax2 = plt.subplot(2,1,2)
            plt.plot(time, angles, color='r')
            plt.xlabel(u"Time (s)")
            plt.ylabel(u"Angular (a.u.)")
            ax2.set_xlim([0, max(time)])
            ax2.set_ylim([0, 1.0])
            plt.grid()
            
            plt.savefig(realname)
            plt.clf()
            
    print ("Files counter = " + str(fcounter))