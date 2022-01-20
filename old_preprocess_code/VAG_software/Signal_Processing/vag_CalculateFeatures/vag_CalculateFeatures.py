#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
-------------------------------
Vibroarthrography (VAG) Project
-------------------------------
Perform my signal analysis

Author: Tuan Nam Le
Last modified: 03/12/2014
"""
from os import path
from PySide.QtGui import QFileDialog
import matplotlib.pyplot as plt
from vaghelpers import *
import numpy

if __name__ == '__main__':

#    ##### SELECT RAW VAGFILE #####
#    vagFile = QFileDialog.getOpenFileName(caption ="Open WAV File", filter = "*.wav")
#    vagFile = vagFile[0]  # get the filename
#    vagFileRealname, vagFileExtension = path.splitext(vagFile)
#    xmlFileRealname = vagFileRealname + ".xml"
#
#    (vibration, angular, time, fs) = import_vagfile(vagFile)
#    normalizedVibration = normalize_raw_signal(vibration)
#    normalizedAngular = normalize_raw_signal(angular)

    filenameListDialog = QFileDialog.getOpenFileName(caption ="Open filename list", filter = "*.txt")
    filenameList = filenameListDialog[0]  # get the list of filename
    absoluteDataPath = "/Users/admin/Desktop/walther/vag/Measurements_SEG_PREPROCESSED/"
    filesListwithPath = return_list_from_textfile(filenameList, absoluteDataPath)

    resultFeaturesEXTENSIONFilename = "DKOU_Patients_test_EXTENSION.txt"
    resultFeaturesFLEXIONFilename = "DKOU_Patients_test_FLEXION.txt"
    absoluteOutputPath = "/Users/admin/Desktop/walther/vag/Results/GOTSws103/"

    # Calculate time domain features
    #calculate_features_in_list_td(filesListwithPath, os.path.join(absoluteOutputPath, resultFeaturesEXTENSIONFilename), segmentationVersion="EXTENSION", loadIgnoreSegmentsNode=True)
    #calculate_features_in_list_td(filesListwithPath, os.path.join(absoluteOutputPath, resultFeaturesFLEXIONFilename), segmentationVersion="FLEXION", loadIgnoreSegmentsNode=True)

    # Calcutale frequency domain features
    #calculate_features_in_list_fd(filesListwithPath, os.path.join(absoluteOutputPath, resultFeaturesEXTENSIONFilename), segmentationVersion="EXTENSION", loadIgnoreSegmentsNode=True)
    calculate_features_in_list_fd(filesListwithPath, os.path.join(absoluteOutputPath, resultFeaturesFLEXIONFilename), segmentationVersion="FLEXION", loadIgnoreSegmentsNode=True)
