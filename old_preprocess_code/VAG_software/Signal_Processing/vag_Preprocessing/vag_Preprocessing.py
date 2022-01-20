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
#import matplotlib.pyplot as plt
#import numpy

from vaghelpers import *
from vag_ouss_helpers import *
from vag_ouss_helpers_segments import *
from ouss_analysis import *

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
    absoluteDataPath = "/Users/admin/Desktop/walther/vag/Measurements_ASK_SEG_PREPROCESSED_walther/"
    filesListwithPath = return_list_from_textfile(filenameList, absoluteDataPath)
    #filesListOnly = return_list_from_textfile(filenameList, '')

    ## Check if all signale have segmentation
    for i in filesListwithPath:
        signalXmlFileDialogFilename = vag_ouss_get_signal_xml_filename(i)
        result = vag_ouss_signal_xml_has_segmentation(signalXmlFileDialogFilename, segmentationVersion="ONN")
        if result is False:
            print "Segmentation not found: " + signalXmlFileDialogFilename

    # Divide each segment to extension and flexion
    vag_ouss_segment_flexion_extension_v1_process_files_list(filesListwithPath)

    vag_ouss_find_all_bad_segments_and_append_to_signal_xml_file(filenameList,absoluteDataPath,"ONN")

#vag_ouss_append_ignored_segments_to_signal_xml_file
