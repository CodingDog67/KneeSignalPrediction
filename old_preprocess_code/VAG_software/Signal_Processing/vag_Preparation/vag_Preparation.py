#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Make a list of all filenames, prepare for the signal analysis
Author: Tuan Nam Le
Last modified: 18/11/2014
"""
import os
from PySide.QtGui import QFileDialog, QApplication
from vaghelpers import *

if __name__ == '__main__':

    metadataDir = QFileDialog.getExistingDirectory(caption ="Select directory")
    outputPathname = "/Users/admin"
    outputTXTFilename = "Study02_Filelist_Patients_OnlyForDKOU.txt"
    openmode = "w+" # create file if not exists

    # Open a file object for writing the filenames
    fout = open(os.path.join(outputPathname, outputTXTFilename), openmode)
    listOfSelectedFiles = select_file_from_parent_folder(metadataDir, endswithPhrase=".wav", containingPhrase="")
    write_list_to_textfile(outputPathname, outputTXTFilename, openmode, listOfSelectedFiles)
