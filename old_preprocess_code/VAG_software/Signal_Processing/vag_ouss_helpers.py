#!/usr/bin/env python

#Filename manipulations
import os

def vag_ouss_get_signal_xml_filename(signalWavFileDialogFilename):
    signalWavFileRealname, signalWavFileExtension = os.path.splitext(signalWavFileDialogFilename)
    signalXmlFileDialogFilename = signalWavFileRealname + ".xml"
    return signalXmlFileDialogFilename
    
def vag_ouss_process_a_list_of_items(theList, processor, arguments=None):
    itemsCount = len(theList)
    counter = 1
    for item in theList:
        percent = 100.0 * counter/float(itemsCount)
        print str(percent)+"% - processing: " + item
        processor(item, arguments)
        counter = counter+1
    return
    
def vag_ouss_get_files_list(filesListFileDialogFilename,prefix=""):
    filesListFile = open(filesListFileDialogFilename,'r')
    filesList = []
    for filename in filesListFile:
        filesList.append(prefix + filename.rstrip())
    filesListFile.close()
    return filesList