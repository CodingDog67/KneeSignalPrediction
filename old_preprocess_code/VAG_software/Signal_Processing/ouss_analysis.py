#Foreach Signal in All Signals
#   Foreach segment in All Segments

import sys
#For the file common dialog
from PySide.QtGui import QFileDialog
#Filename manipulations
from os import path as OsPath
#Parsing and Manipulating xml files
import xml.etree.ElementTree as XMLETreeElementTree
import numpy
#creating matlab style plots
import matplotlib.pyplot as plt
#import matplotlib.transforms as transforms
#from mpl_toolkits.mplot3d import Axes3D
#fft
import scipy

import vag_ouss_helpers as oussHelps
import vag_ouss_helpers_plotting as oussPlots
import vag_ouss_helpers_segments as oussSegments
import vag_ouss_helpers_signals as oussSignals

def vag_ouss_fft_significance_padding_EXTENSION(signalWavFileDialogFilename):
    return vag_ouss_fft_significance_padding(signalWavFileDialogFilename,"EXTENSION")

def vag_ouss_fft_significance_padding_FLEXION(signalWavFileDialogFilename):
    return vag_ouss_fft_significance_padding(signalWavFileDialogFilename,"FLEXION")

def vag_ouss_fft_significance_resampling(signalWavFileDialogFilename,segmentationVersion="FLEXION"):
    signalXmlFileDialogFilename = oussHelps.vag_ouss_get_signal_xml_filename(signalWavFileDialogFilename)
#    print signalXmlFileDialogFilename
    (vibrationRawData, angularData) = oussSignals.vag_ouss_get_signal_raw_data(signalWavFileDialogFilename)
    vibrationRawData = vibrationRawData.tolist()
    segments = oussSegments.vag_ouss_get_signal_xml_segments(signalXmlFileDialogFilename,segmentationVersion)
#    print segments
    signals = oussSignals.vag_ouss_arrange_segments_signals_in_a_list(vibrationRawData, segments)
    shortestSegmentIndex = oussSegments.vag_ouss_find_shortest_segment(segments)
    FFTs = oussPlots.vag_ouss_rfft_of_signals(signals,16000.0)
    destinationSamplesCount = len(FFTs[shortestSegmentIndex][0])
    resampledFFTs = oussSignals.vag_ouss_resample_ffts(FFTs,destinationSamplesCount,16000.0)
#    sys.exit()
    significance = oussSignals.vag_ouss_sum_of_fft_of_signals(resampledFFTs)
    plt.figure()
    oussSignals.vag_ouss_plot_fft_log(resampledFFTs[0][0], significance)
    plt.show()
    return significance

def vag_ouss_fft_significance_padding(signalWavFileDialogFilename,segmentationVersion="FLEXION"):
    signalXmlFileDialogFilename = oussHelps.vag_ouss_get_signal_xml_filename(signalWavFileDialogFilename)
    (vibrationRawData, angularData) = oussSignals.vag_ouss_get_signal_raw_data(signalWavFileDialogFilename)
    vibrationRawData = vibrationRawData.tolist()
    #figure = plt.figure()
    #plot EXTENSION Significance
#    segmentsEXTENSION = vag_ouss_get_signal_xml_segments_EXTENSION(signalXmlFileDialogFilename)
#    signalsEXTENSION= vag_ouss_arrange_segments_signals_in_a_list(vibrationRawData, segmentsEXTENSION)
#    signalsEXTENSION  = vag_ouss_padd_to_same_length(signalsEXTENSION)
#    FFTs = vag_ouss_fft_of_signals(signalsEXTENSION)
#    significanceEXTENSION = vag_ouss_sum_of_fft_of_signals(FFTs)
#    vag_ouss_plot_fft_log(FFTs[0][0][0::6], significanceEXTENSION[0::6] , 100)
    #plot FLEXION Significance
    segments = oussSegments.vag_ouss_get_signal_xml_segments(signalXmlFileDialogFilename,segmentationVersion)
    signals = oussSignals.vag_ouss_arrange_segments_signals_in_a_list(vibrationRawData, segments)
    signals = oussSignals.vag_ouss_padd_to_same_length(signals)
    FFTs = oussSignals.vag_ouss_rfft_of_signals(signals)
    significance = oussSignals.vag_ouss_sum_of_fft_of_signals(FFTs)
    oussPlots.vag_ouss_plot_fft_log(FFTs[0][0], significance)
    return significance

def vag_ouss_variances_of_segments_EXTENSION(signalWavFileDialogFilename):
    signalXmlFileDialogFilename = oussHelps.vag_ouss_get_signal_xml_filename(signalWavFileDialogFilename)
    (vibrationRawData, angularData) = oussSignals.vag_ouss_get_signal_raw_data(signalWavFileDialogFilename)
    vibrationRawData = vibrationRawData.tolist()
    segmentsEXTENSION = oussSegments.vag_ouss_get_signal_xml_segments_FLEXION(signalXmlFileDialogFilename)
    signalsEXTENSION= oussSignals.vag_ouss_arrange_segments_signals_in_a_list(vibrationRawData, segmentsEXTENSION)
    vag_ouss_segments_variances(signalsEXTENSION)
    return

def vag_ouss_variances_of_segments_FLEXION(signalWavFileDialogFilename):
    signalXmlFileDialogFilename = oussHelps.vag_ouss_get_signal_xml_filename(signalWavFileDialogFilename)
    (vibrationRawData, angularData) = oussSignals.vag_ouss_get_signal_raw_data(signalWavFileDialogFilename)
    vibrationRawData = vibrationRawData.tolist()
    segmentsFLEXION = oussSegments.vag_ouss_get_signal_xml_segments_FLEXION(signalXmlFileDialogFilename)
    signalsFLEXION = oussSignals.vag_ouss_arrange_segments_signals_in_a_list(vibrationRawData, segmentsFLEXION)
    vag_ouss_segments_variances(signalsFLEXION)
    return

def vag_ouss_segments_variances(segments):
    variances = []
    plt.figure()
    for segment in segments:
        variances.append(numpy.var(segment))
    plt.plot(variances)
    return variances

def vag_calculate_sum_of_squares_in_band(frequencies, amplitudes, f_s, f_e):
    theSum = 0
    # Define frequency range of numerator, denominator
    # Find the corresponding indexes
    #Find the corresponding first index, by getting the first index where frequency[i] is bigger that f_s and the last index where frequencies[j] is smaller that f_e
    #Taking advantage taht frequencies is sorted in ascending order => user binary search. Complexity of order O(log n)
    #Taking advantage that frequencies is a sorted list with equidistant values delta_f. Complexity of order O(1) SO.... WHO'S THE BOSS ;-) ????
    delta_f = frequencies[1]-frequencies[0]
    indexStart = int(numpy.ceil((f_s-frequencies[0])/delta_f))
    indexEnd = int(numpy.floor((f_e-frequencies[0])/delta_f))
    # Integrate of X(f)
    amplitudesSelected = amplitudes[indexStart:indexEnd] # l to u
    theSum = numpy.sum(amplitudesSelected**2)
    return theSum

def vag_ouss_calculate_average_segment_signal(signals,destinationSamplesCount):
    resampledSignals = oussSignals.vag_ouss_resample_signals(signals,destinationSamplesCount)
    averageSegmentSignal = numpy.mean(resampledSignals, axis=0)
    return averageSegmentSignal

def vag_ouss_calculate_time_domain_features_of_signal(signal):
    # Calculate mean
    mean = numpy.mean(signal)
    # Calculate var
    var = numpy.var(signal)
    # Calculate standard deviation
    std = numpy.std(signal)
    # Calculate root mean square value - equivalent average energy
    rms = oussSignals.vag_nam_root_mean_square(signal)
    # Calculate zero-crossing rate, threshold = 0
    zcr_0 = oussSignals.vag_nam_zero_crossing_rate(signal, threshold=0)
    # Calculate zero-crossing rate, threshold = 0.5*std
    # http://www.ncbi.nlm.nih.gov/pubmed/19015987
    zcr_std = oussSignals.vag_nam_zero_crossing_rate(signal, threshold=0.5*std)
    return (mean, var, std, rms, zcr_0, zcr_std)

def vag_ouss_calculate_time_domain_features_of_averaged_segment(signalWavFileDialogFilename, segmentor):
    signalXmlFileDialogFilename = oussHelps.vag_ouss_get_signal_xml_filename(signalWavFileDialogFilename)
    (vibrationRawData, angularData) = oussSignals.vag_ouss_get_signal_raw_data(signalWavFileDialogFilename)
    #read segments from the xml file depending on segments reader function: segmentor
    segments = segmentor(signalXmlFileDialogFilename)
    signals = oussSignals.vag_ouss_arrange_segments_signals_in_a_list(vibrationRawData, segments)
    #time domain average segment
    destinationSamplesCount = oussSegments.vag_ouss_get_shortest_segment_samples_count(segments)
    averagedSegmentSignal = vag_ouss_calculate_average_segment_signal(signals,destinationSamplesCount)
    return vag_ouss_calculate_time_domain_features_of_signal(averagedSegmentSignal)

def vag_ouss_calculate_time_domain_features_of_concatinated_segments(signalWavFileDialogFilename, segmentor):
    signalXmlFileDialogFilename = oussHelps.vag_ouss_get_signal_xml_filename(signalWavFileDialogFilename)
    (vibrationRawData, angularData) = oussSignals.vag_ouss_get_signal_raw_data(signalWavFileDialogFilename)
    segments = segmentor(signalXmlFileDialogFilename)
    # Concatinate segments
    concatinatedSignal = oussSignals.vag_ouss_concatinate_segments(vibrationRawData, segments)
    # Normalize signal
    concatinatedSignal = oussSignals.vag_ouss_normalize_raw_signal(concatinatedSignal)
    return vag_ouss_calculate_time_domain_features_of_signal(concatinatedSignal)

def vag_ouss_calculate_frequency_domain_features_of_averaged_segment(signalWavFileDialogFilename, segmentor):
    signalXmlFileDialogFilename = oussHelps.vag_ouss_get_signal_xml_filename(signalWavFileDialogFilename)
    (vibrationRawData, angularData) = oussSignals.vag_ouss_get_signal_raw_data(signalWavFileDialogFilename)
    #read segments from the xml file depending on segments reader function: segmentor
    segments = segmentor(signalXmlFileDialogFilename)
    signals = oussSignals.vag_ouss_arrange_segments_signals_in_a_list(vibrationRawData, segments)
    #time domain average segment
    destinationSamplesCount = oussSegments.vag_ouss_get_shortest_segment_samples_count(segments)
    #Generate Template
    averagedSegmentSignal = vag_ouss_calculate_average_segment_signal(signals,destinationSamplesCount)

    return vag_ouss_calculate_frequency_domain_features_of_signal(averagedSegmentSignal)

def vag_ouss_calculate_frequency_domain_features_of_signal(signal):
    (frequencies, amplitudes) = oussSignals.vag_ouss_fft_of_signal(signal)
    sum_all = vag_calculate_sum_of_squares_in_band(frequencies, amplitudes, 10, 8000)
    r_10_50= vag_calculate_sum_of_squares_in_band(frequencies, amplitudes, 10, 50)/sum_all
    r_25_320=vag_calculate_sum_of_squares_in_band(frequencies, amplitudes, 25, 320)/sum_all
    r_40_140=vag_calculate_sum_of_squares_in_band(frequencies, amplitudes, 40, 140)/sum_all
    r_50_500=vag_calculate_sum_of_squares_in_band(frequencies, amplitudes, 50, 500)/sum_all
    r_300_600=vag_calculate_sum_of_squares_in_band(frequencies, amplitudes, 300, 600)/sum_all
    r_500_8k=vag_calculate_sum_of_squares_in_band(frequencies, amplitudes, 500, 8000)/sum_all
    r_10_100=vag_calculate_sum_of_squares_in_band(frequencies, amplitudes, 10, 100)/sum_all
    r_3k_5k=vag_calculate_sum_of_squares_in_band(frequencies, amplitudes, 3000, 5000)/sum_all
    r_6k_8k=vag_calculate_sum_of_squares_in_band(frequencies, amplitudes, 6000, 8000)/sum_all
    return (r_10_50,r_25_320,r_40_140,r_50_500,r_300_600,r_500_8k,r_10_100,r_3k_5k,r_6k_8k)

def vag_ouss_skyfall(signalWavFileDialogFilename, segmentor):
    #get signal
    signalXmlFileDialogFilename = oussHelps.vag_ouss_get_signal_xml_filename(signalWavFileDialogFilename)
    (vibrationRawData, angularData) = oussSignals.vag_ouss_get_signal_raw_data(signalWavFileDialogFilename)
    #get segments ONN
    segments = segmentor(signalXmlFileDialogFilename)
    #concatinatesegments
    concatinatedSignal = oussSignals.vag_ouss_concatinate_segments(vibrationRawData, segments)
    #vag_ouss_plot_signal(concatinatedSignal)
    #normalize signal
    normalizedConcatinatedSignal = oussSignals.vag_ouss_normalize_raw_signal(concatinatedSignal)
    #calculate mean
    mean = numpy.mean(normalizedConcatinatedSignal)
    #calculate var
    var = numpy.var(normalizedConcatinatedSignal)
    #calculate std
    std = numpy.std(normalizedConcatinatedSignal)
    #calculate fft
    (frequencies, amplitudes) = oussSignals.vag_ouss_fft_of_signal(normalizedConcatinatedSignal)
    #calculate sum of frequency bereich 3D
    d = vag_calculate_sum_of_squares_in_band(frequencies, amplitudes, 10, 8000)
    sum_1 = vag_calculate_sum_of_squares_in_band(frequencies, amplitudes, 10, 50)
    sum_2 = vag_calculate_sum_of_squares_in_band(frequencies, amplitudes, 50, 500)
    sum_3 = vag_calculate_sum_of_squares_in_band(frequencies, amplitudes, 500, 8000)
    r_1 = sum_1/d
    r_2 = sum_2/d
    r_3 = sum_3/d
    return (mean, var, std, sum_1, sum_2, sum_3, d, r_1, r_2, r_3)

def vag_ouss_skyfall_of_list(theList, logFile, segmentor):
    itemsCount = len(theList)
    counter = 1
    fout = open(logFile, 'w')
    fout.write("mean, variance, std, sum_10-50, sum_50-500, sum_500-8000, sum_10-8000, r_10-50,r_50-500,r_500-8000,\n")
    #print "mean, variance, std, sum_10-50, sum_50-500, sum_500-8000, sum_10-8000, r_10-50,r_50-500,r_500-8000,\n"
    for item in theList:
        percent = 100.0 * counter/float(itemsCount)
        print str(percent)+"% - processing: " + item
        (mean, var, std, sum_1, sum_2, sum_3, d, r_1, r_2, r_3)=vag_ouss_skyfall(item, segmentor)
        print item
        features = item + ","
        features += "%.9f,%.9f,%9f" % (mean, var, std)
        features += "%.9f,%.9f,%9f,%9f" % (sum_1, sum_2, sum_3, d)
        features += "%.9f,%.9f,%9f" % (r_1, r_2, r_3)
        features += "\n"
        print features
        fout.write(features)
        fout.flush
        counter = counter+1
    fout.close()
    return

def vag_ouss_find_all_bad_segments_and_append_to_signal_xml_file(signalsListFileDialogFilename, prefix="", segmentationVersion="ONN", loadIgnoreSegmentsNode=False, bitsLimit=16, maxVarianceRatio = 2.0, minimumDuration = 2.0):
    filesList = oussHelps.vag_ouss_get_files_list(signalsListFileDialogFilename, prefix)
    #"/home/ouss/Dropbox/Projects/LMU/vag/VAG_Paper_1/Data/"
    for afile in filesList:
        #print afile
        (vibrationRawData, angularData) = oussSignals.vag_ouss_get_signal_raw_data(afile)
        vibrationRawData = vibrationRawData.tolist()
        signalXmlFileDialogFilename = oussHelps.vag_ouss_get_signal_xml_filename(afile)
        segments = oussSegments.vag_ouss_get_signal_xml_segments(signalXmlFileDialogFilename,segmentationVersion,loadIgnoreSegmentsNode)
        signals = oussSignals.vag_ouss_arrange_segments_signals_in_a_list(vibrationRawData, segments)

        indicesSaturated = []
        indicesSaturated = oussSegments.vag_ouss_find_saturated_segments(signals,bitsLimit)
        #indicesSaturated.sort()
        #indicesSaturated.reverse()
        #for index in indicesSaturated:
        #    del signals[index]
        #    del segments[index]

        if len(indicesSaturated)>0:
            print OsPath.basename(afile)," has saturated segments: ", indicesSaturated

        indicesNoisy = []
        indicesNoisy = oussSegments.vag_ouss_find_noisy_segments_concatinated(signals,maxVarianceRatio)
        #indicesNoisy.sort()
        #indicesNoisy.reverse()
        #for index in indicesNoisy:
        #    del signals[index]
        #    del segments[index]

        if len(indicesNoisy)>0:
            print OsPath.basename(afile)," has noisy segments: ", indicesNoisy

        indicesShort = []
        indicesShort = oussSegments.vag_ouss_find_short_segments(signals,minimumDuration)
        #indicesShort.sort()
        #indicesShort.reverse()
        #for index in indicesShort:
        #    del signals[index]
        #    del segments[index]
        if len(indicesShort)>0:
            print OsPath.basename(afile)," has short segments: ", indicesShort

        ignoredSegmentsIndices = []
        ignoredSegmentsIndices = indicesSaturated + indicesShort + indicesNoisy

        if len(ignoredSegmentsIndices)>0:
#            print ignoredSegmentsIndices
            ignoredSegmentsIndices = list(set(ignoredSegmentsIndices))
            ignoredSegmentsIndices.sort()
            ignoredSegmentsIndices.reverse()
#            print ignoredSegmentsIndices
            for index in ignoredSegmentsIndices:
                del signals[index]
                del segments[index]

#            fig = vag_ouss_plot_signal(afile,OsPath.basename(afile))
#            vag_ouss_plot_segments(fig, segments)
            ignoredSegmentsIndices.reverse()
            oussSegments.vag_ouss_append_ignored_segments_to_signal_xml_file(signalXmlFileDialogFilename, ignoredSegmentsIndices, True)
        else:
            print OsPath.basename(afile)," is clean."
    return

import csv

def vag_ouss_import_features(featuresListFileDialogFilename):
    #open the file
    features = []
    with open(featuresListFileDialogFilename , 'rb') as csvfile:
        featuresReader = csv.DictReader(csvfile)
#        print featuresReader.fieldnames
        for row in featuresReader:
           features.append(row)
    return features

def vag_ouss_calculate_and_export_time_domain_features(filesList, segmentor, timeDomainFeaturesCalculator, outputFilename):
    itemsCount = len(filesList)
    counter = 1
    fout = open(outputFilename, 'w')
    fout.write("filename,mean,variance,std,rms,zcr_0,zcr_std\n")
    for item in filesList:
        percent = 100.0 * counter/float(itemsCount)
        print str(percent)+"% - processing: " + item
        (mean, var, std, rms, zcr_0, zcr_std) = timeDomainFeaturesCalculator(item, segmentor)
        features = item + ","
        features += "%.9f,%.9f,%.9f,%.9f,%.9f,%.9f" % (mean, var, std, rms, zcr_0, zcr_std)
        features += "\n"
        fout.write(features)
        fout.flush()
        counter=counter+1
    fout.close()
    return

def vag_ouss_calculate_and_export_frequency_domain_features(filesList, segmentor, frequencyDomainFeaturesCalculator, outputFilename):
    itemsCount = len(filesList)
    counter = 1
    fout = open(outputFilename, 'w')
    fout.write("filename,r_10_50,r_25_320,r_40_140,r_50_500,r_300_600,r_500_8k,r_10_100,r_3k_5k,r_6k_8k\n")
    for item in filesList:
        percent = 100.0 * counter/float(itemsCount)
        print str(percent)+"% - processing: " + item
        (r_10_50,r_25_320,r_40_140,r_50_500,r_300_600,r_500_8k,r_10_100,r_3k_5k,r_6k_8k) = frequencyDomainFeaturesCalculator(item, segmentor)
        features = item + ","
        features += "%.9f,%.9f,%.9f,%.9f,%.9f,%.9f,%.9f,%.9f,%.9f" % (r_10_50,r_25_320,r_40_140,r_50_500,r_300_600,r_500_8k,r_10_100,r_3k_5k,r_6k_8k)
        features += "\n"
        fout.write(features)
        fout.flush()
        counter=counter+1
    fout.close()
    return

def vag_ouss_merge_two_lists_of_dictionaries_depending_on_key(listOne,listTwo,key,overwrite=True):
    mergedList = []
    #check if key exists in both lists of dictionary
    if not key in listOne[0]:
        print 'Key does not exists in the first dictionary list\n'
        return False
    if not key in listTwo[0]:
        print 'Key does not exists in the first dictionary list\n'
        return False

    #go over the first list
    for rowInListOne in listOne:
        #get the value of the key
        keyValue = rowInListOne[key]
        #go over the second list and find the row that corresponds to the same key value
        for rowInListTwo in listTwo:
            if rowInListTwo[key] == keyValue:
                rowInMergedList = dict(rowInListOne.items() + rowInListTwo.items())
                mergedList.append(rowInMergedList)
    return mergedList

if __name__ == '__main__':
    oussPlots.vag_ouss_close_plots()
    oussPlots.vag_ouss_prepare_plots()

#    signalsListFileDialogFilename, signalsListFileDialogFilter = QFileDialog.getOpenFileName(caption ="Select *.txt file", filter = "*.txt")
#    signalsListFileRealname, signalsListFileExtension = OsPath.splitext(signalsListFileDialogFilename)
#    vag_ouss_find_all_bad_segments("/home/ouss/Dropbox/Projects/LMU/vag_git_repository/VAG_Paper_1/Filelist/Filelist_All_Patella.txt","/home/ouss/Dropbox/Projects/LMU/vag_git_repository/VAG_Paper_1/Data/")

###    vag_ouss_find_all_bad_segments_and_append_to_signal_xml_file("/home/ouss/Dropbox/Projects/LMU/vag_git_repository/VAG_Paper_1/Filelist/Filelist_All.txt","/home/ouss/Dropbox/Projects/LMU/vag_git_repository/VAG_Paper_1/Data/","ONN")
###    vag_ouss_segment_flexion_extension_v1("/home/ouss/Dropbox/Projects/LMU/vag_git_repository/VAG_Paper_1/Filelist/Filelist_All.txt","/home/ouss/Dropbox/Projects/LMU/vag_git_repository/VAG_Paper_1/Data/")

#    signalWavFileDialogFilename, signalWavDialogFilter = QFileDialog.getOpenFileName(caption ="Select *.wav file", filter = "*.wav")
#    print vag_ouss_calculate_time_domain_features_of_averaged_segment(signalWavFileDialogFilename, vag_ouss_get_signal_xml_segments_FLEXION)
#    print vag_ouss_calculate_time_domain_features_of_concatinated_segments(signalWavFileDialogFilename, vag_ouss_get_signal_xml_segments_FLEXION)
#    sys.exit()
#####To Do
#   http://scikit-learn.org/stable/auto_examples/datasets/plot_iris_dataset.html

#filename,r_10_50,r_25_320,r_40_140,r_50_500,r_300_600,r_500_8k,r_10_100,r_3k_5k,r_6k_8k
#filename,mean,variance,std,rms,zcr_0,zcr_std
#    f1 = vag_ouss_import_features('/home/ouss/Dropbox/Projects/LMU/vag_git_repository/VAG_Paper_1/Result/Filelist_Healthy_All_Result_TimeDomainFeatures_EXTENSION.csv')
#    f2 = vag_ouss_import_features('/home/ouss/Dropbox/Projects/LMU/vag_git_repository/VAG_Paper_1/Result/Filelist_Healthy_All_Result_FrequencyDomainFeatures_EXTENSION.csv')
#    print vag_ouss_merge_two_lists_of_dictionaries_depending_on_key(f1,f2,'filename')
    print "hello"
    filesList = oussHelps.vag_ouss_get_files_list("/home/ouss/Dropbox/Projects/LMU/vag_git_repository/VAG_Paper_1/Filelist/Study01_Filelist_Healthy_All.txt","/home/ouss/Dropbox/Projects/LMU/vag_git_repository/VAG_Paper_1/Data/")
    print "hello"
#    vag_ouss_calculate_and_export_time_domain_features(filesList, oussSegments.vag_ouss_get_signal_xml_segments_EXTENSION, vag_ouss_calculate_time_domain_features_of_averaged_segment, "/home/ouss/Dropbox/Projects/LMU/vag_git_repository/VAG_Paper_1/Result/Study01_Filelist_Healthy_All_Result_TimeDomainFeatures_EXTENSION_template.csv")
#    vag_ouss_calculate_and_export_time_domain_features(filesList, oussSegments.vag_ouss_get_signal_xml_segments_FLEXION, vag_ouss_calculate_time_domain_features_of_averaged_segment, "/home/ouss/Dropbox/Projects/LMU/vag_git_repository/VAG_Paper_1/Result/Study01_Filelist_Healthy_All_Result_TimeDomainFeatures_FLEXION_template.csv")
    vag_ouss_calculate_and_export_frequency_domain_features(filesList,oussSegments.vag_ouss_get_signal_xml_segments_EXTENSION, vag_ouss_calculate_frequency_domain_features_of_averaged_segment, "/home/ouss/Dropbox/Projects/LMU/vag_git_repository/VAG_Paper_1/Result/Study01_Filelist_Healthy_All_Result_FrequencyDomainFeatures_EXTENSION_template.csv")
    print "hello"
    vag_ouss_calculate_and_export_frequency_domain_features(filesList,oussSegments.vag_ouss_get_signal_xml_segments_FLEXION, vag_ouss_calculate_frequency_domain_features_of_averaged_segment, "/home/ouss/Dropbox/Projects/LMU/vag_git_repository/VAG_Paper_1/Result/Study01_Filelist_Healthy_All_Result_FrequencyDomainFeatures_FLEXION_template.csv")
    print "hello"

##    filesList = vag_ouss_get_files_list("/home/ouss/Dropbox/Projects/LMU/vag_git_repository/VAG_Paper_1/Filelist/Filelist_Ouss_1.txt","/home/ouss/Dropbox/Projects/LMU/vag_git_repository/VAG_Paper_1/Data/")
##    vag_ouss_segment_flexion_extension_v1_process_files_list(filesList)
##    vag_ouss_find_all_bad_segments_and_append_to_signal_xml_file("/home/ouss/Dropbox/Projects/LMU/vag_git_repository/VAG_Paper_1/Filelist/Filelist_Ouss_1.txt","/home/ouss/Dropbox/Projects/LMU/vag_git_repository/VAG_Paper_1/Data/")
    print "hello"
    filesList = oussHelps.vag_ouss_get_files_list("/home/ouss/Dropbox/Projects/LMU/vag_git_repository/VAG_Paper_1/Filelist/Study01_Filelist_Patients_All.txt","/home/ouss/Dropbox/Projects/LMU/vag_git_repository/VAG_Paper_1/Data/")
    print "hello"
#    vag_ouss_calculate_and_export_time_domain_features(filesList, oussSegments.vag_ouss_get_signal_xml_segments_EXTENSION, vag_ouss_calculate_time_domain_features_of_averaged_segment, "/home/ouss/Dropbox/Projects/LMU/vag_git_repository/VAG_Paper_1/Result/Study01_Filelist_Patients_All_Result_TimeDomainFeatures_EXTENSION_template.csv")
#    vag_ouss_calculate_and_export_time_domain_features(filesList, oussSegments.vag_ouss_get_signal_xml_segments_FLEXION, vag_ouss_calculate_time_domain_features_of_averaged_segment, "/home/ouss/Dropbox/Projects/LMU/vag_git_repository/VAG_Paper_1/Result/Study01_Filelist_Patients_All_Result_TimeDomainFeatures_FLEXION_template.csv")
    print "hello"
    vag_ouss_calculate_and_export_frequency_domain_features(filesList,oussSegments.vag_ouss_get_signal_xml_segments_EXTENSION, vag_ouss_calculate_frequency_domain_features_of_averaged_segment, "/home/ouss/Dropbox/Projects/LMU/vag_git_repository/VAG_Paper_1/Result/Study01_Filelist_Patients_All_Result_FrequencyDomainFeatures_EXTENSION_template.csv")
    print "hello"
    vag_ouss_calculate_and_export_frequency_domain_features(filesList,oussSegments.vag_ouss_get_signal_xml_segments_FLEXION, vag_ouss_calculate_frequency_domain_features_of_averaged_segment, "/home/ouss/Dropbox/Projects/LMU/vag_git_repository/VAG_Paper_1/Result/Study01_Filelist_Patients_All_Result_FrequencyDomainFeatures_FLEXION_template.csv")
    print "hello"
    print filesList

    sys.exit()
####    filesList = vag_ouss_get_files_list("/home/ouss/Dropbox/Projects/LMU/vag_git_repository/VAG_Paper_1/Filelist/Filelist_Ouss_1.txt","/home/ouss/Dropbox/Projects/LMU/vag_git_repository/VAG_Paper_1/Data/")
####    for afile in filesList:
####        print afile
####        vag_ouss_fft_significance_resampling(afile,"EXTENSION")


#    signalsListFileDialogFilename, signalsListFileDialogFilter = QFileDialog.getOpenFileName(caption ="Select *.txt file", filter = "*.txt")
#    signalsListFileRealname, signalsListFileExtension = OsPath.splitext(signalsListFileDialogFilename)
#    filesList = vag_ouss_get_files_list(signalsListFileDialogFilename,prefix)
#    for afile in filesLists:
#        fig = vag_ouss_plot_signal(afile)
#        signalXmlFileDialogFilename = vag_ouss_get_signal_xml_filename(afile)
#        segments = vag_ouss_get_signal_xml_segments_EXTENSION(signalXmlFileDialogFilename )
#        vag_ouss_plot_segments(fig, segments)
#        segments = vag_ouss_get_signal_xml_segments_FLEXION(signalXmlFileDialogFilename )
#        vag_ouss_plot_segments(fig, segments,'green')
#    vag_ouss_segment_flexion_extension_v1_process_files_list(filesList)
#   fout = open("/home/ouss/Dropbox/Projects/LMU/vag/VAG_Paper_1/Filelist/Filelist_All_no_segmentation.log", 'w')
#    for afile in filesList:
#        signalXmlFileDialogFilename = vag_ouss_get_signal_xml_filename(afile)
#        hasSegments = vag_ouss_signal_xml_has_segmentation(signalXmlFileDialogFilename)
#        if not hasSegments:
#            fout.write(afile+"\n")
#    fout.close()
    sys.exit()
    #signalWavFileDialogFilename, signalWavDialogFilter = QFileDialog.getOpenFileName(caption ="Select *.wav file", filter = "*.wav")
    #signalXmlFileDialogFilename = vag_ouss_get_signal_xml_filename(signalWavFileDialogFilename)
    #(vibrationRawData, angularData) = vag_ouss_get_signal_raw_data(signalWavFileDialogFilename)
    #segments = vag_ouss_get_signal_xml_segments(signalXmlFileDialogFilename,"EXTENSION")
    #vibrationRawData = vibrationRawData.tolist()
    ##concatinatedSignal = vag_ouss_concatinate_segments(vibrationRawData, segments)
    ##plt.plot(concatinatedSignal)
    #signals = vag_ouss_arrange_segments_signals_in_a_list(vibrationRawData, segments)
    #signals = vag_ouss_padd_to_same_length(signals)
    #FFTs = vag_ouss_fft_of_signals(signals)
    #f0 = plt.figure(0)
    #for signal in signals:
    #    plt.plot(signal)
    #plt.show()
    #f1 = plt.figure(1)
    #for FFT in FFTs:
    #   vag_ouss_plot_fft_log(f1, FFT[0], FFT[1], 100)
    #f2 = plt.figure(2)
    #significance = vag_ouss_sum_of_fft_of_signals(FFTs)
    #vag_ouss_plot_fft_log(f2, FFTs[0][0], significance, 100)
#    signalWavFileDialogFilename, signalWavDialogFilter = QFileDialog.getOpenFileName(caption ="Select *.wav file", filter = "*.wav")
#    signalXmlFileDialogFilename = vag_ouss_get_signal_xml_filename(signalWavFileDialogFilename)
#    (vibrationRawData, angularData) = vag_ouss_get_signal_raw_data(signalWavFileDialogFilename)
#    segmentsFlexion = vag_ouss_get_signal_xml_segments_FLEXION(signalXmlFileDialogFilename)

#    normalizedSignal = vag_ouss_normalize_raw_signal(vibrationRawData)
#    normalizedAngularData= vag_ouss_normalize_raw_signal(angularData)
#    figure, (ax1, ax2) = plt.subplots(2, sharex=True)
#    figure.set_title(OsPath.basename(signalWavFileDialogFilename))
#    time = numpy.linspace(0,numpy.divide(float(len(normalizedSignal)),16000.0),len(normalizedSignal))
#    ax1.plot(time, normalizedSignal, 'k')
#    ax1.set_title('Normalized vibration sensor signal')
#    ax1.grid(True)
#    ax2.plot(time, normalizedAngularData, 'k')
#    ax2.set_title('Normalized potentiometer signal')
#    ax2.grid(True)
#    ax1.set_ylim([-1.1,1.1])
#    xlim = (ax1.dataLim.bounds[0], ax1.dataLim.bounds[2])
#    ax1.set_xlim(xlim)
#    ax2.set_xlim(xlim)
#    ax2.set_xlabel('Time (s)')
#    plt.show()
#    vag_ouss_plot_segments(figure,segmentsExtension,"red",0.5,16000.0)
#    vag_ouss_plot_segments(figure,segmentsFlexion,"green",0.5,16000.0)
#    figure.canvas.draw()
#    figure.canvas.flush_events()

# plt.savefig("segmentation_plot.pdf", dpi=300)
    #go over files list
   # filesList = vag_ouss_get_files_list(signalsListFileDialogFilename)
    #for each file
   # vag_ouss_process_a_list_of_items(filesList, vag_ouss_fft_significance_FLEXION)
    #vag_ouss_process_a_list_of_items(filesList, vag_ouss_variances_of_segments_EXTENSION)
    #vag_ouss_process_a_list_of_items(filesList, vag_ouss_variances_of_segments_FLEXION)

#    signalsListFileDialogFilename, signalsListFileDialogFilter = QFileDialog.getOpenFileName(caption ="Select *.txt file", filter = "*.txt")
#    signalsListFileDialogFilename = "/home/ouss/Dropbox/Projects/LMU/Skyfall/Data/BMT_Fileslist_All.txt"

#    vag_ouss_segment_flexion_extension_v1(signalsListFileDialogFilename)
#    signalsListFileRealname, signalsListFileExtension = OsPath.splitext(signalsListFileDialogFilename)
#    filesList = vag_ouss_get_files_list(signalsListFileDialogFilename)

#    outputFileDialogFilename = signalsListFileRealname + ".ONN.csv"
#    vag_ouss_skyfall_of_list(filesList, outputFileDialogFilename, vag_ouss_get_signal_xml_segments_ONN)
    #outputFileDialogFilename = signalsListFileRealname + ".FLEXION.csv"
    #vag_ouss_skyfall_of_list(filesList, outputFileDialogFilename, vag_ouss_get_signal_xml_segments_FLEXION)
    #outputFileDialogFilename = signalsListFileRealname + ".EXTENSION.csv"
    #vag_ouss_skyfall_of_list(filesList, outputFileDialogFilename, vag_ouss_get_signal_xml_segments_EXTENSION)

#if __name__ == '__main__':
#    main()
