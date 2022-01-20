#!/usr/bin/env python

#Filename manipulations
import os
#Parsing and Manipulating xml files
import xml.etree.ElementTree as XMLETreeElementTree

import numpy

import vag_ouss_helpers as oussHelps
import vag_ouss_helpers_plotting as oussPlots
import vag_ouss_helpers_signals as oussSignals
        
def vag_ouss_signal_xml_has_segmentation(signalXmlFileDialogFilename, segmentationVersion="ONN"):
    if not os.path.isfile(signalXmlFileDialogFilename):
        print "can't find: "+os.path.basename(signalXmlFileDialogFilename)+"\n"
        return    
    xmlElementTree = XMLETreeElementTree.parse (signalXmlFileDialogFilename)
    xmlRoot = xmlElementTree.getroot()
    xmlSegmentationNodeFound = False
    for xmlSegmentationNode in xmlRoot.iter('segmentation'):
        if xmlSegmentationNode is not None:
            if xmlSegmentationNode.attrib['version']==segmentationVersion:
                xmlSegmentationNodeFound = True
                #xmlSegmentationNodeRequestedVersion = xmlSegmentationNode
                break
    if not xmlSegmentationNodeFound:
        print "Could not find Segmentation Node of version: " + segmentationVersion + " for: " + os.path.basename(signalXmlFileDialogFilename)
        return False
    return True

def vag_ouss_segment_flexion_extension_v1_process_files_list(filesList):
    return oussHelps.vag_ouss_process_a_list_of_items(filesList, vag_ouss_segment_flexion_extension_v1_process_file)

def vag_ouss_segment_flexion_extension_v1_process_file(signalWavFileDialogFilename, plotPlease = False, overwrite = True, loadIgnoreSegmentsNode = False):
    signalXmlFileDialogFilename = oussHelps.vag_ouss_get_signal_xml_filename(signalWavFileDialogFilename)
    #print signalXmlFileDialogFilename
    #Read the ONN segments from the xml file
    if os.path.isfile(signalXmlFileDialogFilename):
        segments = vag_ouss_get_signal_xml_segments_ONN(signalXmlFileDialogFilename,loadIgnoreSegmentsNode)
        if plotPlease:
            figure = oussPlots.vag_ouss_plot_signal(signalWavFileDialogFilename)
            figure = oussPlots.vag_ouss_plot_segments(figure, segments, 'orange', 0.5)
            figure.canvas.draw()
            figure.canvas.flush_events()
    else:
        print "can't find: " + os.path.basename(signalXmlFileDialogFilename)+"\n"
        return
    #Read the WaveFile
    (signaldata, angularData) = oussSignals.vag_ouss_get_signal_raw_data(signalWavFileDialogFilename)
    if plotPlease:
        figure2 = oussPlots.vag_ouss_plot_signal(signalWavFileDialogFilename)

    extensionSegments = []
    flexionSegments = []
    #Go over each segment and
    for segment in segments:
        #Get the FE indices
        (extensionEndIndex , flexionBeginIndex) = vag_ouss_segment_define_flexion_extension_v1(segment, angularData)
        if plotPlease:
            oussPlots.vag_ouss_plot_segment(figure2, (segment[0], extensionEndIndex), 'red', 0.5)
            oussPlots.vag_ouss_plot_segment(figure2, (flexionBeginIndex, segment[1]), 'green', 0.5)
        extensionSegments.append( (segment[0], extensionEndIndex) )
        flexionSegments.append((flexionBeginIndex, segment[1]))

    #Export the extension segmentation to the xml file
    #print "extension"
    #print
    vag_ouss_append_segmentation_to_signal_xml_file(signalXmlFileDialogFilename,'EXTENSION',extensionSegments, overwrite)
    #Export the flexion segmentation to the xml file
    #print "flexion"
    vag_ouss_append_segmentation_to_signal_xml_file(signalXmlFileDialogFilename,'FLEXION',flexionSegments, overwrite)
    return

def vag_ouss_append_segmentation_to_signal_xml_file(signalXmlFileDialogFilename,segmentationVersion, segments, overwrite = True):
    #if not file exists
    if not os.path.isfile(signalXmlFileDialogFilename):
        #exit
        print "can't find: " + os.path.basename(signalXmlFileDialogFilename)+"\n"
        return
    # Parse the XML file into XML element tree
    xmlElementTree = XMLETreeElementTree.parse (signalXmlFileDialogFilename)
    xmlRoot = xmlElementTree.getroot()
    # Find the element node 'segmentation' of the XML element tree with the right version
    #xmlSegmentationNode = xmlRoot.find('segmentation')
    #print "Segmentation Version:"+xmlSegmentationNode.get('version')
    xmlSegmentationNodeFound = False
    for xmlSegmentationNode in xmlRoot.iter('segmentation'):
        if xmlSegmentationNode is not None:
            if xmlSegmentationNode.attrib['version']==segmentationVersion:
                xmlSegmentationNodeFound = True
                xmlSegmentationNodeRequestedVersion = xmlSegmentationNode
                break

    #print "I am here"
    #if segmentation exists
    if xmlSegmentationNodeFound:
        if overwrite:
            #delete segme83ntation
            xmlRoot.remove(xmlSegmentationNodeRequestedVersion)
        else:
            print "Segmentation of version: " + segmentationVersion + " already exists in: " + os.path.basename(signalXmlFileDialogFilename)
            return
    #
    #do some xml magic
    #append Segmentation
    xmlSegmentationNode = XMLETreeElementTree.SubElement(xmlRoot, 'segmentation', {'version':segmentationVersion})
    XMLETreeElementTree.SubElement(xmlSegmentationNode, 'segmentscount').text = str(len(segments))
    counter = 1
    for segment in segments:
        xmlSegmentNode = XMLETreeElementTree.SubElement(xmlSegmentationNode, 'segment', {'index':str(counter)})
        XMLETreeElementTree.SubElement(xmlSegmentNode, 'index').text = str(counter)
        XMLETreeElementTree.SubElement(xmlSegmentNode, 'begin').text = str(segment[0])
        XMLETreeElementTree.SubElement(xmlSegmentNode, 'end').text = str(segment[1])
        XMLETreeElementTree.SubElement(xmlSegmentNode, 'type').text = "Measurement"
        counter = counter+1
    #save
    xmlElementTree.write(signalXmlFileDialogFilename)
    return

def vag_ouss_segment_define_flexion_extension_v1(segment,angularData):
    segmentBegin=segment[0]
    segmentEnd=segment[1]
#    t_m1 = Find first Local Maxima of Angular Values, that is the point of
    segmentAngularData = angularData[segmentBegin:segmentEnd+1]
    maximumExtension = max(segmentAngularData)
    maximumExtensionIndexInThisSegment= numpy.where(segmentAngularData==maximumExtension)[0][0]
    extensionEndIndex = segmentBegin + maximumExtensionIndexInThisSegment
#    print extensionEndIndex
#    t_m2 = Find last Local Maxima of Angular
    segmentFlexionAngularData = angularData[extensionEndIndex:segmentEnd+1]
#    print segmentFlexionAngularData
    segmentFlexionAngularDataReversed = segmentFlexionAngularData[::-1]
#    print segmentFlexionAngularDataReversed
    maximumAngleBeforeFlexion = max(segmentFlexionAngularDataReversed)
#    print maximumAngleBeforeFlexion
    maximumAngleBeforeFlexionIndexInThisReversedSegment = numpy.where(segmentFlexionAngularDataReversed==maximumAngleBeforeFlexion)[0][0]
#    print maximumAngleBeforeFlexionIndexInThisReversedSegment
#    print len(segmentFlexionAngularDataReversed) - 1
    maximumAngleBeforeFlexionIndexInThisSegment = len(segmentFlexionAngularDataReversed) - 1 - maximumAngleBeforeFlexionIndexInThisReversedSegment
#    print maximumAngleBeforeFlexionIndexInThisSegment
    flexionBeginIndex = extensionEndIndex + maximumAngleBeforeFlexionIndexInThisSegment
    #Add information to XML file
    return (extensionEndIndex , flexionBeginIndex)


    
def vag_ouss_append_ignored_segments_to_signal_xml_file(signalXmlFileDialogFilename,ignoredSegments, overwrite = True):
     #if not file exists
    if not os.path.isfile(signalXmlFileDialogFilename):
        #exit
        print "can't find: " + os.path.basename(signalXmlFileDialogFilename)+"\n"
        return
    # Parse the XML file into XML element tree
    xmlElementTree = XMLETreeElementTree.parse (signalXmlFileDialogFilename)
    xmlRoot = xmlElementTree.getroot()
    # Find the element node 'segmentation' of the XML element tree with the right version   
    
    
    #ignore segments
    #Read the segments to be ignored from the xml file
    xmlIgnoreNodeFound = False
    for xmlIgnoreNode in xmlRoot.iter('ignore_segments'):
        if xmlIgnoreNode is not None:
            xmlIgnoreNodeFound = True
            xmlIgnoreNodeRequestedVersion = xmlIgnoreNode
            break
                
    #if segmentation exists
    if xmlIgnoreNodeFound:
        if overwrite:
            #delete segmentation
            xmlRoot.remove(xmlIgnoreNodeRequestedVersion)
        else:
            print "Ignore node already exists in: ", os.path.basename(signalXmlFileDialogFilename), " and overwrite is set to False."
            return   
       #do some xml magic
    #append Segmentation
    xmlIgnoreNode = XMLETreeElementTree.SubElement(xmlRoot, 'ignore_segments')
    for index in ignoredSegments:
        xmlSegmentNode = XMLETreeElementTree.SubElement(xmlIgnoreNode, 'segment')
        XMLETreeElementTree.SubElement(xmlSegmentNode, 'index').text = str(index+1)
    #save
    xmlElementTree.write(signalXmlFileDialogFilename)
 #   print "Check: " + signalXmlFileDialogFilename
    return

def vag_ouss_segment_flexion_extension_v1(filesListFileDialogFilename,prefix=""):
    #vag_ouss_segment_flexion_extension_v1_process_file(signalWavFileDialogFilename, true)
    filesList = oussSignals.vag_ouss_get_files_list(filesListFileDialogFilename,prefix)
    vag_ouss_segment_flexion_extension_v1_process_files_list(filesList)
    return
    
def vag_ouss_find_short_segments(signals,duration=1.0):
    indices = []
    index = 0
    for signal in signals:
        T = float(len(signal))/16000.0
        if T < duration:
            indices.append(index)
        index = index+1
    return indices

def vag_ouss_find_saturated_segments(signals,length=16):
    indices = []
    index = 0
    for signal in signals:
        if numpy.max(numpy.absolute(signal)) >= 2**(length-1)-1:
            indices.append(index)
        index = index+1
    return indices

def vag_ouss_find_noisy_segments_concatinated(signals,ratio=2.0):
    concatinatedSignal = []
    for signal in signals:
        concatinatedSignal = concatinatedSignal + signal
    concatinatedVariance = numpy.var(concatinatedSignal)
    indices = []
    index = 0
    for signal in signals:
        variance = numpy.var(signal)
        varianceRatio = variance/concatinatedVariance
        if varianceRatio>=ratio:
            indices.append(index)
        index = index +1
    return indices
    
def vag_ouss_find_noisy_segments(signals,ratio=2.0):    
    indices = []
    variances = []
    for signal in signals:
        #print signal
        #calculate variances
        variance = numpy.var(signal)
        variances.append(variance)
        
    variancesMean = numpy.mean(variances)
    #print variancesMean 
#    ms = [variancesMean for i in range(len(signals))]    
#    plt.figure()
#    plt.plot(variances)
#    plt.plot(ms)
#    plt.show    
    index = 0
    for variance in variances:
        #print i
        #print variancesMean 
        #print variancesegment
        varianceRatio = variance/variancesMean
#        print ratio 
        if varianceRatio>=ratio:
            indices.append(index)
#            print "the index to ignore is:", index
        index=index+1
    #Check if there is something outlendish
    #Var
    return indices

def vag_ouss_find_shortest_segment(segments):
    index = 0
    minWidth = segments[0][1]-segments[0][0]
    i = 0
    for segment in segments:
        segmentWidth = segment[1]-segment[0]
        if segmentWidth < minWidth:
            minWidth = segmentWidth
            index = i
        i=i+1 
    return index
    
def vag_ouss_get_shortest_segment_samples_count(segments):
    minWidth = segments[0][1]-segments[0][0]
    for segment in segments:
        segmentWidth = segment[1]-segment[0]
        #print segmentWidth
        if segmentWidth < minWidth:
            minWidth = segmentWidth
    return minWidth
    
def vag_ouss_get_shortest_segment_duration(segments,fs):
    return float(vag_ouss_get_shortest_segment_samples_count(segments))/float(fs)
    
def vag_ouss_get_signal_xml_segments_ONN(signalXmlFileDialogFilename,loadIgnoreSegmentsNode=True):
    return vag_ouss_get_signal_xml_segments(signalXmlFileDialogFilename, "ONN",loadIgnoreSegmentsNode)

#http://en.wikipedia.org/wiki/Anatomical_terms_of_motion
#http://en.wikipedia.org/wiki/Anatomical_terms_of_motion#mediaviewer/File:Body_Movements_I.jpg
def vag_ouss_get_signal_xml_segments_EXTENSION(signalXmlFileDialogFilename,loadIgnoreSegmentsNode=True):
        return vag_ouss_get_signal_xml_segments(signalXmlFileDialogFilename, "EXTENSION",loadIgnoreSegmentsNode)

#http://en.wikipedia.org/wiki/Anatomical_terms_of_motion
#http://en.wikipedia.org/wiki/Anatomical_terms_of_motion#mediaviewer/File:Body_Movements_I.jpg
def vag_ouss_get_signal_xml_segments_FLEXION(signalXmlFileDialogFilename,loadIgnoreSegmentsNode=True):
        return vag_ouss_get_signal_xml_segments(signalXmlFileDialogFilename, "FLEXION",loadIgnoreSegmentsNode)

def vag_ouss_get_signal_xml_segments(signalXmlFileDialogFilename, segmentationVersion="ONN",loadIgnoreSegmentsNode=True):
    if not os.path.isfile(signalXmlFileDialogFilename):
        print "can't find: "+os.path.basename(signalXmlFileDialogFilename)+"\n"
        return
    #print segmentationVersion
    #open the xml file and get the indices of the segmentation

    #Prepare an empty list that will contain the SegmentsIndices
    segmentsIndices = []

    # Parse the XML file into XML element tree
    xmlElementTree = XMLETreeElementTree.parse (signalXmlFileDialogFilename)
    xmlRoot = xmlElementTree.getroot()
    # Find the element node 'segmentation' of the XML element tree with the right version
    #xmlSegmentationNode = xmlRoot.find('segmentation')
    #print "Segmentation Version:"+xmlSegmentationNode.get('version')
    xmlSegmentationNodeFound = False
    for xmlSegmentationNode in xmlRoot.iter('segmentation'):
        if xmlSegmentationNode is not None:
            if xmlSegmentationNode.attrib['version']==segmentationVersion:
                xmlSegmentationNodeFound = True
                xmlSegmentationNodeRequestedVersion = xmlSegmentationNode
                break
    if not xmlSegmentationNodeFound:
        print "Could not find Segmentation Node of version: " + segmentationVersion + " for: " + os.path.basename(signalXmlFileDialogFilename)
        return
    #print xmlSegmentationNodeRequestedVersion
    for segment in xmlSegmentationNodeRequestedVersion.findall('segment'):
         segmentBegin = segment.find('begin').text
         segmentEnd = segment.find('end').text
         segmentsIndices.append((int(segmentBegin), int(segmentEnd)))
    
    if not loadIgnoreSegmentsNode:
        return segmentsIndices
    #ignore segments
    #Read the segments to be ignored from the xml file
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
         ignoredSegments.append(int(index)-1)
    
#    print ignoredSegments
        
    #remove the ignoredIndices from segmentsIndices and return the latter
    ignoredSegments.sort()
    ignoredSegments.reverse()
    for index in ignoredSegments:
        del segmentsIndices[index]
        
#    print segmentsIndices
    return segmentsIndices
    
    
def vag_ouss_plot_segmented_signal(signalWavFileDialogFilename, segmentationVersion='ONN', color = 'orange', alpha = 0.5):
    #get signal
    signalXmlFileDialogFilename = oussHelps.vag_ouss_get_signal_xml_filename(signalWavFileDialogFilename)
    #print signalXmlFileDialogFilename
    #Read the ONN segments from the xml file
    if os.path.isfile(signalXmlFileDialogFilename):
        segments = vag_ouss_get_signal_xml_segments(signalXmlFileDialogFilename, segmentationVersion)
        (vibrationData, angularData) = oussSignals.vag_ouss_get_signal_raw_data(signalWavFileDialogFilename,False)
        figure = oussPlots.vag_ouss_plot_signal(vibrationData, angularData)
        figure = oussPlots.vag_ouss_plot_segments(figure, segments, color, alpha)
        figure.canvas.draw()
        figure.canvas.flush_events()
        return figure       
    return