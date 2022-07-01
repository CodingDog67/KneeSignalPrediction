#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
-------------------------------
Vibroarthrography (VAG) Project
-------------------------------
Semi-automatic Segmentation GUI - SPECIAL EDITION 143
    - Import the WAV file
    - Import the associated XML file
    - Automatic segmentation using FEN's algorithm
    - Select/deselect the beginning and the end of each segment by using left/right click
    - Export the indices of each segment to 3 XML files (Patella, TibMed, TibLat)
    - Export the segments to WAV files
To solve Tkinter's issue (unrecognized selector): Canopy > Preferences > Python > Pylab backend: Inline SVG 
Version: 1.143
Author: Tuan Nam Le
Last modified: 27/08/2014
"""

import os
from tkinter import Tk, W, N, E, S, Toplevel
from tkinter import messagebox
from tkinter.filedialog import askopenfilename, askdirectory
from tkinter.ttk import Button, Frame, Label
from xml.etree.ElementTree import Element, SubElement, ElementTree, parse

import matplotlib.backends.backend_tkagg as tkagg
import numpy
from matplotlib import pylab
from matplotlib.widgets import Cursor
from scipy.io import wavfile

import segmentation
import vaghelpers


def OnClick(event):
    global SEGINDEX
    global SEGLINES
    global SEGLINESID
    global SEGSPANS
    # Left click to add the beginning and the end of the segments
    if event.button == 1 and toolbar.mode == "" and event.xdata <= len(vagsamples):
        if not SEGINDEX:
            # If there is nothing
            SEGINDEX.append(int(event.xdata))
            newline = ax.axvline(int(event.xdata), color='k', linewidth=1, picker=5, label='seg')
            SEGLINES.append(newline)
            SEGLINESID.append(id(newline))
            print(u"Begin of segment is selected" + ", INDEX=" + str(SEGINDEX[-1]))
        elif len(SEGINDEX) > 0 and not len(SEGINDEX) % 2:
            # Beginning of segments
            if CheckSEGINDEXBegin(SEGINDEX[-1], int(event.xdata)):
                SEGINDEX.append(int(event.xdata))
                newline = ax.axvline(int(event.xdata), color='k', linewidth=1, picker=5, label='seg')
                SEGLINES.append(newline)
                SEGLINESID.append(id(newline))
                print(u"Begin of segment selected" + ", INDEX=" + str(SEGINDEX[-1]))
            else:
                print(u"Error")
        elif len(SEGINDEX) % 2:
            # End of segments
            if CheckSEGINDEXEnd(SEGINDEX[-1], int(event.xdata)):
                SEGINDEX.append(int(event.xdata))
                newline = ax.axvline(int(event.xdata), color='k', linewidth=2, picker=5, label='seg')
                SEGLINES.append(newline)
                SEGLINESID.append(id(newline))
                newarea = ax.axvspan(SEGINDEX[-2], SEGINDEX[-1], facecolor='r', alpha=0.1, label='segspan')
                SEGSPANS.append(newarea)
                print(u"End of segment selected" + ", INDEX=" + str(SEGINDEX[-1]))
            else:
                print(u"Error")
        canvas.draw()
        # print SEGINDEX


def OnPick(event):
    # Right click on the beginning or the end of the segments to remove it
    if event.mouseevent.button == 3 and toolbar.mode == "":
        xdata, ydata = event.artist.get_data()
        ind = event.ind
        if hasattr(event.artist, 'get_label') and event.artist.get_label() == 'seg':
            idx = SEGLINESID.index(id(event.artist))
            if idx % 2:
                # If idx odd -> delete the end of segment first
                SEGINDEX.remove(xdata[ind])
                SEGLINES.pop(idx).remove()
                SEGLINESID.pop(idx)
                # Then delete the beginning of segment
                SEGINDEX.pop(idx - 1)
                SEGLINES.pop(idx - 1).remove()
                SEGLINESID.pop(idx - 1)
                # Then delete the segment span
                SEGSPANS.pop((idx - 1) / 2).remove()
            else:
                # If idx even -> the delete the beginning of segment first
                SEGINDEX.remove(xdata[ind])
                SEGLINES.pop(idx).remove()
                SEGLINESID.pop(idx)
                # Then delete the end of segment, index changes!!!
                SEGINDEX.pop(idx)
                SEGLINES.pop(idx).remove()
                SEGLINESID.pop(idx)
                # Then delete the segment span
                SEGSPANS.pop(idx / 2).remove()
            print(u"Segment removed")
            canvas.draw()
            # print SEGINDEX


def AutoSegmentation():
    global SEGINDEX
    global SEGLINES
    global SEGLINESID
    global SEGSPANS

    clear()

    SEGINDEX = segmentation.segmentation_jhu(fs, angles)

    for idx in range(0, len(SEGINDEX)):
        # If idx even
        if not idx % 2:
            newline = ax.axvline(SEGINDEX[idx], color='k', linewidth=1, picker=5, label='seg')
            SEGLINES.append(newline)
            SEGLINESID.append(id(newline))
        # If idx odd
        else:
            newline = ax.axvline(SEGINDEX[idx], color='k', linewidth=2, picker=5, label='seg')
            SEGLINES.append(newline)
            SEGLINESID.append(id(newline))
            newarea = ax.axvspan(SEGINDEX[idx - 1], SEGINDEX[idx], facecolor='r', alpha=0.1, label='segspan')
            SEGSPANS.append(newarea)
        canvas.draw()

    print(u"AutoSegmentation ... Finish!")


def ImportFromXML():
    # Check if associated XML file exists
    if os.path.isfile(input_xmlfile):
        # Parse the XML file into XML element tree
        xml_Tree = parse(input_xmlfile)
        xml_Root = xml_Tree.getroot()
        # Find element node 'segmentation'
        # If there are some segments, plot them
        xml_Segmentation = xml_Root.find('segmentation')
        if xml_Segmentation:
            for segment in xml_Segmentation.findall('segment'):
                begin = segment.find('begin').text
                SEGINDEX.append(int(begin))
                end = segment.find('end').text
                SEGINDEX.append(int(end))
            # print SEGINDEX
            XSEGMENTS = numpy.reshape(SEGINDEX, (-1, 2))
            for i in range(0, len(XSEGMENTS)):
                newlinebegin = ax.axvline(XSEGMENTS[i][0], color='k', linewidth=1, picker=5, label='seg')
                SEGLINES.append(newlinebegin)
                SEGLINESID.append(id(newlinebegin))
                newlineend = ax.axvline(XSEGMENTS[i][1], color='k', linewidth=2, picker=5, label='seg')
                SEGLINES.append(newlineend)
                SEGLINESID.append(id(newlineend))
                newarea = ax.axvspan(XSEGMENTS[i][0], XSEGMENTS[i][1], facecolor='r', alpha=0.1, label='segspan')
                SEGSPANS.append(newarea)
            canvas.draw()
            print(u"ImportFromXML ... Finish!")
        else:
            messagebox.showerror(title="Error", message="Segment(s) not found")
    else:
        messagebox.showerror(title="Error", message="XML file not found")


def ExportToXML():
    global SEGINDEX
    XSEGMENTS = numpy.sort(SEGINDEX)
    if len(XSEGMENTS) % 2:
        messagebox.showerror(title="Error", message="Uncompleted segment! Please try again.")
    else:
        XSEGMENTS = numpy.reshape(XSEGMENTS, (-1, 2))
        # input_xmlfile = os.path.join(os.path.normcase(dirname), realname)+".xml"
        # output_xmlfile = os.path.join(os.path.normcase(dirname), realname)+".xml"
        for elem in input_xmllist:
            # Check if associated XML file exists
            if os.path.isfile(input_xmlfile):
                print(u"XML file exists")
                # Parse the XML file into XML element tree
                xml_Tree = parse(input_xmlfile)
                xml_Root = xml_Tree.getroot()
            else:
                print(u"XML file does not exist")
                create_xml = messagebox.askokcancel(title="XML file does not exist", message="Create XML file?")
                if create_xml:
                    # Build a new XML element tree
                    xml_Root = Element('vagdata')
                    xml_Root.attrib = {'version': str(2.0)}
                    # Add element node 'signal'
                    xml_Signal = SubElement(xml_Root, 'signal')
                    xml_Signal.text = vagname

            # Add element node 'segmentation'
            xml_Segmentation = xml_Root.find('segmentation')
            # If the element node not found, create new one; otherwise remove and then create new
            if xml_Segmentation is None:
                xml_Segmentation = SubElement(xml_Root, 'segmentation')
                xml_Segmentation.attrib = {'version': str('ONN')}
            else:
                xml_Root.remove(xml_Segmentation)
                xml_Segmentation = SubElement(xml_Root, 'segmentation')
                xml_Segmentation.attrib = {'version': str('ONN')}
            xml_SegmentsCount = SubElement(xml_Segmentation, 'segmentscount')
            xml_SegmentsCount.text = str(len(XSEGMENTS))
            for i in range(0, len(XSEGMENTS)):
                xml_Segment = SubElement(xml_Segmentation, 'segment')
                xml_Segment.attrib = {'index': str(i + 1)}
                xml_Begin = SubElement(xml_Segment, 'begin')
                xml_Begin.text = str(XSEGMENTS[i][0])
                xml_End = SubElement(xml_Segment, 'end')
                xml_End.text = str(XSEGMENTS[i][1])
                xml_Type = SubElement(xml_Segment, 'type')
                xml_Type.text = "Measurement/Calibration"
            ElementTree(xml_Root).write(elem)
        print(u"ExportToXML ... Finish!")


def ExportToFiles():
    global SEGINDEX
    XSEGMENTS = numpy.sort(SEGINDEX)
    XSEGMENTS = numpy.reshape(XSEGMENTS, (-1, 2))
    output_filepath = askdirectory(parent=root)
    output_filepath = os.path.join(os.path.normcase(output_filepath), realname)
    for i in range(0, len(XSEGMENTS)):
        wavfile.write(output_filepath + "_segment_" + str(i + 1) + ".wav", fs, samples[XSEGMENTS[i][0]:XSEGMENTS[i][1]])
    print(u"ExportToFiles ... Finish!")


def CheckSEGINDEXBegin(idx_end, idx_begin):
    if idx_end < idx_begin:
        return True
    else:
        return False


def CheckSEGINDEXEnd(idx_begin, idx_end):
    if idx_end > idx_begin:
        return True
    else:
        return False


def clear():
    global SEGINDEX
    global SEGLINES
    global SEGLINESID
    global SEGSPANS

    while len(SEGINDEX) > 0: SEGINDEX.pop()
    while len(SEGLINES) > 0: SEGLINES.pop().remove()
    while len(SEGLINESID) > 0: SEGLINESID.pop()
    while len(SEGSPANS) > 0: SEGSPANS.pop().remove()
    canvas.draw()


def quit():
    root.quit()
    root.destroy()
    canvas.mpl_disconnect(cidpress)
    canvas.mpl_disconnect(cidpick)


def help():
    help_win = Toplevel(bg='w')
    help_win.title("Help")
    help_win.geometry('%dx%d+%d+%d' % (w, h, x, y))
    help_text = Label(help_win, text="Welcome to Semi-automatic Segmentation GUI")
    help_text.pack()

    Button(help_win, text='OK', command=help_win.destroy).pack()


if __name__ == '__main__':

    root = Tk()
    w = 800
    h = 600
    ws = root.winfo_screenwidth()
    hs = root.winfo_screenheight()
    x = (ws / 2) - (w / 2)
    y = (hs / 2) - (h / 2)
    root.geometry('%dx%d+%d+%d' % (w, h, x, y))
    root.wm_title("Semi-automatic Segmentation GUI 143")
    root.columnconfigure(1, weight=1)
    root.columnconfigure(4, pad=10)
    root.rowconfigure(0, pad=10)
    root.rowconfigure(1, pad=10)
    root.rowconfigure(2, pad=10)
    root.rowconfigure(3, pad=10)
    root.rowconfigure(4, pad=10)
    root.rowconfigure(5, pad=10)
    root.rowconfigure(6, weight=1)
    root.rowconfigure(8, pad=10)

    # Figure
    pylab.close('all')
    fig = pylab.figure()
    ax = fig.add_subplot(111)
    ax.grid(True)
    canvas = tkagg.FigureCanvasTkAgg(fig, master=root)
    canvas.get_tk_widget().grid(row=0, column=0, columnspan=3, rowspan=7, sticky=E + W + S + N)
    canvas.draw()

    # Toolbar
    toolbar_frame = Frame(root)
    toolbar_frame.grid(row=7, column=0, padx=5)
    toolbar = tkagg.NavigationToolbar2Tk(canvas, toolbar_frame)

    # Cursor
    cursor = Cursor(ax, useblit=True, color='blue', linewidth=0.5)
    cidpress = fig.canvas.mpl_connect('button_press_event', OnClick)
    cidpick = fig.canvas.mpl_connect('pick_event', OnPick)

    # Auto Segmentation
    button_autoseg = Button(root, text="Automatic segmentation", command=AutoSegmentation)
    button_autoseg.grid(row=0, column=4, padx=5, pady=5, sticky=E + W + S + N)

    # Import from XML
    button_importxml = Button(root, text="Import from XML", command=ImportFromXML)
    button_importxml.grid(row=1, column=4, padx=5, pady=5, sticky=E + W + S + N)

    # Export to XML
    button_exportxml = Button(root, text="Export to XML", command=ExportToXML)
    button_exportxml.grid(row=2, column=4, padx=5, pady=5, sticky=E + W + S + N)

    # Export to Files
    button_exportfiles = Button(root, text="Export to files", command=ExportToFiles)
    button_exportfiles.grid(row=3, column=4, padx=5, pady=5, sticky=E + W + S + N)

    # Clear
    button_clear = Button(root, text="Clear", command=clear)
    button_clear.grid(row=4, column=4, padx=5, pady=5, sticky=E + W + S + N)

    # Quit
    button_quit = Button(root, text="Quit", command=quit)
    button_quit.grid(row=7, column=4, padx=5, pady=5, sticky=E + W + S + N)

    SEGINDEX = list()
    SEGLINES = list()
    SEGLINESID = list()
    SEGSPANS = list()

    # Open the file and plot the VAG signal
    input_vagfile = askopenfilename(parent=root, defaultextension='.wav', title="Select VAG file")
    vagname = os.path.basename(input_vagfile)
    dirname = os.path.dirname(input_vagfile)
    (realname, extension) = os.path.splitext(vagname)
    input_xmlfile = os.path.join(os.path.normcase(dirname), realname) + ".xml"

    # Search for another XML files (Patella, TibiaplateauMedial, TibiaplateauLateral)
    input_xmllist = [input_xmlfile]
    smatch = vagname[:51]
    for filename in os.listdir(dirname):
        if filename.endswith(".xml") & filename.startswith(smatch):
            input_xmllist.append(os.path.join(os.path.normcase(dirname), filename))
    input_xmllist = numpy.unique(input_xmllist)
    print('\n'.join(input_xmllist))
<<<<<<< Updated upstream

=======
    
>>>>>>> Stashed changes
    try:
        # Import VAG file
        (fs, samples) = wavfile.read(input_vagfile)
        # scipy.io.wavread does not support 32-bit float files
        vagsamples = vaghelpers.vag2float(samples, numpy.float32)
        # Separate signal and angular values for segmentation
        signal = vagsamples[:, 0]
        angles = vagsamples[:, 1]

        ax.plot(vagsamples, label='vag')
        canvas.draw()
        root.resizable(True, False)
        root.mainloop()
    except:
        root.quit()
        root.destroy()
