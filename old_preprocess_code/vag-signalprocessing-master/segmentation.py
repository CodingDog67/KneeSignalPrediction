import numpy
from matplotlib import mlab
from scipy import signal
import numpy as np
from copy import deepcopy
from scipy.signal import find_peaks



def finding(condition):
    res, = np.nonzero(np.ravel(condition))
    return res  # return tuple might cause a problem

def segmentation_jhu(fs, angles):
    """
    Segmentation algorithm using angular signals from FEN/JHU
    (after clean up the code for better understanding)
    """
    # IIR filter design for low pass
    Fpass = 1
    Fstop = 10
    Wip = float(Fpass) / float(fs / 2)  # passband edge frequency
    Wis = float(Fstop) / float(fs / 2)  # stopband edge frequency
    Rp = 1  # passband ripple
    As = 40  # stopband attenuation
    bc, ac = signal.iirdesign(Wip, Wis, Rp, As, ftype='butter')
    filt_ang = signal.filtfilt(bc, ac, angles)

    # Rescaling angle measurements to 0-90 degrees
    mi = min(filt_ang)
    ma = max(filt_ang)
    rescale_90 = lambda x: 90 * (x - mi) / (ma - mi)
    ang_rescale = numpy.array(rescale_90(filt_ang))

    ang_vel = numpy.gradient(ang_rescale)  # gradient to detect pause in movement

    #peaks = list(peak_finder.maxdet(ang_rescale, 20))
    peaks, _ = find_peaks(ang_rescale, prominence=20)  #new

    # Filtering where values are greater than 53 degrees
    peaks = list(filter(lambda x: ang_rescale[x] > 53, peaks))
    relmin = finding(numpy.round(ang_vel, decimals=4) == 0.000)  # search for all 0 gradient, angles minima (resting pose)  0.0000 normally

    # Filtering where values are less than 53 degrees
    relmin = list(filter(lambda x: ang_rescale[x] < 53, relmin))
    section = deepcopy(peaks)
    section.extend((0, len(ang_rescale) - 1))  # why insert a 0 second to last position ?
    section = numpy.sort(section)  # why sort it now? and not insert in first pos in the first place?
    newrelmin = list()

    # Between peaks, defining thresholds based on standard deviations
    for pk in range(len(section) - 1):
        pos = list(filter(lambda x: x > section[pk] and x < section[pk + 1], relmin))
        mean = numpy.mean(ang_rescale[pos])
        std = numpy.std(ang_rescale[pos])
        tokeep = list(filter(lambda x: ang_rescale[x] <= mean + std/2, pos))
        # filter to kick out outliers
        newrelmin.extend(tokeep)

    # Definitions for search for max and min to the left, right
    combined = sorted(peaks + newrelmin)

    segments = list()
    for pk in peaks:
        pos = combined.index(pk)
        segments.extend((combined[pos - 1], combined[pos + 1]))  #take corners of rel-min

    # # Calibration Phase values for rescaling angular values
    # win = 50000  #experiment with this value?? dont hardcode it?
    # pos_1 = segments[0] + numpy.argmax(ang_rescale[segments[0]:segments[0] + win])
    # pos_2 = segments[1] - win + numpy.argmax(ang_rescale[segments[1] - win:segments[1]])
    #
    # new_mi = numpy.mean(filt_ang[0:segments[0]])
    # new_ma = numpy.mean(filt_ang[pos_1:pos_2])
    #
    # # Rescaling Angular values for storing, segmentation
    # new_angles = numpy.array(map(lambda x: 90 * (x - new_mi) / (new_ma - new_mi), filt_ang))
    #
    # # Find Time derivative (angular velocity)
    # new_angvel = numpy.gradient(new_angles)

    # Redefine segments for segmentation, storage
    new_segs = segments[2:]

    return new_segs
