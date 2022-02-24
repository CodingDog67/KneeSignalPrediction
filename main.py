# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

# todo read The new descriptor in processing of vibroacoustic signal of knee joint
# https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7506694/#B14-sensors-20-05015

from workstationfile import return_data_locs
from preprocess import *
from helpers import *
import librosa

preprocess_data = False
filter_data = True
sorting = False

# 1 for großhadern, 2 for home, 3 for laptop, 4 for server
paths = return_data_locs(1)


def prepare_data():

    # preprocess data and labels, splitting into segments and saving, or simple loading from split location save labels
    # in the same way
    if preprocess_data:
        preprocessing(paths['filePath'], paths['savePath'])

    # read in data into datastructure
    if sorting:
        sortdata(paths['savePath'])


def main():

    # read in labels and data
    retropatellar, lateral, medial, innenmeniskus, aussenmeniskus, session = \
        read_save_labels(paths['labelPath'])

    prepare_data()

    # read in names of data + create a label list based on that ( labels need extending due to segmentation of segments)
    name_list_patella, name_list_medial, name_list_lateral = read_data_names_in(paths)

    label_patella = extend_labels(retropatellar, name_list_patella, session)
    label_medial = extend_labels(medial, name_list_medial, session)
    label_lateral = extend_labels(lateral, name_list_lateral, session)


    # filter data to get rid of outliers
    if filter_data:
        data = 0  # filter this
        file_data_patella, samplerate_data = read_final_data(paths['patella_data'])

        for counter, single_file in enumerate(file_data_patella):
            alpha = 0.95
            single_smooth_file = exponential_smoothing(file_data_patella[0], alpha=alpha)
            plot_simple_data(single_file, single_smooth_file, samplerate_data[counter],
                             name_list_patella[counter], alpha, paths['patella_data_smooth'])

        #test filter
        smooth_data = exponential_smoothing(file_data_patella[0], alpha=0.95)
        plot_simple_data(file_data_patella[0], smooth_data,  samplerate_data[0], save=True)

        #spectogram
        f, t, Sxx = signal.spectrogram(x=smooth_data, fs= samplerate_data[0], nperseg=5)
        plt.pcolormesh(t, f, Sxx, shading='gouraud')
        plt.ylabel('Frequency [Hz]')
        plt.xlabel('Time [sec]')
        plt.show()

        #stft
        amp = 0.25
        f, t, Zxx = signal.stft(smooth_data, samplerate_data[0], nperseg=5)
        plt.pcolormesh(t, f, np.abs(Zxx), vmin=0, vmax=amp, shading='gouraud')
        plt.title('STFT Magnitude')
        plt.ylabel('Frequency [Hz]')
        plt.xlabel('Time [sec]')
        plt.show()

        #mel_spec
        spec = librosa.feature.melspectrogram( y=smooth_data, sr=samplerate_data[0], S=None, n_fft=2048, hop_length=50, win_length=None,
                                       window='hann', center=True, pad_mode='constant', power=2.0)


if __name__ == '__main__':
    main()



######################### graveyard

    # quick check for sessions in file_list und official session list
    # elements = [element[0:11] for element in file_list_patella]
    # myset = list(set(elements))
    # myset.sort()
    # diff = list(set(session) - set(myset))
    # diff.sort()