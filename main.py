# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

#todo watch this https://www.youtube.com/watch?v=PPmNYwVbcts&list=PLmZlBIcArwhN8nFJ8VL1jLM2Qe7YCcmAb&index=3

from workstationfile import return_data_locs
from preprocess import *
from helpers import *

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

    #read in labels and data
    retropatellar, lateral, medial, innenmeniskus, aussenmeniskus, session = \
        read_save_labels(paths['labelPath'])

    prepare_data()

    file_list_patella = os.listdir(paths["patella_data"])
    approved = ["wav"]
    # filter out all the wav files and sort by sensor (2 runs/2 files)
    file_list_patella[:] = [w for w in file_list_patella if any(sub in w for sub in approved)].sort()
    extend_labels(retropatellar, file_list_patella, session)



    label_patella = retropatellar
    # filter data to get rid of outliers
    if filter_data:
        data = 0  # filter this



if __name__ == '__main__':
    main()
