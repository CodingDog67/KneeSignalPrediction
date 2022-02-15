# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


from workstationfile import return_data_locs
from preprocess import *
from helpers import *

from pathlib import Path

preprocess_data = False
filter_data = True
sorting = False


def main():
    # 1 for großhadern, 2 for home, 3 for laptop, 4 for server
    filePath, labelPath, savePath = return_data_locs(2)

    # preprocess data and labels, splitting into segments and saving, or simple loading from split location save labels
    # in the same way
    if preprocess_data:
        preprocessing(filePath, labelPath, savePath)

    # read in data into datastructure
    if sorting:
        sortdata(labelPath, savePath)

    retropatellar, lateral, medial, innenmeniskus, aussenmeniskus, session = \
        read_save_labels(labelPath)

    # filter data to get rid of outliers
    if filter_data:
        data = 0  # filter this


if __name__ == '__main__':
    main()
