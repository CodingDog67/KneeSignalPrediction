# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


from workstationfile import return_data_locs
from preprocess import preprocessing

from pathlib import Path

def main():
    # 1 for großhadern, 2 for home, 3 for laptop, 4 for server
    filePath, labelPath, savePath = return_data_locs(2)

    # preprocess data and labels, splitting into segments and saving, or simple loading from split location
    preprocessing(filePath, labelPath, savePath)

if __name__ == '__main__':
    main()