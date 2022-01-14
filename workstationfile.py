# 1 f2lab
# 2 home
# 3 laptop
# 4 server

filePath = ''
labelPath = ''
def return_data_locs(location):

    # data locations f2lab
    match location:
        case 1:
            filePath = '/media/ari/Harddisk/knee data/knee signal/Patienten/'
            labelPath = '/media/ari/Harddisk/knee data/knee signal/'

    # data locations home

    # data locations laptop

    # data locations server

    return filePath, labelPath