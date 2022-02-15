# 1 f2lab
# 2 home
# 3 laptop
# 4 server

filePath = ''
labelPath = ''

def return_data_locs(location):

    # data locations f2lab
    if location == 1:
        filePath = '/media/ari/Harddisk/knee data/knee signal/Patienten/'
        labelPath = '/media/ari/Harddisk/knee data/knee signal/'
        savePath = '/media/ari/Harddisk/knee data/knee signal/results/'
    # data locations home
    if location == 2:
        filePath = 'D:\\knee signal project\\daten\\Patienten\\'
        labelPath = 'D:\\knee signal project\\'
        savePath = 'D:\\knee signal project\\daten\\results\\'
    # data locations laptop

    # data locations server

    return filePath, labelPath, savePath