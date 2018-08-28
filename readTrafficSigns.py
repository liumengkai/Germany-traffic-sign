# The German Traffic Sign Recognition Benchmark
#
# sample code for reading the traffic sign images and the
# corresponding labels
#
# example:
#
# trainImages, trainLabels = readTrafficSigns('GTSRB/Training')
# print len(trainLabels), len(trainImages)
# plt.imshow(trainImages[42])
# plt.show()
#
# have fun, Christian

import matplotlib.pyplot as plt
import csv
import matplotlib
import skimage.data
import skimage.transform
# function for reading the images
# arguments: path to the traffic sign data, for example './GTSRB/Training'
# returns: list of images, list of corresponding labels
def readTrafficSigns(rootpath):
    '''Reads traffic sign data for German Traffic Sign Recognition Benchmark.

    Arguments: path to the traffic sign data, for example './GTSRB/Training'
    Returns:   list of images, list of corresponding labels'''
    images = [] # images
    labels = [] # corresponding labels
    # loop over all 42 classes
    for c in range(0,2):
        prefix = rootpath + '/' + format(c, '05d') + '/' # 先打开类文件夹
        gtFile = open(prefix + 'GT-'+ format(c, '05d') + '.csv') # 再打开csv注释文件
        gtReader = csv.reader(gtFile, delimiter=';') # 用分号隔开每个元素，下一步可以按行读取
        next(gtReader) # 跳过第一行
        # 读取csv文件中的每一行并且提取出来第一个元素是要打开的文件名，第八行是标签
        for row in gtReader:
            images.append(plt.imread(prefix + row[0])) # the 1th column is the filename
            labels.append(row[7]) # the 8th column is the label
        gtFile.close()
    images=[skimage.transform.resize(image,(28,28),mode='constant')
          for image in images]
    print(type(images[0]),type(labels[0]))
    return images,labels
readTrafficSigns('D:/files/比赛数据/交通标志识别数据/GTSRB/Final_Training/Images')






