import matplotlib
import skimage.data
import skimage.transform
import matplotlib.pyplot as plt
import tensorflow as tf
from readTrafficSigns import readTrafficSigns
def display_images_and_labels(images, labels):
    """Display the first image of each label."""
    unique_labels = set(labels)
    plt.figure(figsize=(12,12))
    i = 1
    for label in unique_labels:
        # Pick the first image for each label.
        image = images[labels.index(label)]
        plt.subplot(7, 7, i)  # A grid of 8 rows x 8 columns
        plt.axis('off')
        plt.title("Label {0} ({1})".format(label, labels.count(label)))
        i += 1
        _ = plt.imshow(image)
    plt.show()
images,labels=readTrafficSigns('D:/files/比赛数据/交通标志识别数据/GTSRB/Final_Training/Images')
for image in images[:5]:
    print("shape:{0},min:{1},max:{2}".format(image.shape,image.min(),image.max()))
images32=[skimage.transform.resize(image,(12,12),mode='constant')
          for image in images]
for image in images32[:5]:
    print("shape:{0},min:{1},max:{2}".format(image.shape,image.min(),image.max()))
display_images_and_labels(images,labels)