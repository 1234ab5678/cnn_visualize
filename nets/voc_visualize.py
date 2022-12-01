import os
from PIL import Image
import numpy as np

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    classes_nums = np.zeros([256], np.int)
    image = Image.open('./factory_in_10206.png')
    image = np.array(image)
    print(image)
    classes_nums += np.bincount(np.reshape(image, [-1]), minlength=256)
    print(classes_nums)
    size = image.shape
    label_temp=np.zeros([size[0], size[1], 3],np.uint8)

    for i in range(0, 3):
        if(i==0):
            continue
        elif(i==1):
            print(size)
            for j in range(size[0]):
                for k in range(size[1]):
                    if (image[j, k] == i):
                        label_temp[j, k, 0] = 255
                        label_temp[j, k, 1] = 0
                        label_temp[j, k, 2] = 0

        elif (i == 2):
            print(size)
            for j in range(size[0]):
                for k in range(size[1]):
                    if (image[j, k] == i):
                        label_temp[j, k, 0] = 0
                        label_temp[j, k, 1] = 255
                        label_temp[j, k, 2] = 0

    label_temp = Image.fromarray(label_temp)
    label_temp.save('./test.png')