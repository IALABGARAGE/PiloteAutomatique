import numpy as np
import matplotlib.image as mpli

IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS = 256, 64, 1
INPUT_SHAPE = (IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS)

def black_white(original):
    w, h, c = original.shape
    image = np.array([[0]*h]*w)

    for i in range(w):
        for j in range(h):
            sum = 0
            for chan in range(c):
                sum += original[i, j, chan]

            image[i, j] = sum/3.0

    return image

def flip(original):
    #TODO
    return original

def batch_generator(X, Y, batch_size, is_training):
    input = np.empty([batch_size, IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS])
    output = np.empty([batch_size, 2])

    while True:
        i = 0
        for index in np.random.permutation(X.shape[0]):
            image = X[index]
            angle, speed = Y[index]

            image = mpimg.imread(image)

            # convert to b&w
            image = black_white(image)

            # horizontal flip
            if is_training and np.random.rand() < 0.5:
                image = flip(image)
                angle = -angle

            # add the image and steering angle to the batch
            input[i] = image
            output[i][0] = angle
            output[i][1] = speed

            i += 1
            if i == batch_size:
                break

        yield input, output