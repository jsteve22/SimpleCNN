import numpy as np
from tensorflow import keras
from tensorflow.keras import layers


def conv_layer_prediction(images, kernels):
    z = multi_layer_convolution(images, kernels)
    return z

def single_convolution(image, image_height, image_width, kernel, kernel_height, kernel_width):
    z1_height = image_height - kernel_height + 1
    z1_width = image_width - kernel_width + 1
    z1 = [ [0]*z1_width for _ in range(z1_height)] # [z1_height][z1_width]

    for i in range(z1_height):
        for j in range(z1_width):
            z1[i][j] = 0
            for k1 in range(kernel_height):
                for k2 in range(kernel_width):
                    z1[i][j] += image[i + k1][j + k2] * kernel[k1][k2]        
                    
    return z1

def multi_layer_convolution(images, kernels):
    convolutions = []
    image_height = len(images[0])
    image_width = len(images[0][0])
    kernel_height = len(kernels[0][0])
    kernel_width  = len(kernels[0][0][0])
    z1_height = image_height - kernel_height + 1
    z1_width = image_width - kernel_width + 1
    for kernel in kernels:
        next_conv = np.zeros( (z1_height, z1_width) )
        for channel, image in zip(kernel, images):
            temp = single_convolution(image, image_height, image_width, channel, kernel_height, kernel_width)
            temp = np.array(temp)
            next_conv += temp
        convolutions.append(next_conv)

    return convolutions
