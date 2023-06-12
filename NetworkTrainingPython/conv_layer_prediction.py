import numpy as np
from tensorflow import keras
from tensorflow.keras import layers


def conv_layer_prediction(image, kernels):
    z = multi_layer_convolution(image, kernels)
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

def multi_layer_convolution(image, kernels):
    convolutions = []
    image_height = len(image)
    image_width = len(image[0])
    for kernel in kernels:
        temp = single_convolution(image, image_height, image_width, kernel.T, len(kernel), len(kernel[0]))
        temp = np.array(temp)
        convolutions.append(temp.T)

    return convolutions
