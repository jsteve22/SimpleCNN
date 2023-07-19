import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
import conv_layer_poly_mult


def conv_layer_prediction(images, kernels, stride=1, BFV=None, padding=0):
    if padding:
        for ind, image in enumerate(images):
            images[ind] = pad_image( image, padding )

    z = multi_layer_convolution(images, kernels, stride, BFV)
    return z

def single_convolution(image, image_height, image_width, kernel, kernel_height, kernel_width, stride=1, BFV=None):
    z1_height = (image_height - kernel_height) // stride + 1
    z1_width = (image_width - kernel_width) // stride + 1
    z1 = [ [0]*z1_width for _ in range(z1_height)] # [z1_height][z1_width]

    i = 0
    j = 0
    for img_i in range(0, image_height - kernel_height + 1, stride):
        for img_j in range(0, image_width - kernel_width + 1, stride):
            z1[i][j] = 0
            for k1 in range(kernel_height):
                for k2 in range(kernel_width):
                    z1[i][j] += image[img_i + k1][img_j + k2] * kernel[k1][k2]   
            j += 1 
        j = 0
        i += 1  
                    
    return z1

def multi_layer_convolution(images, kernels, stride=1, BFV=None):
    convolutions = []
    image_height = len(images[0])
    image_width = len(images[0][0])
    kernel_height = len(kernels[0][0])
    kernel_width  = len(kernels[0][0][0])
    z1_height = (image_height - kernel_height) // stride + 1
    z1_width = (image_width - kernel_width) // stride + 1
    for kernel in kernels:
        next_conv = np.zeros( (z1_height, z1_width) )
        for channel, image in zip(kernel, images):
            # temp = conv_layer_poly_mult.single_convolution(image, image_height, image_width, channel, kernel_height, kernel_width, BFV)
            temp = single_convolution(image, image_height, image_width, channel, kernel_height, kernel_width, stride, BFV)
            temp = np.array(temp)
            next_conv += temp
        convolutions.append(next_conv)

    return convolutions

def pad_images(images, pad):
    ret = []
    for image in images:
        ret.append( pad_image(image, pad) )
    arr = np.array(ret)
    return arr

def pad_image(image, pad):
    width, height = image.shape
    padded_image = np.zeros( (width+pad*2, height+pad*2), dtype=image.dtype )
    for ind, row in enumerate(image):
        for jnd, value in enumerate(row):
            padded_image[ind + pad][jnd + pad] = value
    return padded_image