import numpy as np
from tensorflow import keras
from tensorflow.keras import layers

image_height = 28
image_width = 28

kernel_height = 3
kernel_width = 3

image = [image_height][image_width]
kernel = [kernel_height][kernel_width]

z1_height = image_height - kernel_height + 1
z1_width = image_width - kernel_width + 1

def single_convolution(kernel_height, kernel_width):
    z1 = [z1_height][z1_width]

    for i in range(z1_height):
        for j in range(z1_width):
            z1[i][j] = 0
            for k1 in range(kernel_height):
                for k2 in range(kernel_width):
                    z1[i][j] += image[i + k1][j + k2] * kernel[k1][k2]        
                    
    return z1

def multi_layer_convolution(kernel):
    z = []
    for f in kernel:
        z1 = single_convolution(len(f), len(f[0]))
        z.append(z1)

