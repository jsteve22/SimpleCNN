import numpy as np
from custom_bfv.poly import Poly
import custom_bfv.bfv


def single_convolution(og_image, image_height, image_width, og_kernel, kernel_height, kernel_width, enc_scheme):
    z1_height = image_height - kernel_height + 1
    z1_width = image_width - kernel_width + 1
    z1 = [ [0]*z1_width for _ in range(z1_height)] # [z1_height][z1_width]

    image = og_image.reshape(image_height * image_width)
    image = Poly(image.tolist())

    encrypted_image = enc_scheme.encrypt(image)

    kernel = og_kernel[0].tolist()
    
    for row in og_kernel[1:]:
        kernel.append([0]*(image_height - kernel_height))
        arr = []
        for j in row:
            arr.append(j)
        kernel.append(arr)
        
    kernel = Poly(kernel)
    
    output = enc_scheme.plaintext_mult(encrypted_image, kernel)
    output = enc_scheme.decrypt(output)

    for i in range(len(z1)):
        for j in range(len(z1[0])):
            oi = image_height - len(z1) + i
            oj = image_height - len(z1) + j
            z1[i][j] = output[oj + (image_width * oi)]

    return z1
