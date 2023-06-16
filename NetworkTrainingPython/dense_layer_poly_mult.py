import numpy as np
import pickle as pkl

from custom_bfv.bfv import BFV
from custom_bfv.poly import Poly


def dense_layer(input_layer, kernel, bias, enc_scheme):
  output_count, length = kernel.shape

  enc_input_layer = enc_scheme.encrypt(input_layer)

  output_layer = [0]*output_count
  for ind in range(output_count):
    kernel_layer = kernel[ind]
    kernel_layer = np.flip(kernel_layer)
    kernel_layer = Poly(kernel_layer.tolist())
    poly_product = enc_scheme.plaintext_mult(enc_input_layer, kernel_layer)
    decrypted    = enc_scheme.decrypt(poly_product)
    output_layer[ind] += decrypted[ length-1 ]
    output_layer[ind] += bias
  pass