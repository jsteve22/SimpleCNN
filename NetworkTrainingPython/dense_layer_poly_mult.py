import numpy as np
import pickle as pkl

from custom_bfv.bfv import BFV
from custom_bfv.poly import Poly


def dense_layer(enc_scheme, input_layer, kernel, bias=None):
  '''
  output_count, length = kernel.shape
  i = Poly(input_layer.astype(int).tolist())
  k = Poly(np.flip(kernel[0]).astype(int).tolist())
  res = i * k
  return res[length-1]
  '''
  # hardcode this for now
  output_count, length = kernel.shape
  split = 4
  length = length // split


  bias = bias if bias is None else [0]*output_count

  poly_input = input_layer.astype(int).tolist()
  enc_input_layers = []
  for i in range(split):
    enc_input_layer = enc_scheme.encrypt(Poly(list(poly_input[i*length:(i+1)*length])))
    # enc_input_layer = (Poly(list(poly_input[i*length:(i+1)*length])))
    enc_input_layers.append( enc_input_layer )

  output_layer = [0]*output_count
  for ind in range(output_count):
    kernel_layer = kernel[ind]
    # kernel_layer = np.flip(kernel_layer).astype(int).tolist()
    for i in range(split):
      spec_layer = list(kernel_layer[i*length:(i+1)*length])
      spec_layer = np.flip(spec_layer).astype(int).tolist()
      poly_layer = Poly(spec_layer)
      poly_product = enc_scheme.plaintext_mult(enc_input_layers[i], poly_layer)
      decrypted    = enc_scheme.decrypt(poly_product)
      # decrypted = enc_input_layers[i] * poly_layer
      res = decrypted[ length-1 ]
      if res >= enc_scheme.t//2:
        res -= enc_scheme.t
      output_layer[ind] += res
    output_layer[ind] += bias[ind]
  return output_layer