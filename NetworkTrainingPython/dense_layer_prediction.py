import numpy as np
import pickle as pkl
import tensorflow as tf
from tensorflow.keras.models import load_model

def main():
  model_name = 'simple_model'
  model = load_model(f'{model_name}.h5')

  weights_dictionary = model.get_weight_paths()

  print( weights_dictionary.keys() )

  dense_kernel = weights_dictionary['dense.kernel'].numpy()
  dense_bias   = weights_dictionary['dense.bias'].numpy()

  # transpose the matrix so that the shape is (# of outputs, # of values for dot prod)
  dense_kernel = dense_kernel.T
  outputs, length = dense_kernel.shape

  print()
  test_softmax()

  pass

def dot_product(vec_a, vec_b, size=None):
  return sum([val_a*val_b for val_a, val_b in zip(vec_a, vec_b)])
  '''
  size = size if size else min(len(vec_a), len(vec_b))
  total = 0
  for ind in range(size):
    total += (vec_a[ind] * vec_b[ind])
  return total
  '''

# it is assumed that the kernel and bias will be numpy arrays and that they have been processed before
def dense_layer(input_layer, kernel, bias):
  output_count, length = kernel.shape

  output_layer = [0]*output_count
  for ind in range(output_count):
    dot_sum = dot_product(input_layer, kernel[ind])
    output_layer[ind] += dot_sum
    output_layer[ind] += bias[ind]

  return output_layer

def wrapper_dense_layer(input_layer, layer_name, weights_dictionary):
  dense_kernel = weights_dictionary[f'{layer_name}.kernel'].numpy()
  dense_bias   = weights_dictionary[f'{layer_name}.bias'].numpy()
  
  dense_kernel = dense_kernel.T
  return dense_layer(input_layer, dense_kernel, dense_bias)

def softmax(input_layer):
  from math import exp
  exp_sum = sum( [exp(val) for val in input_layer] )
  output_layer = [exp(val) / exp_sum for val in input_layer]
  return output_layer

def test_softmax():
  arr = [1, 3, 2]
  print( softmax(arr) )
  return

def test_dot_product():
  a = [1, 2, 3, 4, 5]
  b = [5, 4, 3, 2, 1]
  print(dot_product(a, b))
  return

if __name__ == '__main__':
  main()