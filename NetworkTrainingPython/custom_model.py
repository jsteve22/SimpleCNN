import numpy as np
import pickle as pkl
import tensorflow as tf
import conv_layer_prediction
import dense_layer_prediction
import dense_layer_poly_mult
import mean_pooling_layer_prediction
from write_weights import write_weights, read_weights
# from custom_bfv.bfv import BFV
from custom_bfv.bfv_ntt import BFV
import save_mnist_test
from generate_prediction import scale_down, ReLU, scale_to_int

def main():
  model_name = 'miniONN_cifar_model'
  single_test = load_pickle('cifar_test.pkl')
  Xtest = single_test[0]
  Ytest = single_test[1]
  custom_model = custom_load_miniONN()
  custom_predict(custom_model, Xtest)
  return

P_2_SCALE = 8

def custom_load_miniONN():
  model_name = 'miniONN_cifar_model'
  directory = f'./model_weights/{model_name}'

  model_layers = []

  def conv_func(input_layer, **kwargs):
    weights = kwargs['weights']
    padded = False if 'padded' not in kwargs else kwargs['padded']
    name = '' if 'name' not in kwargs else kwargs['name']

    layer = input_layer.copy()
    if (padded == True):
      layer = conv_layer_prediction.pad_images( layer )
    output = conv_layer_prediction.conv_layer_prediction( layer, weights )
    output = np.array(output)
    output = scale_down(output, 2**P_2_SCALE)
    output = ReLU(output)
    print(f'{name} Done')
    return output
  
  def mean_func(input_layer, **kwargs):
    return mean_pooling_layer_prediction.mean_pooling_layer( input_layer )

  def dense_func(input_layer, **kwargs):
    weights = kwargs['weights']
    name = '' if 'name' not in kwargs else kwargs['name']

    output = dense_layer_prediction.dense_layer( input_layer, weights )
    for ind, val in enumerate(output):
      output[ind] = val // (2**P_2_SCALE)
    print(f'{name} Done')
    return output

  def reshape_func(input_layer, **kwargs):
    filters, width, height = input_layer.shape
    output = input_layer.reshape(width*height*filters)
    return output
  
  def softmax_func(input_layer, **kwargs):
    output = input_layer.copy()
    for ind, val in enumerate(output):
      output[ind] = val // (2**P_2_SCALE)
    output = dense_layer_prediction.softmax( output )
    return output
  
  model_layers.append({'func': conv_func, 'weights':read_weights(f'{directory}/conv2d.kernel.txt'), 'padded':True, 'name':'conv2d'})
  model_layers.append({'func': conv_func, 'weights':read_weights(f'{directory}/conv2d_1.kernel.txt'), 'padded':True, 'name':'conv2d_1'})
  model_layers.append({'func': mean_func})
  model_layers.append({'func': conv_func, 'weights':read_weights(f'{directory}/conv2d_2.kernel.txt'), 'padded':True, 'name':'conv2d_2'})
  model_layers.append({'func': conv_func, 'weights':read_weights(f'{directory}/conv2d_3.kernel.txt'), 'padded':True, 'name':'conv2d_3'})
  model_layers.append({'func': mean_func})
  model_layers.append({'func': conv_func, 'weights':read_weights(f'{directory}/conv2d_4.kernel.txt'), 'padded':True, 'name':'conv2d_4'})
  model_layers.append({'func': conv_func, 'weights':read_weights(f'{directory}/conv2d_5.kernel.txt'), 'name':'conv2d_5'})
  model_layers.append({'func': conv_func, 'weights':read_weights(f'{directory}/conv2d_6.kernel.txt'), 'name':'conv2d_6'})
  model_layers.append({'func': reshape_func})
  model_layers.append({'func': dense_func, 'weights':read_weights(f'{directory}/dense.kernel.txt'), 'name':'dense'})
  model_layers.append({'func': softmax_func})
  return model_layers

def custom_predict(custom_layers, Xtest):
  Xtest = scale_to_int(Xtest)
  image_height, image_width, image_channels = Xtest.shape
  images = np.zeros( (image_channels, image_height, image_width), dtype=int )
  # images = np.zeros( (image_channels, image_height, image_width))
  for wi, w in enumerate(Xtest):
    for hi, h in enumerate(w):
      for ci, c in enumerate(h):
        images[ci][wi][hi] = c
  output = images

  for layer in custom_layers:
    func = list(layer.values())[0]
    output = func(output, **layer)

  print(f'Prediction: [', end=' ')
  for pred in output:
    pred = float(pred)
    print(f'{pred:.7f}', end=' ')
  print(']')
  return output

if __name__ == '__main__':
  main()