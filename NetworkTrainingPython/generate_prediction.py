import numpy as np
import pickle as pkl
import tensorflow as tf
import conv_layer_prediction
import dense_layer_prediction

def load_pickle(filename):
  with open(f'{filename}', 'rb') as f:
    return pkl.load(f)

def main():
  model_name = 'cifar_model'
  single_test = load_pickle('cifar_test.pkl')
  Xtest = single_test[0]
  Ytest = single_test[1]
  tf_test(model_name, Xtest, Ytest)
  print()
  print()
  print()
  custom_test(model_name, Xtest, Ytest)
  return

def scale_to_int(arr):
  scale = 2**8
  rescaled = arr * scale
  return rescaled.astype(int)

def custom_test(model_name, Xtest, Ytest):
  # Xtest = Xtest.reshape( (28,28) )

  # model_name = 'conv_model'
  model = tf.keras.models.load_model(f'{model_name}.h5')
  weights_dictionary = model.get_weight_paths()
  # print(weights_dictionary.keys())

  conv2d_kernel = weights_dictionary['conv2d.kernel'].numpy()
  conv2d_bias   = weights_dictionary['conv2d.bias'].numpy()

  Xtest = scale_to_int(Xtest)
  conv2d_kernel = scale_to_int(conv2d_kernel)
  conv2d_bias = scale_to_int(conv2d_bias)

  width, height, channels, filters = conv2d_kernel.shape
  # print(conv2d_kernel.T.shape)

  kernel = np.zeros( (filters, channels, width, height) )

  for wi, w in enumerate(conv2d_kernel):
    for hi, h in enumerate(w):
      for ci, c in enumerate(h):
        for fi, f in enumerate(c):
          kernel[fi][ci][wi][hi] = f 
  conv2d_kernel = kernel

  image_height, image_width, image_channels = Xtest.shape
  images = np.zeros( (image_channels, image_height, image_width) )
  for wi, w in enumerate(Xtest):
    for hi, h in enumerate(w):
      for ci, c in enumerate(h):
        images[ci][wi][hi] = c
  Xtest = images

  output = conv_layer_prediction.conv_layer_prediction( Xtest, conv2d_kernel )
  output = np.array(output)

  filters, width, height = output.shape
  for i in range(filters):
    for j in range(width):
      for k in range(height):
        output[i][j][k] += conv2d_bias[i]
        if output[i][j][k] < 0:
          output[i][j][k] = 0

  # output = output.T
  ## output = output.reshape((26*26*4))
  # output = output.reshape(30*30*filters)
  '''
  temp = np.zeros( (width, height, filters) )
  for fi, f in enumerate(output):
    for wi, w in enumerate(f):
      for hi, h in enumerate(w):
        temp[wi][hi][fi] = h
  output = temp
  '''

  output = output.reshape(width*height*filters)

  output = dense_layer_prediction.wrapper_dense_layer( output, 'dense', weights_dictionary, (filters, width, height) )
  # print(output)
  max_scale = max(output)
  norm_scale = (2**8)**3
  for ind, val in enumerate(output):
    output[ind] = val / norm_scale
    # print(f'{val/max_scale}', end=' ')

  # print(output)
  output = dense_layer_prediction.softmax( output )
  # print( output )
  print(f'Prediction: [', end=' ')
  for pred in output:
    pred = float(pred)
    print(f'{pred:.7f}', end=' ')
  print(']')

def tf_test(model_name, Xtest, Ytest):
  # model_name = 'small_model'
  model = tf.keras.models.load_model(f'{model_name}.h5')

  Xtest = np.expand_dims(Xtest, 0)
  # print(f'Xtest: {Xtest}')
  print(f'Xtest shape: {Xtest.shape}')
  print(f'Ytest: {Ytest}')
  print(f'Ytest shape: {Ytest.shape}')

  Ypred = model.predict( Xtest )

  print(f'Prediction: [', end=' ')
  for pred in Ypred[0]:
    pred = float(pred)
    print(f'{pred:.7f}', end=' ')
  print(']')
  print(f'Acutal: {Ytest}')

if __name__ == '__main__':
  main()