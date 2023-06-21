import numpy as np
import pickle as pkl
import tensorflow as tf
import conv_layer_prediction
import dense_layer_prediction
import dense_layer_poly_mult
from write_weights import write_weights, read_weights
# from custom_bfv.bfv import BFV
from custom_bfv.bfv_ntt import BFV

def load_pickle(filename):
  with open(f'{filename}', 'rb') as f:
    return pkl.load(f)

def main():
  model_name = 'small_model'
  single_test = load_pickle('single_test.pkl')
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

  # Load and reshape the test image
  Xtest = scale_to_int(Xtest)
  image_height, image_width, image_channels = Xtest.shape
  images = np.zeros( (image_channels, image_height, image_width), dtype=int )
  for wi, w in enumerate(Xtest):
    for hi, h in enumerate(w):
      for ci, c in enumerate(h):
        images[ci][wi][hi] = c
  Xtest = images

  # load in the weights for the conv2d layer
  conv2d_kernel = read_weights("conv2d.kernel.txt")
  conv2d_bias = read_weights("conv2d.bias.txt")

  enc_scheme = BFV(q = 2**38, t = 2**25, n = 2**10)

  output = conv_layer_prediction.conv_layer_prediction( Xtest, conv2d_kernel, enc_scheme )
  output = np.array(output)

  filters, width, height = output.shape
  for i in range(filters):
    for j in range(width):
      for k in range(height):
        if output[i][j][k] < 0:
          output[i][j][k] = 0
        continue
        output[i][j][k] += conv2d_bias[i]
        output[i][j][k] //= 2**8
        if output[i][j][k] < 0:
          output[i][j][k] = 0

  output = output.reshape(width*height*filters)

  dense_kernel = read_weights("dense.kernel.txt")
  dense_bias = read_weights("dense.bias.txt")

  output = dense_layer_prediction.dense_layer( output, dense_kernel, dense_bias)
  # output = dense_layer_poly_mult.dense_layer( enc_scheme, output, dense_kernel, dense_bias )

  print(output)
  return
  norm_scale = (2**8)**2
  for ind, val in enumerate(output):
    output[ind] = val // norm_scale
    pass
    # print(f'{val/max_scale}', end=' ')

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