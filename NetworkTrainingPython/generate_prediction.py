import numpy as np
import pickle as pkl
import tensorflow as tf
import conv_layer_prediction
import dense_layer_prediction
import dense_layer_poly_mult
from write_weights import write_weights, read_weights
# from custom_bfv.bfv import BFV
from custom_bfv.bfv_ntt import BFV
import save_mnist_test

def load_pickle(filename):
  with open(f'{filename}', 'rb') as f:
    return pkl.load(f)

def main():
  model_name = 'simple_model'
  single_test = load_pickle('single_test.pkl')
  Xtest = single_test[0]
  Ytest = single_test[1]
  tf_test(model_name, Xtest, Ytest)
  custom_test(model_name, Xtest, Ytest)
  return

P_2_SCALE = 8

def scale_to_int(arr):
  scale = 2**P_2_SCALE
  rescaled = arr * scale
  return rescaled.astype(int)

def custom_test(model_name, Xtest, Ytest):

  # Load and reshape the test image
  Xtest = scale_to_int(Xtest)
  image_height, image_width, image_channels = Xtest.shape
  images = np.zeros( (image_channels, image_height, image_width), dtype=int )
  # images = np.zeros( (image_channels, image_height, image_width))
  for wi, w in enumerate(Xtest):
    for hi, h in enumerate(w):
      for ci, c in enumerate(h):
        images[ci][wi][hi] = c
  Xtest = images

  directory = f'./model_weights/{model_name}'

  # load in the weights for the conv2d layer
  conv2d_kernel = read_weights(f"{directory}/conv2d.kernel.txt")
  conv2d_bias = read_weights(f"{directory}/conv2d.bias.txt")

  enc_scheme = BFV(q = 2**38, t = 2**25, n = 2**10)

  output = conv_layer_prediction.conv_layer_prediction( Xtest, conv2d_kernel, enc_scheme )
  output = np.array(output)

  filters, width, height = output.shape
  for i in range(filters):
    for j in range(width):
      for k in range(height):
        '''
        if output[i][j][k] < 0:
          # print(f'relined: {output[i][j][k]}')
          output[i][j][k] = 0
        '''
        output[i][j][k] += conv2d_bias[i]
        output[i][j][k] //= 2**P_2_SCALE
        if output[i][j][k] < 0:
          output[i][j][k] = 0

  output = output.reshape(width*height*filters)

  dense_kernel = read_weights(f"{directory}/dense.kernel.txt")
  dense_bias = read_weights(f"{directory}/dense.bias.txt")

  output = dense_layer_prediction.dense_layer( output, dense_kernel, dense_bias)
  # output = dense_layer_poly_mult.dense_layer( enc_scheme, output, dense_kernel, dense_bias )

  # print(output)
  # return
  norm_scale = (2**P_2_SCALE)**2
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
  # print(f'Xtest shape: {Xtest.shape}')
  # print(f'Ytest: {Ytest}')
  # print(f'Ytest shape: {Ytest.shape}')

  Ypred = model.predict( Xtest )

  print(f'Prediction: [', end=' ')
  for pred in Ypred[0]:
    pred = float(pred)
    print(f'{pred:.7f}', end=' ')
  print(']')
  print(f'Acutal: {Ytest}')

def test_many_mnist_examples():
  for i in range(35):
    save_mnist_test.main(i)
    print(f'Test {i+1}')
    print('---'*8)
    model_name = 'simple_model'
    single_test = load_pickle('single_test.pkl')
    Xtest = single_test[0]
    Ytest = single_test[1]
    tf_test(model_name, Xtest, Ytest)
    custom_test(model_name, Xtest, Ytest)
    print('\n'*2)

if __name__ == '__main__':
  # test_many_mnist_examples()
  main()