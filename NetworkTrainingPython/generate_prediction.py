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
  model_name = '4_layer_mnist_model'
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
  # rescaled = np.round(rescaled)
  return rescaled.astype(int)

def ReLU(arr):
  ret = arr.copy()
  
  for i, image in enumerate(ret):
    for j, row in enumerate(image):
      for k, val in enumerate(row):
        if val < 0:
          ret[i][j][k] = 0
  return ret

def scale_down(arr, scale):
  ret = arr.copy()
  
  for i, image in enumerate(ret):
    for j, row in enumerate(image):
      for k, val in enumerate(row):
        ret[i][j][k] = val // scale
  return ret

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
  # conv2d_bias = read_weights(f"{directory}/conv2d.bias.txt")

  enc_scheme = BFV(q = 2**38, t = 2**25, n = 2**10)

  Xtest = conv_layer_prediction.pad_images( Xtest )

  output = conv_layer_prediction.conv_layer_prediction( Xtest, conv2d_kernel, enc_scheme )
  output = np.array(output)
  filters, width, height = output.shape
  # output = ReLU(output)
  output = scale_down(output, 2**P_2_SCALE)
  print(output.reshape(filters*width*height)[:150])
  print('-'*30)

  output = conv_layer_prediction.pad_images( output )
  output = conv_layer_prediction.conv_layer_prediction( output, read_weights(f"{directory}/conv2d_1.kernel.txt"), enc_scheme )
  output = np.array(output)
  filters, width, height = output.shape
  # output = ReLU(output)
  output = scale_down(output, 2**P_2_SCALE)
  print(output.reshape(filters*width*height)[:150])
  print('-'*30)

  output = conv_layer_prediction.pad_images( output )
  output = conv_layer_prediction.conv_layer_prediction( output, read_weights(f"{directory}/conv2d_2.kernel.txt"), enc_scheme )
  output = np.array(output)
  filters, width, height = output.shape
  # output = ReLU(output)
  output = scale_down(output, 2**P_2_SCALE)
  print(output.reshape(filters*width*height)[:150])
  print('-'*30)

  output = conv_layer_prediction.pad_images( output )
  output = conv_layer_prediction.conv_layer_prediction( output, read_weights(f"{directory}/conv2d_3.kernel.txt"), enc_scheme )
  output = np.array(output)
  output = ReLU(output)
  filters, width, height = output.shape
  output = scale_down(output, 2**P_2_SCALE)
  print(output.reshape(filters*width*height)[:150])
  print('-'*30)

  filters, width, height = output.shape
  output = output.reshape(width*height*filters)

  putput = [ o % 2**32 for o in output ]
  putput = [ o if o < 2**31 else 0 for o in putput ]
  print(putput[:75])
  dense_kernel = read_weights(f"{directory}/dense.kernel.txt")
  # dense_bias = read_weights(f"{directory}/dense.bias.txt")

  output = dense_layer_prediction.dense_layer( output, dense_kernel)
  # output = dense_layer_poly_mult.dense_layer( enc_scheme, output, dense_kernel, dense_bias )
  putput = [ o % 2**32 for o in output ]
  putput = [ o if o < 2**31 else 0 for o in putput ]
  print(putput[:75])
  return

  # PLAINTEXT_MODULUS = 2061584302081
  # output = [o % (2**32) for o in output]
  # output = [o - (2**32) if o > (2**31) else o for o in output]

  max_scale = max(output)
  norm_scale = (2**P_2_SCALE)**6
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
  return output

def tf_test(model_name, Xtest, Ytest):
  # model_name = 'small_model'
  model = tf.keras.models.load_model(f'./models/{model_name}.h5')

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
  return Ypred[0]

def test_many_mnist_examples():
  num_tests = 20
  same_pred = 0
  total_diff = 0
  diff_arr = []
  for i in range(num_tests):
    save_mnist_test.main(i)
    print(f'Test {i+1}')
    print('---'*8)
    model_name = '4_layer_mnist_model'
    single_test = load_pickle('single_test.pkl')
    Xtest = single_test[0]
    Ytest = single_test[1]
    tf_output  = tf_test(model_name, Xtest, Ytest)
    our_output = custom_test(model_name, Xtest, Ytest)
    if ( np.argmax(tf_output) == np.argmax(our_output) ):
      same_pred += 1
    diff = sum(map(abs, [t-o for t,o in zip(tf_output, our_output) ]))
    diff_arr.append( diff )
    total_diff += diff
    print(f'diff: {diff}')
    print('\n'*2)
  print(f'total diff:   {total_diff}')
  print(f'average diff: {total_diff / num_tests}')
  print(f'standard dev: {np.std(diff_arr)}')
  print(f'same pred:    {same_pred} / {num_tests}')

if __name__ == '__main__':
  # test_many_mnist_examples()
  main()