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
from print_outputs import print_1D_output, print_3D_output

def load_pickle(filename):
  with open(f'{filename}', 'rb') as f:
    return pkl.load(f)

def main():
  model_name = 'miniONN_cifar_model'
  single_test = load_pickle('cifar_test.pkl')
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
  ret = scale_down(ret, 2**P_2_SCALE)
  return ret

def scale_down(arr, scale):
  ret = arr.copy()
  
  for i, image in enumerate(ret):
    for j, row in enumerate(image):
      for k, val in enumerate(row):
        ret[i][j][k] = val // scale
  return ret

def wrapper_conv_layer(input_layer, layer_path, padded=False, enc_scheme=None):
  layer = input_layer.copy()
  if (padded == True):
    layer = conv_layer_prediction.pad_images( layer )
  output = conv_layer_prediction.conv_layer_prediction( layer, read_weights(layer_path), enc_scheme )
  output = np.array(output)
  # output = scale_down(output, 2**P_2_SCALE)
  layer_name = layer_path.split('/')[-1].split('.')[0]
  print(f'{layer_name} Done')
  return output

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

  enc_scheme = BFV(q = 2**38, t = 2**25, n = 2**10)

  OUTPUT_PRINT = lambda output: print(output.reshape( np.prod(output.shape) ).astype(int)[:100])

  output = wrapper_conv_layer( Xtest, f'{directory}/conv2d.kernel.txt', padded=True, enc_scheme=enc_scheme )
  print_3D_output('./output_files/1_conv_output.txt', output)
  output = ReLU(output)
  print_1D_output('./output_files/2_relu_output.txt', output)
  output = wrapper_conv_layer( output, f'{directory}/conv2d_1.kernel.txt', padded=True, enc_scheme=enc_scheme )
  print_3D_output('./output_files/3_conv_output.txt', output)
  output = ReLU(output)
  print_1D_output('./output_files/4_relu_output.txt', output)
  output = mean_pooling_layer_prediction.mean_pooling_layer( output )
  print_3D_output('./output_files/5_meanpool_output.txt', output)
  output = wrapper_conv_layer( output, f'{directory}/conv2d_2.kernel.txt', padded=True, enc_scheme=enc_scheme )
  print_3D_output('./output_files/6_conv_output.txt', output)
  output = ReLU(output)
  print_1D_output('./output_files/7_relu_output.txt', output)
  output = wrapper_conv_layer( output, f'{directory}/conv2d_3.kernel.txt', padded=True, enc_scheme=enc_scheme )
  print_3D_output('./output_files/8_conv_output.txt', output)
  output = ReLU(output)
  print_1D_output('./output_files/9_relu_output.txt', output)
  output = mean_pooling_layer_prediction.mean_pooling_layer( output )
  print_3D_output('./output_files/10_meanpool_output.txt', output)
  output = wrapper_conv_layer( output, f'{directory}/conv2d_4.kernel.txt', padded=True, enc_scheme=enc_scheme )
  print_3D_output('./output_files/11_conv_output.txt', output)
  output = ReLU(output)
  print_1D_output('./output_files/12_relu_output.txt', output)
  output = wrapper_conv_layer( output, f'{directory}/conv2d_5.kernel.txt', enc_scheme=enc_scheme )
  print_3D_output('./output_files/13_conv_output.txt', output)
  output = ReLU(output)
  print_1D_output('./output_files/14_relu_output.txt', output)
  output = wrapper_conv_layer( output, f'{directory}/conv2d_6.kernel.txt', enc_scheme=enc_scheme )
  print_3D_output('./output_files/15_conv_output.txt', output)
  output = ReLU(output)
  print_1D_output('./output_files/16_relu_output.txt', output)

  filters, width, height = output.shape
  output = output.reshape(width*height*filters)

  dense_kernel = read_weights(f"{directory}/dense.kernel.txt")
  # output = dense_layer_prediction.dense_layer( output, dense_kernel)
  dense_output = [0] * 10

  split = 4000
  for i in range(0, width*height*filters, split):
    temp = dense_layer_prediction.dense_layer( output[i:min(i+split, width*height*filters - i)], dense_kernel[:][i:min(i+split, width*height*filters - i)])
    for j in range(len(temp)):
      dense_output[j] += temp[j]

  output = dense_output

  print(f'Prediction: [', end=' ')
  for pred in output:
    pred = float(pred)
    print(f'{pred:.7f}', end=' ')
  print(']')
  return output

  max_scale = max(output)
  norm_scale = (2**P_2_SCALE)**2
  for ind, val in enumerate(output):
    output[ind] = val // norm_scale
    pass

  output = dense_layer_prediction.softmax( output )
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