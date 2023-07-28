import numpy as np
import pickle as pkl
import tensorflow as tf
import conv_layer_prediction
import dense_layer_prediction
import dense_layer_poly_mult
import mean_pooling_layer_prediction
import max_pooling_layer
import batch_norm
from write_weights import write_weights, read_weights
# from custom_bfv.bfv import BFV
from custom_bfv.bfv_ntt import BFV
import save_mnist_test
import save_cifar_test
from print_outputs import print_1D_output, print_3D_output
import torch
import torchvision
from residual import residual
import read_model

def load_pickle(filename):
  with open(f'{filename}', 'rb') as f:
    return pkl.load(f)

P_2_SCALE = 8

def scale_to_int(arr):
  scale = 2**P_2_SCALE
  rescaled = arr * scale
  # rescaled = np.round(rescaled)
  return rescaled.astype(int)


def wrapper_conv_layer(input_layer, layer_path, pad=0, enc_scheme=None, stride=1):
  layer = input_layer.copy()
  if pad:
    layer = conv_layer_prediction.pad_images( layer, pad )
  output = conv_layer_prediction.conv_layer_prediction( layer, read_weights(layer_path), stride, enc_scheme )
  output = np.array(output)
  # output = scale_down(output, 2**P_2_SCALE)
  layer_name = layer_path.split('/')[-1].split('.')[0]
  print(f'{layer_name} Done')
  return output

def custom_test(model_name, Xtest):

  # Load and reshape the test image
  Xtest = scale_to_int(Xtest)
  image_height, image_width, image_channels = Xtest.shape
  # images = np.zeros( (image_channels, image_height, image_width), dtype=int )
  images = np.zeros( (image_channels, image_height, image_width))
  for wi, w in enumerate(Xtest):
    for hi, h in enumerate(w):
      for ci, c in enumerate(h):
        images[ci][wi][hi] = c
  Xtest = images

  #directory = f'./model_weights/{model_name}'

  enc_scheme = BFV(q = 2**38, t = 2**25, n = 2**10)

  OUTPUT_PRINT = lambda output: print(output.reshape( np.prod(output.shape) ).astype(int)[:100])

  output = read_model.use_model(f"./model_weights/{model_name}/summary.txt", Xtest, enc_scheme)

  print(f'Prediction: [', end=' ')
  for pred in output:
    pred = float(pred)
    print(f'{pred:.7f}', end=' ')
  print(']')
  return output

def tf_test(model_name, Xtest, Ytest):
  # model_name = 'small_model'
  model = tf.keras.models.load_model(f'./models/{model_name}.h5')
  # model = torchvision.models.resnet18()
  # model.load_state_dict(torch.load('./models/resnet18.pth'))

  Xtest = np.expand_dims(Xtest, 0)
  # Xtest = torch.Tensor(Xtest)
  # print(f'Xtest: {Xtest}')
  # print(f'Xtest shape: {Xtest.shape}')
  # print(f'Ytest: {Ytest}')
  # print(f'Ytest shape: {Ytest.shape}')

  Ypred = model.predict( Xtest )
  print(f'Prediction: [', end=' ')
  for pred in Ypred[0][:100]:
    pred = float(pred)
    print(f'{pred:.7f}', end=' ')
  print(']')
  print(f'Acutal: {Ytest}')
  return Ypred[0]
  model.eval()


  with torch.no_grad():
    Ypred = model(Xtest)
  print(f'Prediction: [', end=' ')
  for pred in Ypred[0][:100]:
    pred = float(pred)
    print(f'{pred:.7f}', end=' ')
  print(']')
  # print(f'Acutal: {Ytest}')
  return Ypred[0]

def main():
  num_tests = 10
  same_pred = 0
  total_diff = 0
  diff_arr = []
  for i in range(num_tests):
    save_cifar_test.main(i)
    print(f'Test {i+1}')
    print('---'*8)
    model_name = 'miniONN_cifar_model'
    single_test = load_pickle('cifar_test.pkl')
    Xtest = single_test[0]
    Ytest = single_test[1]
    tf_output  = tf_test(model_name, Xtest, Ytest)
    our_output = custom_test(model_name, Xtest)
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
  main()