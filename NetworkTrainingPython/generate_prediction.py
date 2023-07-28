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
from print_outputs import print_1D_output, print_3D_output
import torch
import torchvision
from residual import residual
import read_model

def load_pickle(filename):
  with open(f'{filename}', 'rb') as f:
    return pkl.load(f)

def main():
  model_name = 'miniONN_cifar_model'
  # model_name = 'resnet18'
  single_test = load_pickle('cifar_test.pkl')
  Xtest = single_test[0]
  Ytest = single_test[1]
  # jXtest = [[[1]*64] * 64]*3
  # Xtest = np.array(Xtest)
  # Ytest = [[[1]*64] * 64]
  # Ytest = np.array(Ytest)
  # Ytest = np.reshape(Ytest, (64, 64, 1))
  tf_test(model_name, Xtest, Ytest)
  # Xtest = np.reshape(Xtest, (64, 64, 3))
  custom_test(model_name, Xtest)
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
  # ret = scale_down(ret, 2**P_2_SCALE)
  return ret

def scale_down(arr, scale):
  ret = arr.copy()
  
  for i, image in enumerate(ret):
    for j, row in enumerate(image):
      for k, val in enumerate(row):
        ret[i][j][k] = val // scale
  return ret

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
  return 

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
  print(Ypred[0])
  return
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

if __name__ == '__main__':
  # test_many_mnist_examples()
  main()