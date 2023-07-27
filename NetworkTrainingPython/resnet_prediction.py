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

def load_pickle(filename):
  with open(f'{filename}', 'rb') as f:
    return pkl.load(f)

def main():
  # model_name = 'miniONN_cifar_model'
  model_name = 'resnet18'
  # single_test = load_pickle('cifar_test.pkl')
  # Xtest = single_test[0]
  # Ytest = single_test[1]
  num = 1 / 256
  Xtest = [[[num]*64] * 64]*3
  Xtest = np.array(Xtest)
  Ytest = [[[num]*64] * 64]
  Ytest = np.array(Ytest)
  Ytest = np.reshape(Ytest, (64, 64, 1))
  tf_test(model_name, Xtest, Ytest)
  Xtest = np.reshape(Xtest, (64, 64, 3))
  custom_test(model_name, Xtest)
  return

P_2_SCALE = 16

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

def wrapper_conv_layer(input_layer, layer_path, pad=0, enc_scheme=None, stride=1, as_ints=False):
  layer = input_layer.copy()
  if pad:
    layer = conv_layer_prediction.pad_images( layer, pad )
  output = conv_layer_prediction.conv_layer_prediction( layer, read_weights(layer_path), stride, enc_scheme )
  output = np.array(output)
  if (as_ints):
    output = scale_down(output, 2**P_2_SCALE)
  layer_name = layer_path.split('/')[-1].split('.')[0]
  print(f'{layer_name} Done')
  return output

def custom_test(model_name, Xtest):

  as_ints = True

  # Load and reshape the test image
  # Xtest = scale_to_int(Xtest)
  image_height, image_width, image_channels = Xtest.shape
  images = np.zeros( (image_channels, image_height, image_width))

  if (as_ints):
    images = np.zeros( (image_channels, image_height, image_width), dtype=int )
    model_name = 'int_' + model_name

  for wi, w in enumerate(Xtest):
    for hi, h in enumerate(w):
      for ci, c in enumerate(h):
        images[ci][wi][hi] = c
  Xtest = images

  directory = f'./model_weights/{model_name}'

  enc_scheme = BFV(q = 2**38, t = 2**25, n = 2**10)

  OUTPUT_PRINT = lambda output: print(output.reshape( np.prod(output.shape) ).astype(int)[:100])
  
  def basic_block(inp, layer, first_stride=1, as_ints=False):
    output = wrapper_conv_layer(inp, f'{directory}/layer{layer}.conv1.weight.txt', pad=1, enc_scheme=enc_scheme, stride=first_stride, as_ints=as_ints)
    output = batch_norm.batch_main(output, f'{directory}/layer{layer}.bn1.weight.txt', f'{directory}/layer{layer}.bn1.bias.txt', f'{directory}/layer{layer}.bn1.running_mean.txt', f'{directory}/layer{layer}.bn1.running_var.txt', as_ints=as_ints)
    output = ReLU(output)
    output = wrapper_conv_layer(output, f'{directory}/layer{layer}.conv2.weight.txt', pad=1, enc_scheme=enc_scheme, stride=1, as_ints=as_ints)
    output = batch_norm.batch_main(output, f'{directory}/layer{layer}.bn2.weight.txt', f'{directory}/layer{layer}.bn2.bias.txt', f'{directory}/layer{layer}.bn2.running_mean.txt', f'{directory}/layer{layer}.bn2.running_var.txt', as_ints=as_ints)
    # print(inp.shape, output.shape)
    print(f'inp.shape: {inp.shape}')
    print(f'output.shape: {output.shape}')
    print()
    # output = ReLU(output)
    return output
  
  def downsample(inp, layer, as_ints=False):
    output = wrapper_conv_layer(inp, f'{directory}/layer{layer}.downsample.0.weight.txt', pad=0, enc_scheme=enc_scheme, stride=2, as_ints=as_ints)
    output = batch_norm.batch_main(output, f'{directory}/layer{layer}.downsample.1.weight.txt', f'{directory}/layer{layer}.downsample.1.bias.txt', f'{directory}/layer{layer}.downsample.1.running_mean.txt', f'{directory}/layer{layer}.downsample.1.running_var.txt', as_ints=as_ints)
    return output
  
  def _layer(inp, n, _downsample=False, first_stride=1, as_ints=False):
    save_inp = inp.copy()
    if _downsample:
      save_inp = downsample(inp, n, as_ints=as_ints)
    output = basic_block(inp, n, first_stride, as_ints=as_ints)
    output = residual(save_inp, output)
    output = ReLU(output)
    return output


  #test = [[[1]*16]*16]*64
  #test = np.array(test) 
  # begin
  output = wrapper_conv_layer(Xtest, f'{directory}/conv1.weight.txt', pad=3, enc_scheme=enc_scheme, stride=2, as_ints=as_ints)
  #print(output.shape)
  output = batch_norm.batch_main(output, f'{directory}/bn1.weight.txt', f'{directory}/bn1.bias.txt', f'{directory}/bn1.running_mean.txt', f'{directory}/bn1.running_var.txt', as_ints=as_ints)
  output = ReLU(output)
  # pad before max pool
  output = conv_layer_prediction.pad_images(output, 1)
  output = max_pooling_layer.max_pooling_layer(output, (3, 3), stride=2)
  # layer 1.0
  output = _layer(output, "1.0", as_ints=as_ints)
  # layer 1.1
  output = _layer(output, "1.1", as_ints=as_ints)
  # layer 2.0
  output = _layer(output, "2.0", _downsample=True, first_stride=2, as_ints=as_ints)
  # layer 2.1
  output = _layer(output, "2.1", as_ints=as_ints)
  # layer 3.0
  output = _layer(output, "3.0", _downsample=True, first_stride=2, as_ints=as_ints)
  # layer 3.1
  output = _layer(output, "3.1", as_ints=as_ints)
  # layer 4.0
  output = _layer(output, "4.0", _downsample=True, first_stride=2, as_ints=as_ints)
  # layer 4.1
  output = _layer(output, "4.1", as_ints=as_ints)
  print()
  # print((output/256))

  # adaptive avg pooling ?
  output = mean_pooling_layer_prediction.adaptive_mean_pooling_layer(output, output_shape=(1, 1))
  # print()
  # print((output/(2**P_2_SCALE))[:50])
  # return output

  filters, width, height = output.shape
  output = output.reshape(width*height*filters)

  dense_kernel = read_weights(f"{directory}/fc.weight.txt")
  dense_bias = read_weights(f"{directory}/fc.bias.txt")
  # output = dense_layer_prediction.dense_layer( output, dense_kernel)
  dense_output = [0] * 1000

  '''
  split = 4000
  for i in range(0, width*height*filters, split):
    temp = dense_layer_prediction.dense_layer( output[i:min(i+split, width*height*filters - i)], dense_kernel[:][i:min(i+split, width*height*filters - i)])
    for j in range(len(temp)):
      dense_output[j] += temp[j]

  for i in range(len(dense_output)):
    dense_output[i] += dense_bias[i]
  '''

  dense_output = dense_layer_prediction.dense_layer( output, dense_kernel, dense_bias)

  if (as_ints):
    dense_output = np.array(dense_output) / 2**P_2_SCALE
    dense_output = np.array(dense_output) / 2**P_2_SCALE
    # dense_output = dense_output.astype(int)
    pass
  output = dense_output

  print(f'Prediction: [', end=' ')
  for pred in output[:100]:
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
  # model = tf.keras.models.load_model(f'./models/{model_name}.h5')
  model = torchvision.models.resnet18()
  model.load_state_dict(torch.load('./models/resnet18.pth'))

  Xtest = np.expand_dims(Xtest, 0)
  Xtest = torch.Tensor(Xtest)
  # print(f'Xtest: {Xtest}')
  # print(f'Xtest shape: {Xtest.shape}')
  # print(f'Ytest: {Ytest}')
  # print(f'Ytest shape: {Ytest.shape}')

  # Ypred = model.predict( Xtest )
  model.eval()

  # test = [[[[1]*16]*16]*64]
  # test = torch.Tensor(test)
  # temp = model.conv1(Xtest)
  # temp = model.bn1(temp)
  # temp = model.relu(temp)
  # temp = model.maxpool(temp)
  # temp = model.layer1(temp)
  # temp = model.layer2(temp)
  # temp = model.layer3(temp)
  # temp = model.layer4(temp)
  # temp = model.avgpool(temp)
  # print(temp[0][:50])
  # return temp

  #temp = model.layer2(temp)
  #temp = model.layer3(temp)
  #temp = model.layer4(temp)

  #output = model.avgpool(temp)
  #print(output.shape)
  #print(output[0])
  #return
  
  #conv_output_image = Ypred.permute(0, 2, 3, 1).detach().numpy()
  # print(conv_output_image)
  # print(output)

  with torch.no_grad():
    Ypred = model(Xtest)
  print(f'Prediction: [', end=' ')
  for pred in Ypred[0][:100]:
    pred = float(pred)
    print(f'{pred:.7f}', end=' ')
  print(']')
  # print(f'Acutal: {Ytest}')
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