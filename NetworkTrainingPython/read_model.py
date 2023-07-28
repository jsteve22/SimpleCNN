import conv_layer_prediction
import dense_layer_prediction
import dense_layer_poly_mult
import mean_pooling_layer_prediction
import max_pooling_layer
import batch_norm
from write_weights import read_weights
import numpy as np

class Layer:
    def __init__(self, name):
        self.name = name
        self.path = None
        self.activation = None
        self.padding = None
        self.stride = None
        self.bias = None

class Model:
    def __init__(self, name, num_layers):
        self.name = name
        self.num_layers = num_layers
        self.layers = []

def read_model(filename):
    with open(filename) as fp:
        info = fp.readline().split(",")
        model = Model(info[0], info[1]) 
        for line in fp.readlines():
            layer_list = line.split(",")
            layer_list = [x.strip() for x in layer_list]
            if "conv2d" in layer_list[0]:
                conv = Layer(layer_list[0])
                conv.path = layer_list[1]
                if layer_list[2] != "None":     
                    conv.activation = layer_list[2]
                conv.padding = int(layer_list[3])
                conv.stride = int(layer_list[4])
                if layer_list[5] != "_":
                    conv.bias = layer_list[5]
                model.layers.append(conv)
            if "meanpooling" in layer_list[0]:
                meanpooling = Layer(layer_list[0])
                meanpooling.shape = int(layer_list[1])
                meanpooling.padding = int(layer_list[2])
                meanpooling.stride = int(layer_list[3])
                model.layers.append(meanpooling)
            if "dense" in layer_list[0]:
                dense = Layer(layer_list[0])
                dense.path = layer_list[1]
                if layer_list[2] != "None":
                    dense.activation = layer_list[2]
                if layer_list[3] != "_":
                    dense.bias = layer_list[3]
                model.layers.append(dense)
            if "flatten" in layer_list[0]:
                flatten = Layer(layer_list[0])
                model.layers.append(flatten)
    return model

def use_model(filename, test, enc_scheme):
    model = read_model(filename)
    output = test.copy()
    for layer in model.layers:
        if "conv2d" in layer.name:
            output = wrapper_conv_layer( output, layer.path, pad=layer.padding, enc_scheme=enc_scheme )
            if layer.activation == "relu":
               output = ReLU(output)
        if "meanpooling" in layer.name:
           output = mean_pooling_layer_prediction.mean_pooling_layer( output, layer.padding, layer.stride, (layer.shape, layer.shape) )
        if "flatten" in layer.name:
            filters, width, height = output.shape
            output = output.reshape(width*height*filters)
        if "dense" in layer.name:
            dense_kernel = read_weights(layer.path)
            num_vec = len(dense_kernel)
            # output = dense_layer_prediction.dense_layer( output, dense_kernel)
            dense_output = [0] * num_vec

            split = 4000
            for i in range(0, width*height*filters, split):
                temp = dense_layer_prediction.dense_layer( output[i:min(i+split, width*height*filters - i)], dense_kernel[:][i:min(i+split, width*height*filters - i)])
                for j in range(len(temp)):
                    dense_output[j] += temp[j]
            output = dense_output
            # output = np.array(output) / (2**(P_2_SCALE*2))
            output = dense_layer_prediction.softmax( output )
    return output

P_2_SCALE = 8

def wrapper_conv_layer(input_layer, layer_path, pad=0, enc_scheme=None, stride=1):
  layer = input_layer.copy()
  if pad:
    layer = conv_layer_prediction.pad_images( layer, pad )
  output = conv_layer_prediction.conv_layer_prediction( layer, read_weights(layer_path), stride, enc_scheme )
  output = np.array(output)
  #output = scale_down(output, 2**P_2_SCALE)
  layer_name = layer_path.split('/')[-1].split('.')[0]
  print(f'{layer_name} Done')
  return output


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