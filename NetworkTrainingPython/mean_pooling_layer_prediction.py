import numpy as np
import conv_layer_prediction

def mean_pooling(input_layer, shape=(2,2), stride=1):
  width, height = input_layer.shape
  fw, fh = shape
  # output = np.zeros( (width // fw, height // fh ), dtype=int )
  output = np.zeros( ((width - fw) // stride + 1, (height - fh) // stride + 1 ) )

  i = 0
  j = 0
  for img_i in range(0, height - fh + 1, stride):
    for img_j in range(0, width - fw + 1, stride):
      vals = []
      for k1 in range(fh):
        for k2 in range(fw):
          vals.append(input_layer[img_i + k1][img_j + k2])  
      output[i][j] = np.mean(vals)  
      j += 1 
    j = 0
    i += 1  
  return output.astype(input_layer.dtype)

def mean_pooling_layer(inputs, padding=0, stride=1, shape=(2,2)):
  if padding:
    inputs = conv_layer_prediction.pad_images(inputs, padding)
  channels, *_ = inputs.shape
  output = []
  for image in inputs:
    output.append( mean_pooling(image, shape, stride) )
  return np.array(output)

def adaptive_mean_pooling(input_layer, output_shape=(2,2)):
  width, height = input_layer.shape
  fw, fh = (width - output_shape[0] + 1, height - output_shape[1] + 1)
  output = np.zeros( output_shape )

  i = 0
  j = 0
  for img_i in range(height - fh + 1):
    for img_j in range(width - fw + 1):
      vars = []
      for k1 in range(fh):
        for k2 in range(fw):
          vars.append(input_layer[img_i + k1][img_j + k2])
      output[i][j] = np.mean(vars)
      j += 1 
    j = 0
    i += 1  
  return output

def adaptive_mean_pooling_layer(inputs, output_shape=(2,2)):
  channels, *_ = inputs.shape
  output = []
  for image in inputs:
    output.append( adaptive_mean_pooling(image, output_shape) )
  return np.array(output)

def main():

  arr = np.array([[31, 15, 28, 184], 
                  [0, 100, 70, 38], 
                  [12, 12, 7, 2], 
                  [12, 12, 45, 6]])
  
  mean_arr = mean_pooling(arr)
  print(mean_arr)

if __name__ == '__main__':
  main()