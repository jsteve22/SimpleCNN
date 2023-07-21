import numpy as np


def mean_pooling(input_layer, shape=(2,2)):
  width, height = input_layer.shape
  output = np.zeros( (width // shape[0], height // shape[1] ), dtype=input_layer.dtype )
  fw, fh = shape

  for ind, row in enumerate(output):
    for jnd, _ in enumerate(row):
      total = 0
      for fi in range(fw):
        for fj in range(fh):
          total += input_layer[ind + fi][jnd + fj]
      output[ind][jnd] = total / (fw*fh)
  return output.astype(input_layer.dtype)

def mean_pooling_layer(inputs, shape=(2,2)):
  channels, *_ = inputs.shape
  output = []
  for image in inputs:
    output.append( mean_pooling(image, shape) )
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