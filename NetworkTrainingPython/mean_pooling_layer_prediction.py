import numpy as np


def mean_pooling(input_layer, shape=(2,2)):
  width, height = input_layer.shape
  output = np.zeros( (width // shape[0], height // shape[1] ), dtype=int )
  fw, fh = shape

  for ind, row in enumerate(output):
    for jnd, _ in enumerate(row):
      total = 0
      for fi in range(fw):
        for fj in range(fh):
          total += input_layer[ind*fw + fi][jnd*fh + fj]
      output[ind][jnd] = total // (fw*fh)
  return output

def mean_pooling_layer(inputs, shape=(2,2)):
  channels, *_ = inputs.shape
  output = []
  for image in inputs:
    output.append( mean_pooling(image, shape) )
  return np.array(output)

def adaptive_mean_pooling(input_layer, output_shape=(2,2)):
  width, height = input_layer.shape
  shape = (width // output_shape[0], height // output_shape[1])
  output = np.zeros( output_shape, dtype=int )
  fw, fh = shape

  for ind, row in enumerate(output):
    for jnd, _ in enumerate(row):
      total = 0
      for fi in range(fw):
        for fj in range(fh):
          total += input_layer[ind*fw + fi][jnd*fh + fj]
      output[ind][jnd] = total // (fw*fh)
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