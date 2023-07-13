import numpy as np


def max_pooling(input_layer, shape=(3,3), stride=1):
  width, height = input_layer.shape
  output = np.zeros( (width // shape[0], height // shape[1] ), dtype=int )
  fw, fh = shape

  for ind in range(0, len(output), stride):
    for jnd in range(0, len(output[ind]), stride):
      vals = []
      for fi in range(fw):
        for fj in range(fh):
          vals.append(input_layer[ind*fw + fi][jnd*fh + fj])
      output[ind][jnd] = max(vals)
  return output

def max_pooling_layer(inputs, shape=(3,3), stride=1):
  channels, *_ = inputs.shape
  output = []
  for image in inputs:
    output.append( max_pooling(image, shape, stride) )
  return np.array(output)

def main():

  arr = np.array([[31, 15, 28, 184], 
                  [0, 100, 70, 38], 
                  [12, 12, 7, 2], 
                  [12, 12, 45, 6]])
  
  max_arr = max_pooling(arr)
  print(max_arr)

if __name__ == '__main__':
  main()