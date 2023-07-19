import numpy as np


def max_pooling(input_layer, shape=(3,3), stride=1):
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
      output[i][j] = max(vals)  
      j += 1 
    j = 0
    i += 1  
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