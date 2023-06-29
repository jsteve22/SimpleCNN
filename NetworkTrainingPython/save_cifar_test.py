import numpy as np
from tensorflow import keras
from tensorflow.keras import layers

def main(test_num=None):
  # Model / data parameters
  num_classes = 10

  # Load the data and split it between train and test sets
  (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

  # Scale images to the [0, 1] range
  x_train = x_train.astype('float32') / 255
  x_test  = x_test.astype('float32') / 255
  # Make sure images have shape (28, 28, 1)
  # x_train = np.expand_dims(x_train, -1)
  # x_test  = np.expand_dims(x_test, -1)

  '''
  print('x_train shape:', x_train.shape)
  print(x_train.shape[0], "train samples")
  print(x_test.shape[0], "test samples")
  '''

  # convert class vectors to binary class matrices
  y_train = keras.utils.to_categorical(y_train, num_classes)
  y_test  = keras.utils.to_categorical(y_test, num_classes)

  random_value = 3 if test_num is None else test_num
  import pickle as pkl
  single_x_test = x_test[random_value]
  single_y_test = y_test[random_value]
  with open('cifar_test.pkl', 'wb') as f:
    pkl.dump( (single_x_test, single_y_test), f )

  Xtest = single_x_test
  image_height, image_width, image_channels = Xtest.shape
  images = np.zeros( (image_channels, image_height, image_width) )
  for wi, w in enumerate(Xtest):
    for hi, h in enumerate(w):
      for ci, c in enumerate(h):
        images[ci][wi][hi] = c
  Xtest = images

  '''
  print( Xtest.shape )
  '''

  with open('cifar_image.txt', 'w') as f:
    channels, width, height = Xtest.shape
    f.write(f'{channels} {width} {height} \n')
    for channel in Xtest:
      for row in channel:
        for value in row:
          value = int( value * (2**8) )
          f.write(f'{value} ')
    f.write('\n')

if __name__ == '__main__':
  main()