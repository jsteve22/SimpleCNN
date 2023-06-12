import numpy as np
import pickle as pkl
import tensorflow as tf
import conv_layer_prediction
import dense_layer_prediction

def load_pickle(filename):
  with open(f'{filename}', 'rb') as f:
    return pkl.load(f)

def main():
  model_name = 'small_model'
  tf_test(model_name)
  print()
  print()
  print()
  custom_test(model_name)
  return

def custom_test(model_name='small_model'):
  single_test = load_pickle('single_test.pkl')

  Xtest = single_test[0]
  Ytest = single_test[1]


  Xtest = Xtest.reshape( (28,28) )

  # model_name = 'conv_model'
  model = tf.keras.models.load_model(f'{model_name}.h5')
  weights_dictionary = model.get_weight_paths()
  # print(weights_dictionary.keys())

  conv2d_kernel = weights_dictionary['conv2d.kernel'].numpy()
  conv2d_bias   = weights_dictionary['conv2d.bias'].numpy()

  # print(conv2d_kernel.shape)
  # print(conv2d_kernel.T.shape)

  conv2d_kernel = conv2d_kernel.T
  filters, channels, width, height = conv2d_kernel.shape
  conv2d_kernel = conv2d_kernel.reshape( (filters, width, height) )
  

  output = conv_layer_prediction.conv_layer_prediction( Xtest, conv2d_kernel )
  output = np.array(output)

  filters, width, height = output.shape
  for i in range(filters):
    for j in range(width):
      for k in range(height):
        output[i][j][k] += conv2d_bias[i]
        if output[i][j][k] < 0:
          output[i][j][k] = 0

  output = output.T
  ## output = output.reshape((26*26*4))
  output = output.reshape(26*26*filters)

  output = dense_layer_prediction.wrapper_dense_layer( output, 'dense', weights_dictionary )
  output = dense_layer_prediction.softmax( output )
  # print( output )
  print(f'Prediction: [', end=' ')
  for pred in output:
    pred = float(pred)
    print(f'{pred:.7f}', end=' ')
  print(']')

def tf_test(model_name='small_model'):
  single_test = load_pickle('single_test.pkl')
  # model_name = 'small_model'
  model = tf.keras.models.load_model(f'{model_name}.h5')

  Xtest = single_test[0]
  Ytest = single_test[1]

  Xtest = np.expand_dims(Xtest, 0)
  # print(f'Xtest: {Xtest}')
  print(f'Xtest shape: {Xtest.shape}')
  print(f'Ytest: {Ytest}')
  print(f'Ytest shape: {Ytest.shape}')

  Ypred = model.predict( Xtest )

  print(f'Prediction: [', end=' ')
  for pred in Ypred[0]:
    pred = float(pred)
    print(f'{pred:.7f}', end=' ')
  print(']')
  print(f'Acutal: {single_test[1]}')

if __name__ == '__main__':
  main()