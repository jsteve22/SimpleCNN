import numpy as np
import pickle as pkl
import tensorflow as tf
import conv_layer_prediction
import dense_layer_prediction

def load_pickle(filename):
  with open(f'{filename}', 'rb') as f:
    return pkl.load(f)

def main():
  # tf_test()
  custom_test()
  return

def custom_test():
  single_test = load_pickle('single_test.pkl')


def tf_test():
  single_test = load_pickle('single_test.pkl')
  model_name = 'small_model'
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
    print(f'{pred:.5f}', end=' ')
  print(']')
  print(f'Acutal: {single_test[1]}')

if __name__ == '__main__':
  main()