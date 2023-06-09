import numpy as np
import pickle as pkl
import tensorflow as tf

def load_pickle(filename):
  with open(f'{filename}', 'rb') as f:
    return pkl.load(f)


single_test = load_pickle('single_test.pkl')

# print(single_test)

model_name = 'small_model'
model = tf.keras.models.load_model(f'{model_name}.h5')

Xtest = single_test[0]
Ytest = single_test[1]

'''
Xtest = Xtest.astype('float32') / 255
Xtest = np.expand_dims(Xtest, -1)
'''

Xtest = np.expand_dims(Xtest, 0)
print(f'Xtest: {Xtest}')
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