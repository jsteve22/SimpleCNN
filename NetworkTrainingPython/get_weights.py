import tensorflow as tf
from tensorflow.keras.models import load_model

model_name = 'simple_model'

model = load_model(f'./{model_name}.h5')

weights_paths = model.get_weight_paths()

print( weights_paths.keys() )
print()

'''
conv2d = weights_paths['conv2d.kernel']

conv2d_weights = conv2d.numpy()

print(f'first conv2d shape: {conv2d_weights.shape}')
'''

for key in weights_paths.keys():
  layer = weights_paths[key]
  numpy_layer = layer.numpy()
  print(f'{key} shape: {numpy_layer.shape}')
  print(f'{numpy_layer}')
  print()