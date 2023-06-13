import tensorflow as tf
from tensorflow.keras.models import load_model

def write_weights(model_name):
    model = tf.keras.models.load_model(f'{model_name}.h5')
    weights_paths = model.get_weight_paths()

    f = open(f"{model_name}_weights.txt", "w")

    for key in weights_paths:
        layer = weights_paths[key]
        numpy_layer = layer.numpy()
        f.write(f'{key} shape: {numpy_layer.shape}')
        f.write(f'{numpy_layer}')
        f.write(",")

