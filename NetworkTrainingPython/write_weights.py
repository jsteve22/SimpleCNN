import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
from dense_layer_prediction import transform_dense_kernel
import os

def write_weights(model_name, image_shape):
    model = tf.keras.models.load_model(f'models/{model_name}.h5')

    try:
        os.mkdir(f'./model_weights/{model_name}')
    except FileExistsError:
        pass

    weights_paths = model.get_weight_paths()
    model_layer_dict = { ml.name: ml for ml in model.layers }
    model_layer_index_dict = { ml.name: i for i, ml in enumerate(model.layers) }

    dense_kernel_h, dense_kernel_w, dense_kernel_f = 0, 0, 0
    image_height, image_width, image_channels, image_filters = image_shape

    for key in weights_paths:
        layer = weights_paths[key]
        fp = open(f"./model_weights/{model_name}/{key}.txt", "w")
        # fp = open(f"{key}.txt", "w")
        numpy_layer = layer.numpy()
        if "conv" in key and "kernel" in key:
            width, height, channels, filters = numpy_layer.shape
            dense_kernel_h = image_height - height + 1
            dense_kernel_f = filters 
            dense_kernel_w = image_width - width + 1
            kernel = np.zeros( (filters, channels, width, height) )
            for wi, w in enumerate(numpy_layer):
                for hi, h in enumerate(w):
                    for ci, c in enumerate(h):
                        for fi, f in enumerate(c):
                            kernel[fi][ci][wi][hi] = f 
            numpy_layer = kernel
            fp.write(f"{filters} {channels} {height} {width} \n")
            total_sz = filters * channels * height * width
        elif "dense" in key and "kernel" in key:
            dense_index = model_layer_index_dict[key.split('.')[0]]
            layer = model.layers[dense_index]
            index_shift = 0
            while 'conv' not in model.layers[dense_index - index_shift].name:
                index_shift += 1
            
            input_shape = model.layers[dense_index - index_shift].output.shape

            *_, dense_kernel_w, dense_kernel_h, dense_kernel_f = input_shape

            numpy_layer = transform_dense_kernel((dense_kernel_f, dense_kernel_w, dense_kernel_h), numpy_layer)
            channels, vectors = numpy_layer.shape
            fp.write(f"{channels} {vectors} \n")
            total_sz = channels * vectors
        else: 
            sz = ""
            total_sz = 1
            for val in numpy_layer.shape:
                sz += f'{val} '
                total_sz *= val
            fp.write(f"{sz}\n")

        numpy_layer = numpy_layer.reshape(total_sz)
        for i in numpy_layer:
            # fp.write(f"{i} ")
            fp.write(f"{int(i * (2**8))} ")

def read_weights(file_name):
    with open(file_name, "r") as fp:
        line = fp.readline().rstrip()
        dims = list(map(int, line.split(" ")))
        nums = fp.readline().rstrip()
        nums = list(map(int, nums.split(" ")))
        nums = np.array(nums)
        nums = nums.reshape(dims)
        #print(nums)
    return nums


if __name__ == '__main__':
    write_weights("miniONN_cifar_model", (32, 32, 3, 1))
    # write_weights("4_layer_mnist_model", (28, 28, 4, 1))
    # write_weights("simple_model", (28, 28, 4, 1))
    #read_weights("dense.kernel.txt")
