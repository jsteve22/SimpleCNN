import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
from dense_layer_prediction import transform_dense_kernel

def write_weights(model_name, image_shape):
    model = tf.keras.models.load_model(f'{model_name}.h5')
    weights_paths = model.get_weight_paths()

    dense_kernel_h, dense_kernel_w, dense_kernel_f = 0, 0, 0
    image_height, image_width, image_channels, image_filters = image_shape

    for key in weights_paths:
        layer = weights_paths[key]
        fp = open(f"{key}.txt", "w")
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
            fp.write(f"{filters} {channels} {height} {width}\n")
            total_sz = filters * channels * height * width
        elif "dense" in key and "kernel" in key:
            numpy_layer = transform_dense_kernel((dense_kernel_f, dense_kernel_w, dense_kernel_h), numpy_layer)
            channels, vectors = numpy_layer.shape
            fp.write(f"{channels} {vectors}\n")
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
            fp.write(f"{i} ")

def read_weights(file_name):
    with open(file_name, "r") as fp:
        line = fp.readline().rstrip()
        dims = list(map(int, line.split(" ")))
        nums = fp.readline().rstrip()
        nums = list(map(float, nums.split(" ")))
        nums = np.array(nums)
        nums = nums.reshape(dims)
        #print(nums)
    return nums

write_weights("small_model", (28, 28, 1, 1))
#read_weights("dense.kernel.txt")
