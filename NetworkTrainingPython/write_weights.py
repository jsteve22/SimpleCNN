import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
from dense_layer_prediction import transform_dense_kernel
import os

def write_weights(model_name, image_shape, write_ints=True):
    model = tf.keras.models.load_model(f'models/{model_name}.h5')
    model_name = 'test_' + model_name

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
            write_conv_kernel(fp, numpy_layer, key, model_name, write_ints)
            continue
        elif "dense" in key and "kernel" in key:
            write_dense_kernel(fp, numpy_layer, key, model_layer_index_dict, model.layers, write_ints)
            continue
        else: 
            sz = ""
            total_sz = 1
            for val in numpy_layer.shape:
                sz += f'{val} '
                total_sz *= val
            fp.write(f"{sz}\n")

        numpy_layer = numpy_layer.reshape(total_sz)
        for i in numpy_layer:
            if (write_ints):
                fp.write(f"{int(i * (2**8))} ")
            else:
                fp.write(f"{i} ")
        fp.close()

def write_conv_kernel(filepointer, numpy_layer, name, model_name, write_ints=True, limit=32):
    width, height, channels, filters = numpy_layer.shape
    kernel = np.zeros( (filters, channels, width, height) )
    # convert the kernel into proper format
    for wi, w in enumerate(numpy_layer):
        for hi, h in enumerate(w):
            for ci, c in enumerate(h):
                for fi, f in enumerate(c):
                    kernel[fi][ci][wi][hi] = f 
    numpy_layer = kernel
    # write the dimensions of the shape
    new_name = '0.'.join( name.split('.') )
    if channels > limit:
        filepointer.close()
        many_conv_kernel(numpy_layer, name, model_name, write_ints, limit)
        return

    filepointer.write(f"{filters} {channels} {height} {width} \n")
    total_sz = filters * channels * height * width

    # write the weights to the file
    write_layer = numpy_layer.reshape(total_sz)
    for i in write_layer:
        if (write_ints):
            filepointer.write(f"{int(i * (2**8))} ")
        else:
            filepointer.write(f"{i} ")
    filepointer.close()
    return

def many_conv_kernel(numpy_layer, name, model_name, write_ints, limit):
    filters, all_channels, width, height = numpy_layer.shape
    for ind in range(0, all_channels, limit):
        if (ind == 0):
            new_name = name
        else:
            new_name = f'.{1-(ind//limit)}.'
            new_name = new_name.join( name.split('.') )
        filepointer = open(f'./model_weights/{model_name}/{new_name}.txt', 'w')

        channels = min(limit, all_channels-ind)
        filepointer.write(f"{filters} {channels} {height} {width} \n")
        total_sz = filters * channels * height * width
        write_layer = np.zeros( (filters, channels, width, height) )
        for fi, filt in enumerate(numpy_layer):
            for ci, chan in enumerate(filt[ind:ind+channels]):
                for wi, w in enumerate(chan):
                    for hi, val in enumerate(w):
                        write_layer[fi][ci][wi][hi] = val

        # write the weights to the file
        write_layer = write_layer.reshape(total_sz)
        for i in write_layer:
            if (write_ints):
                filepointer.write(f"{int(i * (2**8))} ")
            else:
                filepointer.write(f"{i} ")
        filepointer.close()
    return

def write_dense_kernel(filepointer, numpy_layer, name, model_layer_index, model_layers, write_ints=True):
    dense_index = model_layer_index[name.split('.')[0]]
    layer = model_layers[dense_index]
    index_shift = 0
    while 'conv' not in model_layers[dense_index - index_shift].name:
        index_shift += 1
    
    input_shape = model_layers[dense_index - index_shift].output.shape

    *_, dense_kernel_w, dense_kernel_h, dense_kernel_f = input_shape

    numpy_layer = transform_dense_kernel((dense_kernel_f, dense_kernel_w, dense_kernel_h), numpy_layer)
    channels, vectors = numpy_layer.shape
    filepointer.write(f"{channels} {vectors} \n")
    total_sz = channels * vectors

    # write the weights to the file
    numpy_layer = numpy_layer.reshape(total_sz)
    for i in numpy_layer:
        if (write_ints):
            filepointer.write(f"{int(i * (2**8))} ")
        else:
            filepointer.write(f"{i} ")
    filepointer.close()
    return

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
    # write_weights("mnist_email_model", (28, 28, 1, 1))
    # write_weights("simple_model", (28, 28, 4, 1))
    #read_weights("dense.kernel.txt")
