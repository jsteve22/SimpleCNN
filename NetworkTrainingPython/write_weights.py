#!/usr/bin/env python3
# Filename: write_weights.py
# Date: 7/28/2023
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
from dense_layer_prediction import transform_dense_kernel
import os

# This file reads in the contents of a tensorflow model and 
# writes the weights of the model into text files in a new directory

CONV2D = tf.keras.layers.Conv2D
DENSE = tf.keras.layers.Dense
AVG2D = tf.keras.layers.AveragePooling2D
FLATTEN = tf.keras.layers.Flatten
NO_ACTIVATION = 'none'

def write_weights(model_name, image_shape, write_ints=True):
    """
    this function uses a model_name to load a tensorflow model and then 
    write the weights of each layer into a text file to be used with Gazelle.
    The weights can be scaled up to integers or written as floats

    ----- Arguments -----
    model_name  - the name of the tensorflow model being loaded 
    write_ints  - boolean value to determine if the model should scale to integers

    ----- Output -----
    weights of the model will be written to text files in a new directory
    directory path => ./model_weights/{model_name}/
    """

    model = tf.keras.models.load_model(f'models/{model_name}.h5')
    model_name = model_name

    try:
        os.mkdir(f'./model_weights/{model_name}')
    except FileExistsError:
        pass

    weights_paths = model.get_weight_paths()
    model_layer_dict = { ml.name: ml for ml in model.layers }
    model_layer_index_dict = { ml.name: i for i, ml in enumerate(model.layers) }

    dense_kernel_h, dense_kernel_w, dense_kernel_f = 0, 0, 0
    image_height, image_width, image_channels, image_filters = image_shape


    with open(f'./model_weights/{model_name}/summary.txt', 'w') as fw:
        fw.write(f'{model_name}, {len(model.layers)}\n')
        # print(write_conv_specs(model.layers[0], model_name))
        for layer in model.layers:
            if type(layer) == CONV2D:
                fw.write(f'{write_conv_specs(layer, model_name)}\n')
            if type(layer) == DENSE:
                fw.write(f'{write_dense_specs(layer, model_name)}\n')
            if type(layer) == AVG2D:
                fw.write(f'{write_meanpooling_specs(layer)}\n')
            if type(layer) == FLATTEN:
                fw.write(f'{write_flatten_specs(layer)}\n')


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

def write_conv_kernel(filepointer, numpy_layer, write_ints=True):
    """
    this function will transform the weights of a 2d convolutional layer and
    write the results into the given filepointer

    ----- Arguments -----
    filepointer - the file pointer to write the weights to
    numpy_layer - the numpy array containing the weights of conv layer
    write_ints  - boolean value to determine if the model should scale to integers

    ----- Output -----
    convolutional layer weights written to specified filepointer
    """

    width, height, channels, filters = numpy_layer.shape
    kernel = np.zeros( (filters, channels, width, height) )
    # convert the kernel into proper format
    for wi, w in enumerate(numpy_layer):
        for hi, h in enumerate(w):
            for ci, c in enumerate(h):
                for fi, f in enumerate(c):
                    kernel[fi][ci][wi][hi] = f 
    numpy_layer = kernel


    ''' # depracated code
    new_name = '0.'.join( name.split('.') )
    if channels > limit and False:
        filepointer.close()
        many_conv_kernel(numpy_layer, name, model_name, write_ints, limit)
        return
    '''

    # write the dimensions of the shape
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
    """
    this function will transform the weights of a dense layer and write the results 
    into the given filepointer

    ----- Arguments -----
    filepointer         - the file pointer to write the weights to
    numpy_layer         - the numpy array containing the weights of dense layer
    write_ints          - boolean value to determine if the model should scale to integers
    name                - name of the current dense layer
    model_layer_index   - dictionary containing index of each layer within the model (1st layer -> index: 0)
    model_layers        - array containing list of layers of the model

    ----- Output -----
    dense layer weights written to specified filepointer
    """

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

def write_conv_specs(conv_layer, model_name):
    """
    this function returns a string in the format of convolutional layer parameters
    format of line = "layer_name, path_to_weights, activation_function, padding, strides"

    ----- Arguments -----
    conv_layer  - the conv layer we are converter to specific format
    model_name  - the name of the model we are writing the spec for

    ----- Output -----
    formatted string for convolutional layer
    """

    # format of line = "layer_name, path_to_weights, activation_function, padding, strides"
    ret_string = f'conv2d, ./model_weights/{model_name}/{conv_layer.name}.kernel.txt'

    # get activation function
    activation = conv_layer.activation
    if activation == tf.keras.activations.relu:
        ret_string += ', relu'
    else:
        ret_string += ', {NO_ACTIVATION}'

    # get padding and strides
    padding = conv_layer.padding
    strides = conv_layer.strides
    if padding == 'valid':
        ret_string += ', 0'
    elif padding == 'same':
        assert strides == (1, 1) # only works with strides of 1 right now
        pad_num = int(conv_layer.kernel.shape[0] / 2)
        ret_string += f', {pad_num}'

    # add strides
    ret_string += f', {strides[0]}'

    if type(conv_layer.bias) != type(None):
        ret_string += f', ./model_weights/{model_name}/{conv_layer.name}.bias.txt'
    else:
        ret_string += ', _'

    return ret_string

def write_dense_specs(dense_layer, model_name):
    """
    this function returns a string in the format of dense layer parameters
    format of line = "layer_name, path_to_weights, activation_function, bias_path"

    ----- Arguments -----
    dense_layer - the dense layer we are converter to specific format
    model_name  - the name of the model we are writing the spec for

    ----- Output -----
    formatted string for dense layer
    """

    # format of line = "layer_name, path_to_weights, activation_function, bias_path"
    ret_string = f'dense, ./model_weights/{model_name}/{dense_layer.name}.kernel.txt'

    # get activation function
    activation = dense_layer.activation
    if activation == tf.keras.activations.softmax:
        ret_string += ', softmax'
    else:
        ret_string += ', {NO_ACTIVATION}'

    if type(dense_layer.bias) != type(None):
        ret_string += f', ./model_weights/{model_name}/{dense_layer.name}.bias.txt'
    else:
        ret_string += ', _'

    return ret_string

def write_flatten_specs(flatten_layer):
    """
    this function returns a string in the format of flatten layer
    format of line = "layer_name"

    ----- Arguments -----
    flatten_layer - a flatten layer

    ----- Output -----
    formatted string for flatten layer
    """

    # format of line = "layer_name"
    ret_string = f'flatten'

    return ret_string

def write_meanpooling_specs(meanpooling_layer):
    """
    this function returns a string in the format of meanpooling layer
    format of line = "layer_name, shape, stride"

    ----- Arguments -----
    meanpooling_layer   - a meanpooling layer to create specified format

    ----- Output -----
    formatted string for meanpooling layer
    """

    # format of line = "layer_name, shape, stride"
    ret_string = f'meanpooling'

    pool_size = meanpooling_layer.pool_size
    strides = meanpooling_layer.strides

    # assert square strides and shape
    assert pool_size[0] == pool_size[1]
    assert strides[0] == strides[1]

    # add pool size 
    ret_string += f', {pool_size[0]}'

    # add padding
    padding = meanpooling_layer.padding
    if padding == 'valid':
        ret_string += ', 0'
    elif padding == 'same':
        assert strides == (1, 1) # only works with strides of 1 right now
        pad_num = int(meanpooling_layer.pool_size[0] / 2)
        ret_string += f', {pad_num}'

    # add stride
    ret_string += f', {strides[0]}'

    return ret_string

def read_weights(file_name):
    """
    this function will read in the contents of a file and return a numpy array
    File formatted with the first line containing the dimensions of array
    and the second line containing all of the contents of array e.g.
    ----- example.txt -----
    d0 d1 d2
    x0 x1 x2 x3 x4 ... xn
    -----------------------

    ----- Arguments -----
    file_name   - the name of the file which holds the numpy array

    ----- Output -----
    nums    - numpy array with specified dimenions and values
    """

    with open(file_name, "r") as fp:
        line = fp.readline().rstrip()
        dims = list(map(int, line.split(" ")))
        nums = fp.readline().rstrip()
        # nums = list(map(int, nums.split(" ")))
        nums = list(map(float, nums.split(" ")))
        nums = np.array(nums)
        nums = nums.reshape(dims)
        #print(nums)
    return nums


if __name__ == '__main__':
    write_weights("miniONN_cifar_model", (32, 32, 3, 1), True)
    # write_weights("mnist_email_model", (28, 28, 1, 1))
    # print()
    # write_weights("simple_model", (28, 28, 4, 1))
    #read_weights("dense.kernel.txt")
