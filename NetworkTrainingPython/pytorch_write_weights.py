import torch
import torchvision.models as models
import numpy as np
from dense_layer_prediction import transform_dense_kernel
import os

def write_weights(model, model_name, write_ints=True):
    model_name = model_name

    try:
        os.mkdir(f'./model_weights/{model_name}')
    except FileExistsError:
        pass

    weights_paths = model.state_dict()

    for layer_name in weights_paths:
        layer = weights_paths[layer_name]
        fp = open(f"./model_weights/{model_name}/{layer_name}.txt", "w")
        # fp = open(f"{layer_name}.txt", "w")
        numpy_layer = layer.numpy()
        if "conv" in layer_name and "weight" in layer_name:
            write_conv_kernel(fp, numpy_layer, layer_name, model_name, write_ints)
            continue
        elif "dense" in layer_name and "weight" in layer_name:
            write_dense_kernel(fp, numpy_layer, layer_name, model_layer_index_dict, model.layers, write_ints)
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

def write_dense_kernel(filepointer, numpy_layer, name, model_layer_index, model_layers, write_ints=True):

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

def main():
    model = models.resnet18()
    model.load_state_dict(torch.load('./models/resnet18.pth'))
    write_weights(model, 'resnet18', False) 

if __name__ == '__main__':
    main()