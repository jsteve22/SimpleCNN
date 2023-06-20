import dense_layer_prediction
import conv_layer_prediction
import numpy as np

def read_weights(file_name):
    with open(file_name, "r") as fp:
        line = fp.readline().rstrip()
        dims = list(map(int, line.split(" ")))
        nums = fp.readline().rstrip()
        nums = list(map(int, nums.split(" ")))
        #for i in range(len(nums)):
        #    nums[i] = int(nums[i] % 2**64)
        test_nums = []
        for i in range(dims[0]):
            k = 0
            for j in range(dims[1]):
                test_nums.append(nums[2704*i + k])
                k+=1
        test_nums = np.array(test_nums)
        test_nums = test_nums.reshape(dims)
        size = dims[1]
        # nums = np.array(nums)
        # nums = nums.reshape(dims)
        #print(nums)
    return test_nums, size

def make_fc_matrix():
    matrix = []
    ct = 0

    while ct < 250:
        vector = []
        for i in range(25):
            vector.append(ct)
            ct += 1
        matrix.append(vector)

    matrix = np.array(matrix)

    return matrix

def make_fc_input(size):
    inpt = []

    for i in range(size):
        inpt.append(1)

    return inpt

def test_dense_layer():
    # matrix = make_fc_matrix()
    matrix, size = read_weights("dense.kernel.txt")
    print(matrix)
    inpt = make_fc_input(size)

    bias = [0] * 25
    output = dense_layer_prediction.dense_layer(inpt, matrix, bias)

    return output

def make_conv_image():
    image = []
    ct = 0
    num_vectors = 0

    for i in range(2):
        chan = []
        while ct < 17 and num_vectors < 5:
            vector = []
            for j in range(5):
                vector.append(ct)
                ct += 1
            ct -= 2
            chan.append(vector)
            num_vectors += 1
        image.append(chan)
        ct = 0
        num_vectors = 0

    image = np.array(image)
    return image

def make_conv_image_2():
    image = []
    ct = 0
    num_vectors = 0

    for i in range(2):
        chan = []
        while ct < 25:
            vector = []
            for j in range(5):
                vector.append(ct)
                ct += 1
            chan.append(vector)
        image.append(chan)
        ct = 0
    
    print(image)
    return image

def make_conv_filter():
    dim = 3
    ftr = [[[[1] * dim] * dim],[[[1] * dim] * dim]]
    ftr = np.array(ftr)

    return ftr

def test_conv_layer():
    # ftr = make_conv_filter()
    ftr = read_weights("conv2d.kernel.txt")
    print(ftr)

    image = make_conv_image_2()

    output = conv_layer_prediction.conv_layer_prediction(image, ftr)

    return output

# output = test_conv_layer()
output = test_dense_layer()
print(output)





        
            