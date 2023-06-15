import dense_layer_prediction
import numpy as np

def make_matrix():
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

def make_input():
    inpt = []

    for i in range(25):
        inpt.append(1)

    return inpt

def gazelle():
    matrix = make_matrix()
    inpt = make_input()

    bias = [0] * 25
    output = dense_layer_prediction.dense_layer(inpt, matrix, bias)

    return output

output = gazelle()
print(output)



        
            