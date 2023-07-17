import numpy as np
from write_weights import read_weights

def training_normalization(vectors, batch_size, vec_len, epsil=pow(np.e,-8)):
    mean_vec = []
    std_dev_vec = []
    norm_vectors = []
    for i in range(vec_len):
        temp = []
        for vec in vectors:
            temp.append(vec[i])
        mean_vec.append(np.mean(temp))
        std_dev_vec.append(np.std(temp))
    for i, vec in enumerate(vectors):
        for j, x in enumerate(vec):
            norm_vectors[i][j] = (x - mean_vec[j]) / np.sqrt(std_dev_vec[i]**2 + epsil)
    return norm_vectors

def calc_moving_average(alpha, ma_mean, ma_var, mean_vec, std_dev_vec):
    for i, u in enumerate(ma_mean):
        u = u*alpha + (1 - alpha)*mean_vec[i]
    for i, o in enumerate(ma_var):
        o = o*alpha + (1 - alpha)*std_dev_vec[i]
    return ma_mean, ma_var

def mini_batch(vectors, beta, gamma, ma_mean, ma_var, alpha):
    output = []
    batch_size = len(vectors)
    vec_len = len(vectors[0])
    norm_vectors = training_normalization(vectors, batch_size, vec_len)
    for vec in norm_vectors:
        output.append(gamma*vec + beta)
    
def batch_normalize(vectors, ma_mean, ma_var, e):
    normalized = vectors
    for i, vec in enumerate(vectors):
        for j, x in enumerate(vec):
            for k, y in enumerate(x):
                normalized[i][j][k] = (y - ma_mean[i]) / np.sqrt(ma_var[i] + e)
    return normalized

def batch_inference(vectors, gamma, beta, ma_mean, ma_var, e):
    normalized = batch_normalize(vectors, ma_mean, ma_var, e)
    batch_output = np.zeros(vectors.shape)
    for i, vec in enumerate(normalized):
        for j, x in enumerate(vec):
            for k, y in enumerate(x):
                batch_output[i][j][k] = (gamma[i]*y + beta[i])
    return batch_output

def read_batch_weights(weight_file, bias_file, mean_file, var_file):
    gamma = read_weights(weight_file)
    beta = read_weights(bias_file)
    ma_mean = read_weights(mean_file)
    ma_var = read_weights(var_file)
    return gamma, beta, ma_mean, ma_var

def batch_main(batch_input, weight_file="bn1.weight.txt", bias_file="bn1.bias.txt", mean_file="bn1.running_mean.txt", var_file="bn1.running_var.txt", e=1**(-5)):
    gamma, beta, ma_mean, ma_var = read_batch_weights(weight_file, bias_file, mean_file, var_file)
    output = batch_inference(batch_input, gamma, beta, ma_mean, ma_var, e)
    return output