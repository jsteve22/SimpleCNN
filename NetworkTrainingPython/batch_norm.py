import numpy as np

def training_normalization(vectors, batch_size, vec_len, epsil=1e^-8):
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
    
def batch_normalize(vectors, ma_mean, ma_var):
    normalized = vectors
    for i, vec in enumerate(vectors):
        for j, x in enumerate(vec):
            normalized[i][j] = (x - ma_mean[j]) / ma_var[j]
    return normalized

def batch_inference(vectors, gamma, beta, ma_mean, ma_var):
    batch_output = []
    for i, vec in enumerate(vectors):
        batch_output.append([])
        for j, x in enumerate(vec):
            batch_output[i].append(gamma*x + beta)


    

