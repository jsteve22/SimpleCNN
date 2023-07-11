import numpy as np

def normalization(vectors, batch_size, vec_len):
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
            norm_vectors[i][j] = (x - mean_vec[j]) / np.sqrt(std_dev_vec[i]**2)
    return norm_vectors
    

def mini_batch(vectors, beta, gamma):
    output = []
    batch_size = len(vectors)
    vec_len = len(vectors[0])
    norm_vectors = normalization(vectors, batch_size, vec_len)
    for vec in norm_vectors:
        output.append(gamma*vec + beta)


    

