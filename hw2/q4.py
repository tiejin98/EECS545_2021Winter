import numpy as np
import matplotlib.pyplot as plt

def readMatrix(file):
    # Use the code below to read files
    fd = open(file, 'r')
    hdr = fd.readline()
    rows, cols = [int(s) for s in fd.readline().strip().split()]
    tokens = fd.readline().strip().split()
    matrix = np.zeros((rows, cols))
    Y = []
    for i, line in enumerate(fd):
        nums = [int(x) for x in line.strip().split()]
        Y.append(nums[0])
        kv = np.array(nums[1:])
        k = np.cumsum(kv[:-1:2])
        v = kv[1::2]
        matrix[i, k] = v
    return matrix, tokens, np.array(Y)




def nb_train(matrix, category):
    # Implement your algorithm and return
    state = {}
    def estimate_pi_k(x, y, k):
        x_k = x[np.where(y == k)]
        return np.log(x_k.shape[0] / x.shape[0])

    def estimate_log_pkj(x, y, k, a=1):
        res = []
        x_k = x[np.where(y == k)]
        n_k = np.sum(x_k)
        for j in range(x_k.shape[1]):
            n_kj = np.sum(x_k[:, j])
            p_kj = (n_kj + a) / (n_k + a * x_k.shape[1])
            res.append(np.log(p_kj))
        return res
    k_list = [0,1]
    state['pi'] = [estimate_pi_k(matrix,category,k) for k in k_list]
    state['pkj'] = [estimate_log_pkj(matrix,category,k) for k in k_list]
    return state

def nb_test(matrix, state):
    # Classify each email in the test set (each row of the document matrix) as 1 for SPAM and 0 for NON-SPAM
    pi = state['pi']
    pkj = state['pkj']
    temp_res = []
    for i in range(len(pi)):
        temp_res.append(np.dot(matrix,np.array(pkj[i]).reshape(-1,1))+pi[i])
    temp_res = np.array(temp_res)
    return np.argmax(temp_res,axis=0).reshape(-1,)

def evaluate(output, label):
    # Use the code below to obtain the accuracy of your algorithm
    error = (output != label).sum() * 1. / len(output)
    print('Error: {:2.4f}%'.format(100*error))
    return error

def main():
    # Note1: tokenlists (list of all tokens) from MATRIX.TRAIN and MATRIX.TEST are identical
    # Note2: Spam emails are denoted as class 1, and non-spam ones as class 0.
    # Note3: The shape of the data matrix (document matrix): (number of emails) by (number of tokens)

    # Load files
    dataMatrix_train, tokenlist, category_train = readMatrix('q4_data/MATRIX.TRAIN')
    dataMatrix_test, tokenlist, category_test = readMatrix('q4_data/MATRIX.TEST')
    # Train
    state = nb_train(dataMatrix_train, category_train)

    # Test and evluate
    prediction = nb_test(dataMatrix_test, state)
    evaluate(prediction, category_test)

    pkj = state['pkj']
    minus = (np.array(pkj[1]) - np.array(pkj[0]))
    tokens_index = np.argsort(minus)[::-1][:5]
    token = [tokenlist[i] for i in tokens_index]
    print("5 tokens that are most indicative of the SPAM are:{}".format(token))

    train_size = [50,100,200,400,800,1400]
    train_size.append(2144)
    res = []
    for size in train_size:
        print("training size:{}".format(size))
        if size == 2144:
            state = nb_train(dataMatrix_train, category_train)
            prediction = nb_test(dataMatrix_test, state)
            res.append(evaluate(prediction, category_test))
        else:
            dataMatrix_train_size,tokenlist,category_train_size = readMatrix('q4_data/MATRIX.TRAIN.{}'.format(size))
            state = nb_train(dataMatrix_train_size, category_train_size)
            prediction = nb_test(dataMatrix_test, state)
            res.append(evaluate(prediction, category_test))
    plt.plot(train_size,res)
    plt.xlabel("size of training set")
    plt.ylabel("error")
    plt.show()


if __name__ == "__main__":
    main()
        
