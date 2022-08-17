# EECS 545 HW3 Q5

# Install scikit-learn package if necessary:
# pip install -U scikit-learn

import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import LinearSVC

np.random.seed(545)

def readMatrix(filename: str):
    # Use the code below to read files
    with open(filename, 'r') as fd:
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


def evaluate(output, label) -> float:
    # Use the code below to obtain the accuracy of your algorithm
    error = float((output != label).sum()) * 1. / len(output)
    print('Error: {:2.4f}%'.format(100 * error))

    return error


def main():
    # Load files
    # Note 1: tokenlists (list of all tokens) from MATRIX.TRAIN and MATRIX.TEST are identical
    # Note 2: Spam emails are denoted as class 1, and non-spam ones as class 0.
    dataMatrix_train, tokenlist, category_train = readMatrix('q5_data/MATRIX.TRAIN')
    dataMatrix_test, tokenlist, category_test = readMatrix('q5_data/MATRIX.TEST')

    # Train
    model = LinearSVC()
    model.fit(dataMatrix_train,category_train)

    # Test and evluate
    prediction = model.predict(dataMatrix_test)
    evaluate(prediction, category_test)

    ## part b code here
    train_size = [50, 100, 200, 400, 800, 1400]
    res = []
    for size in train_size:
        print("training size:{}".format(size))
        dataMatrix_train_size, tokenlist, category_train_size = readMatrix('q5_data/MATRIX.TRAIN.{}'.format(size))
        model = LinearSVC()
        model.fit(dataMatrix_train_size,category_train_size)
        scores = model.decision_function(dataMatrix_train_size)
        print("Number of support vector:{}".format(len(scores[abs(scores)<=1])))
        prediction = model.predict(dataMatrix_test)
        res.append(evaluate(prediction, category_test))
    plt.plot(train_size,res)
    plt.xlabel("size of training set")
    plt.ylabel("error")
    plt.show()

    ## part c
    res_na = [0.03875, 0.02625, 0.02625, 0.01875, 0.0175, 0.01625]
    plt.plot(train_size,res,color="blue",label = "linearsvm")
    plt.plot(train_size,res_na,color="red",label = "bayes")
    plt.legend()
    plt.xlabel("size of training set")
    plt.ylabel("error")
    plt.show()


if __name__ == '__main__':
    main()
