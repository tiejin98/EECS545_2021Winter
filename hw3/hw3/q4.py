# EECS 545 HW3 Q4

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(545)

# Instruction: use these hyperparameters for both (b) and (d)
eta = 0.5
C = 5
iterNums = [5, 50, 100, 1000, 5000, 6000]


def svm_train_bgd(matrix, label, nIter):
    # Implement your algorithm and return state (e.g., learned model)
    N, D = matrix.shape
    w = np.zeros(D)
    b = 0
    c = 5
    for n in range(nIter):
        a = 0.5/(1+(n+1)*0.5)
        grad_w = w.copy()
        h = np.dot(w,matrix.T)+b
        I = h*(label.squeeze())
        index = I < 1
        temp_label = np.repeat(label,D,axis=1)
        grad_w -= c*np.sum(temp_label[index]*matrix[index],axis=0)
        grad_b = -c*np.sum(label[index])
        w -= a*grad_w
        b -= 0.01*a*grad_b
    state = {"w":w, "b":b}
    return state


def svm_train_sgd(matrix, label, nIter):
    # Implement your algorithm and return state (e.g., learned model)
    N, D = matrix.shape
    w = np.zeros(D)
    b = 0
    c = 5
    for n in range(nIter):
        a = 0.5/(1+(n+1)*0.5)
        for i in range(N):
            grad_w = w/N
            h = np.dot(w,matrix[i])+b
            I = h*label[i]
            if I < 1:
                grad_w -= c*label[i]*matrix[i]
                grad_b = -c*label[i]
                w -= a * grad_w
                b -= 0.01 * a * grad_b
            else:
                w -= a*grad_w
    state = {"w":w, "b":b}

    return state


def svm_test(matrix, state):
    # Classify each test data as +1 or -1
    output = np.ones( (matrix.shape[0], 1) )
    w = state["w"]
    b = state["b"]
    h = np.dot(w, matrix.T) + b
    for i in range(len(h)):
        if h[i] <0:
            output[i] = -1

    return output


def evaluate(output: np.ndarray, label: np.ndarray, nIter: int) -> float:
    # Use the code below to obtain the accuracy of your algorithm
    accuracy = (label * output > 0).sum() * 1. / len(output)
    print('[Iter {:4d}: accuracy = {:2.4f}%'.format(nIter, 100 * accuracy))

    return accuracy


def load_data():
    # Note1: label is {-1, +1}
    # Note2: data matrix shape  = [Ndata, 4]
    # Note3: label matrix shape = [Ndata, 1]

    # Load data
    q4_data = np.load('q4_data/q4_data.npy', allow_pickle=True).item()

    train_x = q4_data['q4x_train']
    train_y = q4_data['q4y_train']
    test_x = q4_data['q4x_test']
    test_y = q4_data['q4y_test']
    return train_x, train_y, test_x, test_y


def run_bgd(train_x, train_y, test_x, test_y):
    '''(c) Implement SVM using **batch gradient descent**.
    For each of the nIter's, print out the following:

    *   Parameter w
    *   Parameter b
    *   Test accuracy (%)
    '''
    for nIter in iterNums:
        # Train
        state = svm_train_bgd(train_x, train_y, nIter)
        print("w:{},b:{}".format(state["w"],state["b"]))
        # TODO: Test and evluate
        prediction = svm_test(test_x, state)
        evaluate(prediction, test_y, nIter)


def run_sgd(train_x, train_y, test_x, test_y):
    '''(c) Implement SVM using **stocahstic gradient descent**.
    For each of the nIter's, print out the following:

    *   Parameter w
    *   Parameter b
    *   Test accuracy (%)

    [Note: Use the same hyperparameters as (b)]
    [Note: If you implement it correctly, the running time will be ~15 sec]
    '''
    for nIter in iterNums:
        # Train
        state = svm_train_sgd(train_x, train_y, nIter)
        print("w:{},b:{}".format(state["w"],state["b"]))
        # TODO: Test and evluate
        prediction = svm_test(test_x, state)
        evaluate(prediction, test_y, nIter)


def main():
    train_x, train_y, test_x, test_y = load_data()
    run_bgd(train_x, train_y, test_x, test_y)
    run_sgd(train_x, train_y, test_x, test_y)


if __name__ == '__main__':
    main()
