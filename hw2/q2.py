import numpy as np
import matplotlib.pyplot as plt

class softmax_regression():
    def __init__(self, x, y, bias=True):
        """

        :param x: x is the data, it should be n * d matrix
        :param y: y is the label.
        :param bias: control whether the model have the constant item.
        """
        if len(x.shape)== 1:
            x = x.reshape(-1,1)
        self.bias = bias
        self.x = x
        self.y = y
        K = np.unique(y).shape[0]
        if self.bias:
            self.theta = np.zeros((self.x.shape[1]+1,K))
        else:
            self.theta = np.zeros((self.x.shape[1],K))

    def forward(self,x,label=True):
        if len(x.shape)== 1:
            x = x.reshape(-1,1)
        if self.bias:
            b = np.ones(x.shape[0])
            temp_x = np.insert(x,0,b,axis=1)
        if not label:
            temp = np.exp(np.dot(temp_x,self.theta))
            for i in range(temp.shape[0]):
                temp[i] = temp[i]/np.sum(temp[i])
            return temp
        else:
            temp = np.exp(np.dot(temp_x,self.theta))
            for i in range(temp.shape[0]):
                temp[i] = temp[i]/np.sum(temp[i])
            return np.argsort(temp)[:,::-1][:,0]+1

    def one_update(self,lr=1):
        value_res = self.forward(self.x,label=False)
        temp_theta = self.theta.copy()
        if self.bias:
            b = np.ones(self.x.shape[0])
            temp_x = np.insert(self.x,0,b,axis=1)
        else:
            temp_x = self.x.copy()
        for i in range(self.theta.shape[1]-1):
            grad = np.zeros(self.theta.shape[0])
            for j in range(self.x.shape[0]):
                if int(self.y[j]) == i+1:
                    grad += (1-value_res[j,i])*temp_x[j]
                else:
                    grad += -value_res[j,i]*temp_x[j]
            temp_theta[:,i] += lr*grad
        self.theta = temp_theta

    def test_error_01(self,test_y,y_bar):
        test_y = np.array(test_y).reshape(-1,)
        y_bar = np.array(y_bar).reshape(-1,)
        if len(test_y) != len(y_bar):
            raise ValueError('two y should have same length')
        total = 0
        for i in range(len(y_bar)):
            if y_bar[i] - test_y[i] == 0:
                total += 1
        return total/len(y_bar)


    def fit_epoch(self,lr=1,epoch=300):
        y_before = self.forward(self.x)
        acc_before = self.test_error_01(self.y,y_before)
        print("Before training, we have training acc:{}".format(acc_before))
        for _ in range(epoch):
            self.one_update(lr)
        y_after = self.forward(self.x)
        acc_after = self.test_error_01(self.y,y_after)
        print("After {} epoches training, we have training acc:{}".format(epoch,acc_after))







data = np.load("q2_data.npz")
q2x = dict(data)["q2x_train"]
q2y = dict(data)["q2y_train"]
q2x_test = dict(data)["q2x_test"]
q2y_test = dict(data)["q2y_test"]
test = softmax_regression(q2x,q2y)
test.fit_epoch(lr=0.0005,epoch=300)
y_pred = test.forward(q2x_test)
acc = test.test_error_01(q2y_test,y_pred)
print("we can get the test acc is {}".format(acc))

