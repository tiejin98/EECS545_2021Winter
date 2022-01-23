import numpy as np
import math
import matplotlib.pyplot as plt

class linear_regression():
    def __init__(self,x,y,bias=True):
        """

        :param x: x is the data, it should be n * d matrix
        :param y: y is the label.
        :param bias: control whether the model have the constant item.
        """
        self.bias = bias
        x = np.array(x)
        y = np.array(y)
        if len(x.shape) == 1:
            x = x.reshape(-1,1)
        self.x = x
        self.y = y
        if bias == True:
            self.theta = np.zeros(self.x.shape[1]+1)
        else:
            self.theta = np.zeros(self.x.shape[1])

    def forward(self,x):
        x = np.array(x)
        if len(x.shape) == 1:
            x = x.reshape(-1,1)
        if self.bias == True:
            b = np.ones(x.shape[0])
            x = np.insert(x,0,b,axis=1)
        return np.dot(x,self.theta)

    def mse_error(self,y_test,y_predict):
        return np.sum((y_test-y_predict)**2)/len(y_test)

    def one_update(self,x,y,lr=0.1):
        y_pred = self.forward(x)
        x = np.array(x)
        y = np.array(y)
        if len(x.shape) == 1:
            x = x.reshape(-1, 1)
        if self.bias == True:
            b = np.ones(x.shape[0])
            x = np.insert(x,0,b,axis=1)
        grad = np.sum((y_pred-y)*x.T,axis=1)
        self.theta -= lr*grad

    def fit(self,batch="total",lr=0.1,eps=0.2):
        if type(batch) == type(1):
            number_one_epoch = math.ceil(self.x.shape[0]/batch)
            mse_error = self.mse_error(self.y,self.forward(self.x))
            n = 0
            while mse_error > eps:
                for i in range(number_one_epoch):
                    if i != number_one_epoch:
                        train_x = self.x[i*batch:(i+1)*batch]
                        train_y = self.y[i*batch:(i+1)*batch]
                    else:
                        train_x = self.x[i*batch:]
                        train_y = self.y[i*batch:]
                    self.one_update(train_x,train_y,lr=lr)
                mse_error = self.mse_error(self.y, self.forward(self.x))
                n += 1
            print("sgd:after {} epoches, it converges when eps sets to {}".format(n,eps))
        else:
            mse_error = self.mse_error(self.y,self.forward(self.x))
            n = 0
            while mse_error > eps:
                self.one_update(self.x,self.y,lr=lr)
                mse_error = self.mse_error(self.y, self.forward(self.x))
                n += 1
            print("bgd:after {} epoches, it converges when eps sets to {}".format(n,eps))

class poly_linear_regression():
    def __init__(self,x,y,m):
        x = np.array(x)
        y = np.array(y)
        x = x.reshape(-1,1)
        self.x = x
        self.y = y
        self.m = m
        self.theta = np.zeros(m+1)

    def poly_transform(self,x):
        x = np.array(x)
        if len(x.shape) == 1:
            x = x.reshape(-1,1)
        b = np.ones(x.shape[0])
        x = np.insert(x,0,b,axis=1)
        for i in range(2,self.m+1):
            x = np.column_stack((x,x[:,1]**i))
        return x

    def forward(self,x):
        return np.dot(self.poly_transform(x),self.theta)

    def fit(self):
        train_x = self.poly_transform(self.x)
        temp = np.linalg.inv(np.dot(train_x.T,train_x))
        temp = np.dot(temp,train_x.T)
        self.theta = np.dot(temp,self.y)

class poly_ridge_regression():
    def __init__(self,x,y,lamb,m):
        x = np.array(x)
        y = np.array(y)
        x = x.reshape(-1,1)
        self.lamb = lamb
        self.x = x
        self.y = y
        self.m = m
        self.theta = np.zeros(m+1)

    def poly_transform(self,x):
        x = np.array(x)
        if len(x.shape) == 1:
            x = x.reshape(-1,1)
        b = np.ones(x.shape[0])
        x = np.insert(x,0,b,axis=1)
        for i in range(2,self.m+1):
            x = np.column_stack((x,x[:,1]**i))
        return x

    def forward(self,x):
        return np.dot(self.poly_transform(x),self.theta)

    def fit(self):
        train_x = self.poly_transform(self.x)
        temp = np.linalg.inv(np.dot(train_x.T,train_x)+self.lamb*np.eye(self.m+1))
        temp = np.dot(temp,train_x.T)
        self.theta = np.dot(temp,self.y)




def mse_error(y_test,y_predict):
    return np.sum((y_test-y_predict)**2)/len(y_test)

def root_error(y_test,y_predict):
    return np.sqrt(mse_error(y_test,y_predict))














q1x = np.load("q1xTrain.npy")
q1y = np.load("q1yTrain.npy")
q1model_bgd = linear_regression(q1x,q1y)
q1model_sgd = linear_regression(q1x,q1y)

q1xtest = np.load("q1xTest.npy")
q1ytest = np.load("q1yTest.npy")

q1model_bgd.fit(lr=0.001)
q1model_sgd.fit(batch=1,lr=0.001)
print("for bgd, b is {},w is {}".format(q1model_bgd.theta[0],q1model_bgd.theta[1]))
print("for sgd, b is {},w is {}".format(q1model_sgd.theta[0],q1model_sgd.theta[1]))

mlist = list(range(1,10))
train_rms = []
test_rms = []
for m in range(1,10):
    model = poly_linear_regression(q1x,q1y,m)
    model.fit()
    y_train_pred = model.forward(q1x)
    train_rms.append(root_error(q1y,y_train_pred))
    y_test_pred = model.forward(q1xtest)
    test_rms.append(root_error(q1ytest,y_test_pred))


plt.plot(mlist,train_rms,label="training",marker="o",markersize=10)
plt.plot(mlist,test_rms,marker="o",markersize=10,label="test")
plt.xlabel("M")
plt.ylabel("RMS Error")
plt.legend()
plt.savefig("q1-b1.png")
plt.show()


lamb_list = [0, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1]
ln_lamb_list = [-20]
ridge_train_rms = []
ridge_rest_rms = []
for lamb in lamb_list:
    if lamb != 0:
        ln_lamb_list.append(math.log(lamb))
    model = poly_ridge_regression(q1x,q1y,lamb,9)
    model.fit()
    y_train_pred = model.forward(q1x)
    ridge_train_rms.append(root_error(q1y,y_train_pred))
    y_test_pred = model.forward(q1xtest)
    ridge_rest_rms.append(root_error(q1ytest,y_test_pred))

plt.plot(ln_lamb_list,ridge_train_rms,label="training",marker="o",markersize=10)
plt.plot(ln_lamb_list,ridge_rest_rms,marker="o",markersize=10,label="test")
plt.xlabel("ln Lambda")
plt.ylabel("RMS Error")
plt.legend()
plt.savefig("q1-c.png")
plt.show()
