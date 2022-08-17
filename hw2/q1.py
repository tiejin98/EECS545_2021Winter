import numpy as np
import matplotlib.pyplot as plt

class logit_regrssion():
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
        if self.bias:
            self.theta = np.zeros(self.x.shape[1]+1)
        else:
            self.theta = np.zeros(self.x.shape[1])


    def forward(self,x,label=True):
        if len(x.shape)== 1:
            x = x.reshape(-1,1)
        if self.bias == True:
            b = np.ones(x.shape[0])
            x = np.insert(x,0,b,axis=1)
        temp_y = 1/(1+np.exp(-1*(np.dot(x,self.theta))))
        if label == True:
            res = []
            for y_bar in temp_y:
                if y_bar >= 0.5:
                    res.append(1)
                else:
                    res.append(0)
            return res
        else:
            return temp_y

    def test_error_01(self,test_y,y_bar):
        test_y = np.array(test_y).reshape(-1,)
        y_bar = np.array(y_bar).reshape(-1,)
        if len(test_y) != len(y_bar):
            raise ValueError('two y should have same length')
        total = 0
        for i in range(len(y_bar)):
            if y_bar[i] - test_y[i] != 0:
                total += 1
        return total/len(y_bar)


    def likelihood_function(self,x,y):
        forward_res = self.forward(x,label=False)
        L = np.dot(y,np.log(forward_res)) + np.dot(1-y,np.log(1-forward_res))
        return L

    def first_order(self):
        if self.bias:
            b = np.ones(self.x.shape[0])
            temp_x = np.insert(self.x,0,b,axis=1)
        forward_res = self.forward(self.x,label=False)
        return np.dot(self.y-forward_res,temp_x)

    def second_order(self):
        if self.bias:
            b = np.ones(self.x.shape[0])
            temp_x = np.insert(self.x,0,b,axis=1)
        dii = -np.exp(-1*(np.dot(temp_x,self.theta)))/(1+np.exp(-1*(np.dot(temp_x,self.theta))))**2
        D = np.diag(dii)
        temp = np.dot(temp_x.T,D)
        return np.dot(temp,temp_x)

    def one_step_newton(self,lr=1):
        First = self.first_order()
        H = self.second_order()
        self.theta -= lr*np.dot(np.linalg.inv(H),First)

    def fit_epoch(self,lr=1,epoch=10):
        print("Before training, we have log-likelihood is {}".format(self.likelihood_function(self.x,self.y)))
        for _ in range(epoch):
            self.one_step_newton(lr)
        print("After {} round, we can get log-likelihood is {}".format(epoch,self.likelihood_function(self.x,self.y)))

    def fit_eps(self,lr=1,eps=1e-7):
        J0 = eps
        J1 = self.likelihood_function(self.x,self.y)
        i = 0
        print("Before training, we have log-likelihood is {}".format(J1))
        while abs(J1-J0) > eps:
            i +=1
            J0 = J1
            self.one_step_newton(lr)
            J1 = self.likelihood_function(self.x,self.y)
        print("Setting converge condition to {}, it converges after {} epoches, "
              "we can get log-likelihood is {}".format(eps,i,J1))

    def plot_boundry(self,x,y):
        ax = plt.subplot()
        x1 = x[y==0]
        x2 = x[y==1]
        ax.scatter(x1[:,0],x1[:,1],color = "r", label = "y=0")
        ax.scatter(x2[:,0],x2[:,1],color = "b", label = "y=1")
        x1_list = np.linspace(np.min(x1[:,0])-np.mean(x1[:,0]), np.max(x1[:,0])+np.mean(x1[:,0]),200)
        x2_list = -self.theta[0]/self.theta[2] - self.theta[1]*x1_list/self.theta[2]
        ax.plot(x1_list,x2_list,label ="decision boundry")
        plt.xlabel("x1")
        plt.ylabel("x2")
        plt.legend()
        plt.show()






q1x = np.load("q1x.npy")
q1y = np.load("q1y.npy")
test = logit_regrssion(q1x,q1y)
test.fit_eps()
print("theta:",test.theta)
test.plot_boundry(q1x,q1y)