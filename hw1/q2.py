import numpy as np
import math
import matplotlib.pyplot as plt

class linear_regression():
    def __init__(self,x,y,bias=True):
        self.bias = bias
        x = np.array(x)
        y = np.array(y)
        if len(x.shape) == 1:
            x = x.reshape(-1,1)
        if self.bias == True:
            b = np.ones(x.shape[0])
            x = np.insert(x,0,b,axis=1)
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


    def fit(self):
        temp = np.linalg.inv(np.dot(self.x.T,self.x))
        temp = np.dot(temp,self.x.T)
        self.theta = np.dot(temp,self.y)

class local_regression():
    def __init__(self,x,y):
        x = np.array(x)
        y = np.array(y)
        if len(x.shape) == 1:
            x = x.reshape(-1,1)
        b = np.ones(x.shape[0])
        x = np.insert(x,0,b,axis=1)
        self.x = x
        self.y = y
        self.theta = np.zeros(self.x.shape[1]+1)

    def get_weight(self,x,kappa):
        if x.shape[0] == self.x.shape[1]:
            dist =np.sum((x-self.x)**2,axis=1)
        else:
            dist = np.sum((x-self.x[:,1:])**2,axis=1)
        weight = np.exp(dist/(-2*kappa**2))
        return np.diag(weight/2)

    def fit_one_point(self,x,kappa):
        weight = self.get_weight(x,kappa)
        temp1 = np.linalg.inv(np.dot(np.dot(self.x.T,weight),self.x))
        temp2 = np.dot(np.dot(self.x.T,weight),self.y)
        self.theta = np.dot(temp1,temp2)

    def predict_one_point(self,x,kappa):
        self.fit_one_point(x,kappa)
        return np.dot(x,self.theta)

    def predict_all_set(self,kappa):
        y_res = []
        for i in range(self.x.shape[0]):
            y_res.append(self.predict_one_point(self.x[i],kappa))
        return np.array(y_res)

    def plot_wholeset(self,kappa,name=None):
        temp_model = linear_regression(self.x,self.y,bias=False)
        temp_model.fit()
        y_lr_pred = temp_model.forward(self.x)
        plt.scatter(self.x[:,1:],self.y)
        plt.plot(self.x[:,1:],y_lr_pred,color="r")
        y_lwlr_pred = self.predict_all_set(kappa)
        temp = np.column_stack((self.x[:,1:],y_lwlr_pred))
        temp = temp[temp[:,0].argsort()]
        plt.plot(temp[:,0],temp[:,1],color="y")
        plt.xlabel("k={}".format(kappa))
        if name != None:
            plt.savefig(name)
        plt.show()

q2x = np.load("q2x.npy")
q2y = np.load("q2y.npy")
lr = linear_regression(q2x,q2y)
lr.fit()
y_pred = lr.forward(q2x)
plt.scatter(q2x,q2y)
plt.plot(q2x,y_pred,color="r")
plt.savefig("q2-d1.png")
plt.show()


lwlr = local_regression(q2x,q2y)
lwlr.plot_wholeset(0.8,"q2-d2.png")

klist = [0.1,0.3,2,10]

for k in klist:
    lwlr.plot_wholeset(k,"q2-d3-k={}.png".format(k))


