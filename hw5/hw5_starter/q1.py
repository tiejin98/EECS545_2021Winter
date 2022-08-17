import numpy as np
import matplotlib.pyplot as plt
import imageio
from sklearn.metrics import pairwise_distances  # Don't use other functions in sklearn


def own_pairwise_distances(X, Y):
    n1, n_feature = X.shape
    n2, n_feature = Y.shape
    X = X.reshape(n1, n_feature, 1)
    Y = Y.reshape(n2, n_feature, 1)
    data = (X - Y.T) ** 2
    data = np.sum(data, axis=1)
    return np.sqrt(data)

def calculate_error(data, compressed_data):
    assert data.shape == compressed_data.shape
    error = np.sqrt(np.mean(np.power(data - compressed_data, 2)))
    return error




def train_kmeans(train_data, initial_centroids):
    ##### TODO: Implement here!! #####
    # Hint: pairwise_distances() might be useful
    states = {
        'centroids': initial_centroids
    }
    centroids_data = initial_centroids
    print(0, "iter pixel error:", test_kmeans(states, test_data)['pixel-error'])
    for _ in range(50):
        distances = own_pairwise_distances(train_data, centroids_data)
        res = np.argmin(distances, axis=1)
        # centroids_data = np.zeros_like(initial_centroids)
        # for c in range(len(initial_centroids)):
        #   centroids_data[c] = np.mean(train_data[res == c],axis=0)
        res = np.eye(train_data.shape[0], len(initial_centroids))[res]
        num_class = np.sum(res, axis=0)
        res = np.repeat(res, train_data.shape[1], axis=0)
        temp_data = (train_data.reshape(-1, 1) * res).reshape(-1, train_data.shape[1], len(initial_centroids))
        centroids_data = np.sum(temp_data, axis=0)
        centroids_data /= num_class
        centroids_data = centroids_data.T
        states['centroids'] = centroids_data
        print(_+1,"iter pixel error:",test_kmeans(states,test_data)['pixel-error'])
    return states

def test_kmeans(states, test_data, picture= False):
    result = {}
    centroids_data = states['centroids']
    distances = own_pairwise_distances(test_data, centroids_data)
    res = np.argmin(distances, axis=1)
    compressed_data = np.zeros_like(test_data)
    for c in range(len(centroids_data)):
        compressed_data[res == c] = centroids_data[c]
    result['pixel-error'] = calculate_error(test_data, compressed_data)
    if picture:
        test_data /= 255
        compressed_data /= 255
        # plt.imshow(test_data.reshape(512,512,3))
        # plt.show()
        plt.imshow(compressed_data.reshape(512,512,3))
        plt.show()
        #result['new-image'] = compressed_data.reshape(512,512,3)
    return result



### DO NOT CHANGE ###
def calculate_error(data, compressed_data):
    assert data.shape == compressed_data.shape
    error = np.sqrt(np.mean(np.power(data - compressed_data, 2)))
    return error


### DO NOT CHANGE ###

# Load data
img_small = np.array(imageio.imread('q12data/mandrill-small.tiff'))  # 128 x 128 x 3
img_large = np.array(imageio.imread('q12data/mandrill-large.tiff'))  # 512 x 512 x 3

ndim = img_small.shape[-1]
train_data = img_small.reshape(-1, ndim).astype(float)
test_data = img_large.reshape(-1, ndim).astype(float)

# K-means
num_centroid = 16
initial_centroid_indices = [16041, 15086, 15419, 3018, 5894, 6755, 15296, 11460,
                            10117, 11603, 11095, 6257, 16220, 10027, 11401, 13404]
initial_centroids = train_data[initial_centroid_indices, :]
states = train_kmeans(train_data, initial_centroids)
result_kmeans = test_kmeans(states, test_data,picture=True)
print('Kmeans result=', result_kmeans)
