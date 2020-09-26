from sklearn import datasets

from lab2 import *

n_samples = 400
bandwidth = 4

if __name__ == '__main__':
    blobs = datasets.make_blobs(n_samples=n_samples, random_state=8)[0]
    print(blobs)


    labels, centroids  = mean_shift_clustering(blobs, bandwidth=bandwidth)
