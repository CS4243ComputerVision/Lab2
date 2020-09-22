""" CS4243 Lab 2: Image Segmentations
See accompanying Jupyter notebook (lab2.ipynb) and PDF (lab2.pdf) for instructions.
"""
import cv2
import numpy as np
import random

from time import time
from skimage import color
from sklearn.neighbors import NearestNeighbors
from sklearn.utils._joblib import Parallel
from sklearn.utils._joblib import delayed

import matplotlib.pyplot as plt # Extra import added to plot graph for debug
import copy # Extra import (allowed) added.
 
# Part 1 

def smoothing(img):
    """Smooth image using Guassian filter.

    Args:
        img (np.ndarray)            : Input image of size (H, W, 3).

    Returns:
        img_smoothed (np.ndarray)   : Output smoothed image of size (H, W, 3).

    """

    """ YOUR CODE STARTS HERE """
    KERNEL_SIZE = 5
    SIGMA = 5
    img_smoothed = cv2.GaussianBlur(img, (KERNEL_SIZE, KERNEL_SIZE), SIGMA)
    """ YOUR CODE ENDS HERE """

    return img_smoothed

def RGBtoLab(img):
    """Convert RGB image into L*a*b color space.

    Args:
        img (np.ndarray)            : Input RGB image  of size (H, W, 3).


    Returns:
        lab (np.ndarray)            : Converted L*a*b image of size (H, W, 3).

    """

    """ YOUR CODE STARTS HERE """
    lab = color.rgb2lab(img)
    """ YOUR CODE ENDS HERE """
   
    return lab



# Part 2    

# To view how the cluster assignment changes, change DEBUG to True
DEBUG = False 

# Define colmap for plotting of graph for debugging purposes of 
# visualising how the point clusters changes
# Only works for K=3

if DEBUG:
    colmap = {1: 'r', 2: 'g', 3: 'y'}

# HELPER FUNCTION 1
# The notations Pi, Cj, Yi follow that of lab2.pdf
# Pi: Individual point
# Yi: Cluster id for Pi point
# Cj: Individual centroid point
# Inputs:
# (a) data points
# (b) centroids: dictionary where each Yi (key) is mapped to it's Cj (value)
# Idea: 
# (1) Iterate through every single point Pi in data
# (1a) Compute the distance of Pi wrt all the centroid points Cj
# (1b) In index 0 of new_assigned_cluster_data, we store the nearest Cj's cluster id (Yi) for Pi
# Output:
# (a) data points
# (b) new_assigned_cluster_data: index 0 contains Yi, index 1 contains corresponding color (Optional)
def assign_clusters(data, centroids):
    new_assigned_cluster_data = []
    for Pi in data:
        min_distance = float("inf")
        Yi = -1
        # Obtain Yi by comparing Pi with all possible Cj
        for Ci, Cj in centroids.items():
            # sqrt((x1 - x2)^2 - (y1 - y2)^2)
            current_distance = np.sqrt(
                np.square(Pi[0] - Cj[0]) 
                + np.square(Pi[1] - Cj[1])
            )
            if current_distance <= min_distance:
                min_distance = current_distance
                Yi = Ci
        # Append Yi: Cluster id
        # Append Color: For debug purposes
        if DEBUG:
            new_assigned_cluster_data.append([Yi, colmap[Yi]])
        else:
            new_assigned_cluster_data.append([Yi])
           
    return (data, new_assigned_cluster_data)

# HELPER FUNCTION 2
# Idea: Compute the new centroid values based on means of points in that
# cluster
def update_centroids(centroids, data, new_assigned_cluster_data):
    new_centroids = {}
    for Yi in centroids.keys():
        # Compute the mean of all the points that have been assigned to cluster id Yi
        totalSumX = 0
        totalSumY = 0
        count = 0
        for idx, cluster in enumerate(new_assigned_cluster_data):
            if cluster[0] == Yi:
                totalSumX += data[idx][0]
                totalSumY += data[idx][1]
                count += 1
        # Obtain new centroid values for each cluster
        new_centroids[Yi] = [totalSumX / count, totalSumY / count]
    return new_centroids

# HELPER FUNCTION 3
# plot_graph: self defined function to print graph for debugging purposes
def plot_graph(data, new_assigned_cluster_data, centroids, alpha = 1):
    
    fig = plt.figure()
    
    for i in range(len(data)):
        plt.scatter(data[i][0], data[i][1], color = new_assigned_cluster_data[i][1], alpha = alpha, edgecolor = 'k')
    for i in centroids.keys():
        plt.scatter(*centroids[i], color = colmap[i], edgecolor = 'w', marker = 's')
    
    plt.show()
    
def k_means_clustering(data, k):
    """ Estimate clustering centers using k-means algorithm.

    Args:
        data (np.ndarray)           : Input data with shape (n_samples, n_features)
        k (int)                     : Number of centroids

    Returns:
        labels (np.ndarray)         : Input/output integer array that stores the cluster indices for every sample.
                                      The shape is (n_samples, 1)
        centers (np.ndarray)        : Output matrix of the cluster centers, one row per each cluster center. 
                                      The shape is (k, n_features)

    """
    start = time()


    """ YOUR CODE STARTS HERE """
    # (1) Randomly pick k points from data to be centroids
    k_points = random.sample(list(data), k)
    
    # centroids[i] = [x, y], where i = 1,2..k are the cluster ids for centroids
    centroids = {
        i + 1: k_points[i]
        for i in range(k)
    }
   
    # (2) Plot graph to show position of initial centroids and data points
    if DEBUG:
        original_points_data = []
        for i in range(len(data)):
            # Set black color for original pts
            # Note that 0 is just a dummy value
            original_points_data.append([0, 'k']) 
        plot_graph(data, original_points_data, centroids)
    
    # (3) Run functions assign_clusters and update_centroids until the centroids remain 
    # unchanged.
    centroids_different = True
    while centroids_different:
        
        # (3a) Assign data pts to clusters based on nearest centroids
        data, new_assigned_cluster_data = assign_clusters(data, centroids)

        # (3b) Plot graph to show position of centroids and data points in newly assigned clusters
        if DEBUG:
            plot_graph(data, new_assigned_cluster_data, centroids, 0.5)

        # (3c) Update Centroids from the new clusters formed
        old_centroids = copy.deepcopy(centroids)
        centroids = update_centroids(centroids, data, new_assigned_cluster_data)
        
        # (3d) Check if centroids have changed
        centroids_different = False
        for idx, val in old_centroids.items():
            if centroids[idx][0] != old_centroids[idx][0] or centroids[idx][1] != old_centroids[idx][1]:
                centroids_different = True
    
    # (4) Completed K-Means
    # This part is formatting the items to be returned to the caller function
    labels = []
    for item in new_assigned_cluster_data:
        labels.append(item[0]-1)
    
    centers_list = []
    for cen in centroids.values():
        centers_list.append(cen)
    centers = np.array(centers_list)
    """ YOUR CODE ENDS HERE """

    end =  time()
    kmeans_runtime = end - start
    print("K-means running time: %.3fs."% kmeans_runtime)
    return labels, centers


def get_bin_seeds(data, bin_size, min_bin_freq=1):
    """ Generate initial bin seeds for windows sampling.

    Args:
        data (np.ndarray)           : Input data with shape (n_samples, n_features)
        bin_size (float)            : Bandwidth.
        min_bin_freq (int)          : For each bin_seed, number of the minimal points should cover.

    Returns:
        bin_seeds (List)            : Reprojected bin seeds. All bin seeds with total point number 
                                      bigger than the threshold.
    """

    """ YOUR CODE STARTS HERE """
    
    


    """ YOUR CODE ENDS HERE """
    return bin_seeds

def mean_shift_single_seed(start_seed, data, nbrs, max_iter):
    """ Find mean-shift peak for given starting point.

    Args:
        start_seed (np.ndarray)     : Coordinate (x, y) of start seed. 
        data (np.ndarray)           : Input data with shape (n_samples, n_features)
        nbrs (class)                : Class sklearn.neighbors._unsupervised.NearestNeighbors.
        max_iter (int)              : Max iteration for mean shift.

    Returns:
        peak (tuple)                : Coordinate (x,y) of peak(center) of the attraction basin.
        n_points (int)              : Number of points in the attraction basin.
                              
    """

    # For each seed, climb gradient until convergence or max_iter
    bandwidth = nbrs.get_params()['radius']
    stop_thresh = 1e-3 * bandwidth  # when mean has converged

    """ YOUR CODE STARTS HERE """
    
    """ YOUR CODE ENDS HERE """

    return peak, n_points


def mean_shift_clustering(data, bandwidth=0.7, min_bin_freq=5, max_iter=300):
    """pipline of mean shift clustering.

    Args:
        data (np.ndarray)           : Input data with shape (n_samples, n_features)
        bandwidth (float)           : Bandwidth parameter for mean shift algorithm.
        min_bin_freq(int)           : Parameter for get_bin_seeds function.
                                      For each bin_seed, number of the minimal points should cover.
        max_iter (int)              : Max iteration for mean shift.

    Returns:
        labels (np.ndarray)         : Input/output integer array that stores the cluster indices for every sample.
                                      The shape is (n_samples, 1)
        centers (np.ndarray)        : Output matrix of the cluster centers, one row per each cluster center. 
                                      The shape is (k, n_features)
    """
    start = time()
    n_jobs = None
    seeds = get_bin_seeds(data, bandwidth, min_bin_freq)
    n_samples, n_features = data.shape
    center_intensity_dict = {}

    # We use n_jobs=1 because this will be used in nested calls under
    # parallel calls to _mean_shift_single_seed so there is no need for
    # for further parallelism.
    nbrs = NearestNeighbors(radius=bandwidth, n_jobs=1).fit(data)
    # execute iterations on all seeds in parallel
    all_res = Parallel(n_jobs=n_jobs)(
        delayed(mean_shift_single_seed)
        (seed, data, nbrs, max_iter) for seed in seeds)

    # copy results in a dictionary
    for i in range(len(seeds)):
        if all_res[i] is not None:
            center_intensity_dict[all_res[i][0]] = all_res[i][1]

    if not center_intensity_dict:
        # nothing near seeds
        raise ValueError("No point was within bandwidth=%f of any seed."
                         " Try a different seeding strategy \
                         or increase the bandwidth."
                         % bandwidth)
    


    """ YOUR CODE STARTS HERE """

    


    """ YOUR CODE ENDS HERE """
    end =  time()
    kmeans_runtime = end - start
    print("mean shift running time: %.3fs."% kmeans_runtime)
    return labels, centers









#Part 3:

def k_means_segmentation(img, k):
    """Descrption.

    Args:
        img (np.ndarray)            : Input image of size (H, W, 3).
        k (int)                     : Number of centroids
    
    Returns:
        labels (np.ndarray)         : Input/output integer array that stores the cluster indices for every sample.
                                      The shape is (n_samples, 1)
        centers (np.ndarray)        : Output matrix of the cluster centers, one row per each cluster center. 
                                      The shape is (k, n_features)

    """

    """ YOUR CODE STARTS HERE """
    data = img.reshape(-1,3)
    labels, centers = k_means_clustering(data,k)

    """ YOUR CODE ENDS HERE """

    return labels,centers


def mean_shift_segmentation(img,b):
    """Descrption.

    Args:
        img (np.ndarray)            : Input image of size (H, W, 3).
        b (float)                     : Bandwidth.

    Returns:
        labels (np.ndarray)         : Input/output integer array that stores the cluster indices for every sample.
                                      The shape is (n_samples, 1)
        centers (np.ndarray)        : Output matrix of the cluster centers, one row per each cluster center. 
                                      The shape is (k, n_features)

    """

    """ YOUR CODE STARTS HERE """
    data = img.reshape(-1,3)
    labels, centers = mean_shift_clustering(data, bandwidth=b)
    
    """ YOUR CODE ENDS HERE """

    return labels, centers














"""Helper functions: You should not have to touch the following functions.
"""
def load_image(im_path):
    """Loads image and converts to RGB format

    Args:
        im_path (str): Path to image

    Returns:
        im (np.ndarray): Loaded image (H, W, 3), of type np.uint8.
    """


    im = cv2.imread(im_path)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    return im


def colors(k):

    """generate the color for the plt.scatter.

    Args:
        k (int): the number of the centroids

    Returns:
        ret (list): list of colors .

    """

    colour = ["coral", "dodgerblue", "limegreen", "deeppink", "orange", "darkcyan", "rosybrown", "lightskyblue", "navy"]
    if k <= len(colour):
        ret = colour[0:k]
    else:
        ret = []
        for i in range(k):
            ret.append((random.uniform(0, 1), random.uniform(0, 1), random.uniform(0, 1)))
    return ret

def stack_seg(img, labels, centers):
    """stack segmentations for visualization.

    Args:
        img (np.ndarray): image
        labels(np.ndarray): lables for every pixel. 
        centers(np.ndarray): cluster centers.

    Returns:
        np.vstack(result) (np.ndarray): stacked result.

    """

    labels = labels.reshape((img.shape[:-1]))
    reduced = np.uint8(centers)[labels]
    result = [np.hstack([img])]
    for i, c in enumerate(centers):
        mask = cv2.inRange(labels, i, i)
        mask = np.dstack([mask]*3) # Make it 3 channel
        ex_img = cv2.bitwise_and(img, mask)
        result.append(np.hstack([ex_img]))

    return np.vstack(result)
