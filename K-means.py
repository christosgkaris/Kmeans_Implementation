import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt

np.random.seed(12345)


# Function for loading the iris data
# load_data returns a 2D numpy array where each row is an example
# and each column is a given feature.
def load_data():
    iris = datasets.load_iris()
    return iris.data


# calculate the euclidean distance of two points
def calculate_distance(p1, p2):
    return np.sqrt((p1[0]-p2[0])**2+(p1[1]-p2[1])**2+(p1[2]-p2[2])**2+(p1[3]-p2[3])**2)


# Assign labels to each example given the center of each cluster
def assign_labels(X, centers):
    labels = np.zeros(len(X)) # length = 150
    for i in range(0, len(X)):
        dist = 10000 # initialize with a big number
        for j in range(0, len(centers)): # length = K
            if calculate_distance(X[i], centers[j]) < dist: # trying to find the
                dist = calculate_distance(X[i], centers[j]) # smallest distance
                labels[i] = j # assign label to X[i], it belongs to cluster j
    return labels


# Calculate the center of each cluster given the label of each example
def calculate_centers(X, labels):
    
    # calculate the number of centers and initialize centers' matrix
    num_centers = 1
    for i in range(0, len(labels)): # length = 150
        if labels[i] > num_centers:
            num_centers = num_centers + 1
    num_centers = num_centers + 1 # we now have the number of centers
    centers = np.zeros([num_centers, 4]) # length = K

    # calculate the sum in each cluster and the number of elements    
    n = np.zeros([num_centers]) # initialize the number of elements matrix
    for i in range(0, num_centers): # length = K
        for j in range(0, len(labels)): # length = 150
            if labels[j] == i:
                centers[i] = centers[i] + X[j] # calculate the sum
                n[i] = n[i] + 1 # calculate the number of elements   

    # calculate the new centers
    for i in range(0, num_centers): # length = K
        if n[i] > 0:
            centers[i] = centers[i] / n[i]
        else:
            print("Division by 0")
            
    return centers


# Test if the algorithm has converged
# Should return a bool stating if the algorithm has converged or not.
def test_convergence(old_centers, new_centers):
    if np.all(old_centers == new_centers): 
        return True
    else: 
        return False


# Evaluate the preformance of the current clusters
# This function should return the total mean squared error of the given clusters
def evaluate_performance(X, labels, centers):
    sse = 0.0
    for i in range(0, len(X)): # length = 150
        sse = sse + calculate_distance(X[i], centers[int(labels[i])])**2
    return sse


# Algorithm for preforming K-means clustering on the give dataset
def k_means(X, K):    

    # initialize the centers
    centers = np.zeros([K, 4])
    for i in range(0, K):
        v = i*10 + np.random.randint(1, 9) + 3*K # random initializer
        centers[i] = X[v]

    # begin loop process
    old_centers = np.zeros([K, 4])
    while test_convergence(old_centers, centers) == False:
        # first we have to assign every point to a center
        labels = assign_labels(X, centers)
        # then we have to calculate the new centers
        old_centers = centers
        centers = calculate_centers(X, labels)

    # finally we can return the total mean squared error of the given clusters
    return evaluate_performance(X, labels, centers)


# Algorithm for final evaluation
def evaluate(X):
    costs = np.zeros([9])
    for k in range(0, 9):
        costs[k] = k_means(X, k+2)
    plt.plot([2,3,4,5,6,7,8,9,10], costs)
    
    plt.title('Elbow Method')
    plt.xlabel('K')
    plt.ylabel('Cost (SSE)')
    plt.show()
    
    return costs
