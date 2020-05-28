"""
    This is a file you will have to fill in.
    It contains helper functions required by K-means method via iterative improvement
"""
import numpy as np
from random import sample

def init_centroids(k, inputs):
    """
    Selects k random rows from inputs and returns them as the chosen centroids
    Hint: use random.sample (it is already imported for you!)
    :param k: number of cluster centroids
    :param inputs: a 2D Numpy array, each row of which is one input
    :return: a Numpy array of k cluster centroids, one per row
    """
    # TODO

    # Creating a list of random integers, each of which represents a row in the
    # inputs array
    num_samples = inputs.shape[0]
    num_attributes = inputs.shape[1]
    random_rows = sample(range(num_samples), k)

    # Creating the numpy array of k cluster centroids, one per row
    centroids = np.zeros((k, num_attributes))
    for i in range (k):
        centroids[i] = inputs[random_rows[i]]

    return centroids

def assign_step(inputs, centroids):
    """
    Determines a centroid index for every row of the inputs using Euclidean Distance
    :param inputs: inputs of data, a 2D Numpy array
    :param centroids: a Numpy array of k current centroids
    :return: a Numpy array of centroid indices, one for each row of the inputs
    """
    # TODO
    num_samples = inputs.shape[0]
    num_centroids = centroids.shape[0]

    ## Calculating the distance of each sample from the chosen centroid
    distance_array = np.zeros((num_samples,num_centroids))
    for i in range (num_centroids):
        distance= np.linalg.norm(inputs - centroids[i], axis = 1)
        distance_array[:, i] = distance

    # Finding the minimum distance for each sample, aggregating into the centroid_indices array,
    # then returning this value
    centroid_indices = np.argmin(distance_array, axis=1)
    return centroid_indices


def update_step(inputs, indices, k):
    """
    Computes the centroid for each cluster
    :param inputs: inputs of data, a 2D Numpy array
    :param indices: a Numpy array of centroid indices, one for each row of the inputs
    :param k: number of cluster centroids, an int
    :return: a Numpy array of k cluster centroids, one per row
    """
    # TODO

    ## Horizontally stacking the inputs and the indices array, then sorting
    ## based off of last column values. The aggregate_array is sorted based off
    ## centroid index values
    num_samples = inputs.shape[0]
    reshaped_indices = indices.reshape((num_samples, 1))
    aggregate_array = np.hstack((inputs, reshaped_indices))
    indices = np.argsort(aggregate_array[:,-1])
    aggregate_array = aggregate_array[indices]

    ## Splitting into ten different arrays, each of which represents a certain value for the
    ## centroid
    array_list = np.array_split(aggregate_array, np.where(np.diff(aggregate_array[:,-1]) != 0)[0] + 1)


    ## Creating the new cluster centroids array
    updated_centroids = np.zeros((k, inputs.shape[1]))

    ## Averaging each of the arrays in the array_list to get new centroid values
    for array in array_list:

        ## Collecting the centroid_index which needs to be averaged
        centroid_value = array[0][-1]
        centroid_value = int(centroid_value)

        ## Deleting the column with the centroid_indices
        array = np.delete(array, -1, axis = -1)


        ## Averaging across the rows for each column, and inserting into the
        ## updated_centroids array
        averaged_centroid = np.average(array, axis = 0)
        updated_centroids[centroid_value] = averaged_centroid

    return updated_centroids

def kmeans(inputs, k, max_iter, tol):
    """
    Runs the K-means algorithm on n rows of inputs using k clusters via iterative improvement
    :param inputs: inputs of data, a 2D Numpy array
    :param k: number of cluster centroids, an int
    :param max_iter: the maximum number of times the algorithm can iterate trying to optimize the centroid values, an int
    :param tol: relative tolerance with regards to inertia to declare convergence, a float number
    :return: a Numpy array of k cluster centroids, one per row
    """
    # TODO
    centroids = init_centroids (k, inputs)

    for i in range (max_iter):
        centroid_indices = assign_step(inputs, centroids)
        new_centroids = update_step(inputs, centroid_indices, k)
        difference_norm = np.linalg.norm(new_centroids - centroids, axis = 1)
        original_norm = np.linalg.norm(centroids, axis = 1)
        tolerance = np.divide(difference_norm, original_norm)

        ## Checking if tolerance has been reached If it has not, we reassign centroids and
        ## continue the algorithm
        if np.any (tolerance > tol):
            centroids = new_centroids
            continue

        # Tolerance has been reached
        centroids = new_centroids
        return centroids

    return centroids
