# ----- Segmentation parameters -----
N = 4
beta = 0.4 # Smoothing
# -----------------------------------

from demo import load_image

import numpy as np
from scipy.cluster.vq import kmeans2
import matplotlib.pyplot as plt

import lulu
import lulu.connected_region_handler as crh

img = load_image('truck_and_apcs_small.jpg')

pulses = lulu.decompose(img)

impulse_strength = np.zeros(img.shape, dtype=int)
for area in pulses:
    for cr in pulses[area]:
        crh.set_array(impulse_strength, cr, abs(crh.get_value(cr)), 'add')

data = impulse_strength

#data = img

# Initialise segmentation using kmeans
print "K-means initialisation..."
clusters, labels = kmeans2(np.ravel(data), N)

# ICM
print "ICM clustering..."
f = data.copy()

def minimise_cluster_distance(data, labels, N, b=1.0):
    data_flat = np.ravel(data)
    cluster_means = [np.mean(data_flat[labels == k]) for k in range(N)]
    normalised_data_sqr = (data_flat[:, None] - cluster_means)**2
    variance = np.sum(normalised_data_sqr) / N

    # How many of the 4-connected neighbouring pixels are in the same cluster?
    count = np.zeros(data.shape + (4,), dtype=int)
    count_inside = count[1:-1, 1:-1, :]

    labels_img = labels.reshape(data.shape)
    for k in range(N):
        count_inside[..., k] += (k == labels_img[1:-1:, 2:])
        count_inside[..., k] += (k == labels_img[2:, 1:-1])
        count_inside[..., k] += (k == labels_img[:-2, 1:-1])
        count_inside[..., k] += (k == labels_img[1:-1, :-2])

    count = count.reshape((len(labels), 4))
    cluster_measure = normalised_data_sqr - b * variance * count
    labels = np.argmin(cluster_measure, axis=1)

    return cluster_means, labels

cluster_means, labels = minimise_cluster_distance(f, labels, N=N, b=0)

stable_counter = 0
old_label_diff = 0
i = 0
while stable_counter < 3:
    i += 1

    cluster_means, labels_ = minimise_cluster_distance(f, labels, N=N, b=beta)

    new_label_diff = np.sum(labels_ != labels)
    if  new_label_diff != old_label_diff:
        stable_counter = 0
    else:
        stable_counter += 1
    old_label_diff = new_label_diff

    labels = labels_

print "Clustering converged after %d steps." % i
plt.imshow(labels.reshape(img.shape), interpolation='nearest', cmap=plt.cm.gray)
plt.xticks([])
plt.yticks([])
plt.show()
