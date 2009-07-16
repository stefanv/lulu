# ----- Segmentation parameters -----
N = 4
beta = 1.5 # Smoothing
min_area = 500
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
    if area > min_area:
        for cr in pulses[area]:
            crh.set_array(impulse_strength, cr,
                          np.abs(crh.get_value(cr)), 'add')

def ICM(data, N, beta):
    print "Performing ICM segmentation..."

    # Initialise segmentation using kmeans
    print "K-means initialisation..."
    clusters, labels = kmeans2(np.ravel(data), N)

    print "Iterative segmentation..."
    f = data.copy()

    def _minimise_cluster_distance(data, labels, N, beta):
        data_flat = np.ravel(data)
        cluster_means = np.array(
            [np.mean(data_flat[labels == k]) for k in range(N)]
            )
        variance = np.sum((data_flat - cluster_means[labels])**2) \
                   / data_flat.size

        # How many of the 8-connected neighbouring pixels are in the
        # same cluster?
        count = np.zeros(data.shape + (N,), dtype=int)
        count_inside = count[1:-1, 1:-1, :]

        labels_img = labels.reshape(data.shape)
        for k in range(N):
            count_inside[..., k] += (k == labels_img[1:-1:, 2:])
            count_inside[..., k] += (k == labels_img[2:, 1:-1])
            count_inside[..., k] += (k == labels_img[:-2, 1:-1])
            count_inside[..., k] += (k == labels_img[1:-1, :-2])

            count_inside[..., k] += (k == labels_img[:-2, :-2])
            count_inside[..., k] += (k == labels_img[2:, 2:])
            count_inside[..., k] += (k == labels_img[:-2, 2:])
            count_inside[..., k] += (k == labels_img[2:, :-2])

        count = count.reshape((len(labels), N))
        cluster_measure = (data_flat[:, None] - cluster_means)**2 \
                          - beta * variance * count
        labels = np.argmin(cluster_measure, axis=1)

        return cluster_means, labels

    # Initialise segmentation
    cluster_means, labels = _minimise_cluster_distance(f, labels, N, 0)

    stable_counter = 0
    old_label_diff = 0
    i = 0
    while stable_counter < 3:
        i += 1

        cluster_means, labels_ = \
                       _minimise_cluster_distance(f, labels, N, beta)

        new_label_diff = np.sum(labels_ != labels)
        if  new_label_diff != old_label_diff:
            stable_counter = 0
        else:
            stable_counter += 1
        old_label_diff = new_label_diff

        labels = labels_

    print "Clustering converged after %d steps." % i

    return labels.reshape(data.shape)

labels_original = ICM(img, N=N, beta=beta)
labels_impulse = ICM(impulse_strength, N=N, beta=beta)

plt.subplot(2, 2, 1)
plt.imshow(img, interpolation='nearest', cmap=plt.cm.gray)
plt.title('Original Image')

plt.subplot(2, 2, 2)
plt.imshow(impulse_strength, interpolation='nearest', cmap=plt.cm.gray)
plt.title('Impulse Strength\n(pulses with area > %d)' % min_area)

plt.subplot(2, 2, 3)
plt.imshow(labels_original, interpolation='nearest', cmap=plt.cm.gray)
plt.xticks([])
plt.yticks([])
plt.title(r'Segmentation ($\beta$ = %.1f)' % beta)

plt.subplot(2, 2, 4)
plt.imshow(labels_impulse, interpolation='nearest', cmap=plt.cm.gray)
plt.xticks([])
plt.yticks([])
plt.title(r'Segmentation ($\beta$ = %.1f)' % beta)

plt.suptitle('Iterated Conditional Modes (ICM) Segmentation')
plt.show()
