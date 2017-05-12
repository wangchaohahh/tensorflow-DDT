"""
this is a file to load small images and labels
@author: wangchao
@date : 2016.12
"""

import os
import matplotlib.pyplot as plt
from matplotlib import offsetbox
from scipy.misc import imread, imresize
import numpy as np
from sklearn.utils import shuffle
from sklearn import manifold
from time import time



def loaddata(filePath):
    images = []
    labels = []
    i = 0
    ten_class = ['projector','bike','calculator','keyboard','mug',
                 'headphones','monitor','laptop_computer','back_pack','mouse']
    # for datapath in os.listdir(filePath):
    for datapath in ten_class:
        for filename in os.listdir(filePath + '/' + datapath):
            filename = filePath + '/' + datapath + '/' + filename
            img = imread(filename)
            img = imresize(img, (28, 28))
            labels.append(i)
            images.append(img)
        i += 1
    images = np.asarray(images)
    labels = np.array(labels).astype(float)
    images, labels = shuffle(images, labels, random_state=0)
    return images, labels



# test load data
# images, labels = loaddata("/home/wangchao/deeplearning/transfer_learning/dataset/webcam/images")
# np.savetxt('test.tsv',delimiter='\t')


# tensor_images = tf.placeholder('uint8',shape=images.shape)
# batch_img = tf.train.shuffle_batch([tensor_images],30,1000,500,enqueue_many= True)
# init = tf.global_variables_initializer()
# with tf.Session() as sess:
#     sess.run(init)
#     batch_img = sess.run(batch_img,feed_dict={tensor_images:images})
#     print batch_img


# visualize
images, labels = loaddata("/home/wangchao/deeplearning/transfer_learning/dataset/webcam/images")
n_sample = images.shape[0]
images = np.reshape(images,(n_sample,-1))

n_feature = images.shape[1]
n_neighbors = 30

# Scale and visualize the embedding vectors
def plot_embedding(X, title=None):
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)

    plt.figure()
    ax = plt.subplot(111)
    for i in range(X.shape[0]):
        plt.text(X[i, 0], X[i, 1], str(labels[i]),
                 color=plt.cm.Set1(labels[i] / 10.),
                 fontdict={'weight': 'bold', 'size': 9})

    if hasattr(offsetbox, 'AnnotationBbox'):
        # only print thumbnails with matplotlib > 1.0
        shown_images = np.array([[1., 1.]])  # just something big
        for i in range(images.shape[0]):
            dist = np.sum((X[i] - shown_images) ** 2, 1)
            if np.min(dist) < 4e-3:
                # don't show points that are too close
                continue
            shown_images = np.r_[shown_images, [X[i]]]
            imagebox = offsetbox.AnnotationBbox(
                offsetbox.OffsetImage(images[i], cmap=plt.cm.gray_r),
                X[i])
            ax.add_artist(imagebox)
    plt.xticks([]), plt.yticks([])
    if title is not None:
        plt.title(title)


# Plot images of the digits
# n_img_per_row = 20
# img = np.zeros((10 * n_img_per_row, 10 * n_img_per_row))
# for i in range(n_img_per_row):
#     ix = 10 * i + 1
#     for j in range(n_img_per_row):
#         iy = 10 * j + 1
#         img[ix:ix + 28, iy:iy + 28] = images[i * n_img_per_row + j].reshape((28, 28))
#
# plt.imshow(img, cmap=plt.cm.binary)
# plt.xticks([])
# plt.yticks([])
# plt.title('A selection from the 64-dimensional digits dataset')

# t-SNE embedding of the digits dataset
print("Computing t-SNE embedding")
tsne = manifold.TSNE(n_components=2, init='pca', random_state=0)
t0 = time()
X_tsne = tsne.fit_transform(images)

plot_embedding(X_tsne,
               "t-SNE embedding of the digits (time %.2fs)" %
               (time() - t0))

plt.show()