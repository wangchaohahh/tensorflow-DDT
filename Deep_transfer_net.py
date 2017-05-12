"""
this is a demo about my deep transfer net by tensorflow:Alexnet+MMD
@author: wangchao
@date : 2016.12
"""

from loadData import *
import tensorflow as tf
import numpy as np

# ----------------------------load data----------------------------------------------------------------------------
source_data, source_labels = loaddata("/home/wangchao/deeplearning/transfer_learning/dataset/amazon/images")
target_data, target_labels = loaddata("/home/wangchao/deeplearning/transfer_learning/dataset/webcam/images")

test_target_data = target_data
test_target_label = target_labels
# copy target data , ensure that have the same number of sample as source data
num_source = source_labels.shape[0]
num_target = target_labels.shape[0]
# number copy is num_target*num_copy + num_copy_remainder
num_copy = num_source // num_target
num_copy_remainder = num_source % num_target
copy_target = np.array(target_data.tolist() * num_copy)
copy_target_remainder = np.array(target_data[:num_copy_remainder, ...].tolist())
target_data = np.concatenate((copy_target, copy_target_remainder))
target_data = np.uint8(target_data)
# copy target label
copy_target_label = np.array(target_labels.tolist() * num_copy)
copy_target_label_remainder = np.array(target_labels[:num_copy_remainder, ...].tolist())
target_labels = np.concatenate((copy_target_label, copy_target_label_remainder))

# # convert numpy arrays to standard TensorFlow format
# source_data = tf.pack(source_data)
# target_data = tf.pack(target_data)

# create batch data
def create_batch(start,stop,source_data, source_labels,target_data,target_labels,per_batch_size):
    for i in range(start, stop, per_batch_size):
        batch_source_img = source_data[i:i+per_batch_size, ...]
        batch_source_label = source_labels[i:i+per_batch_size]
        batch_target_img = target_data[i:i+per_batch_size, ...]
        batch_target_label = target_labels[i:i+per_batch_size]
        batch_img = np.concatenate((batch_source_img,batch_target_img),axis=0)
        batch_label = np.concatenate((batch_source_label,batch_target_label))
        yield batch_img, batch_label


print 'load data is finished'
#
# batch_data, batch_label = tf.train.shuffle_batch([source_data, source_labels], 50, 1000, 500, enqueue_many=True)


# ----------------------------ALexNet+MMD------------------------------------------------------------------------------
image_size = 28
image_channel = 3
num_label = 10
per_batch_size = 50
regularizers_penals = 0.01
learning_rate = 0.0005
MMD_penals = 0.01
num_epochs = 0.01

inputs_data = tf.placeholder("float", shape=[None, image_size, image_size, image_channel])
inputs_label = tf.placeholder('float32', shape=[None])


# conv layer 1
conv1_weights = tf.Variable(tf.random_normal([7, 7, image_channel, 96], dtype=tf.float32, stddev=0.01))
conv1_biases = tf.Variable(tf.constant(0.0, shape=[96], dtype=tf.float32))
conv1 = tf.nn.conv2d(inputs_data, conv1_weights, [1, 3, 3, 1], padding='SAME')
conv1 = tf.nn.bias_add(conv1, conv1_biases)
conv1_relu = tf.nn.relu(conv1)
conv1_norm = tf.nn.local_response_normalization(conv1_relu, depth_radius=2, alpha=0.0001, beta=0.75, bias=1.0)
conv1_pool = tf.nn.max_pool(conv1_norm, ksize=[1, 2, 2, 1], strides=[1, 1, 1, 1], padding='VALID')
# conv layer 2
conv2_weights = tf.Variable(tf.random_normal([5, 5, 96, 256], dtype=tf.float32, stddev=0.01))
conv2_biases = tf.Variable(tf.constant(1.0, shape=[256], dtype=tf.float32))
conv2 = tf.nn.conv2d(conv1_pool, conv2_weights, [1, 1, 1, 1], padding='SAME')
conv2 = tf.nn.bias_add(conv2, conv2_biases)
conv2_relu = tf.nn.relu(conv2)
conv2_norm = tf.nn.local_response_normalization(conv2_relu)
conv2_pool = tf.nn.max_pool(conv2_norm, ksize=[1, 2, 2, 1], strides=[1, 1, 1, 1], padding='VALID')

# conv layer 3
conv3_weights = tf.Variable(tf.random_normal([3, 3, 256, 384], dtype=tf.float32, stddev=0.01))
conv3_biases = tf.Variable(tf.constant(0.0, shape=[384], dtype=tf.float32))
conv3 = tf.nn.conv2d(conv2_pool, conv3_weights, [1, 1, 1, 1], padding='SAME')
conv3 = tf.nn.bias_add(conv3, conv3_biases)
conv3_relu = tf.nn.relu(conv3)

# conv layer 4
conv4_weights = tf.Variable(tf.random_normal([3, 3, 384, 384], dtype=tf.float32, stddev=0.01))
conv4_biases = tf.Variable(tf.constant(1.0, shape=[384], dtype=tf.float32))
conv4 = tf.nn.conv2d(conv3_relu, conv4_weights, [1, 1, 1, 1], padding='SAME')
conv4 = tf.nn.bias_add(conv4, conv4_biases)
conv4_relu = tf.nn.relu(conv4)

# conv layer 5
conv5_weights = tf.Variable(tf.random_normal([3, 3, 384, 256], dtype=tf.float32, stddev=0.01))
conv5_biases = tf.Variable(tf.constant(1.0, shape=[256], dtype=tf.float32))
conv5 = tf.nn.conv2d(conv4_relu, conv5_weights, [1, 1, 1, 1], padding='SAME')
conv5 = tf.nn.bias_add(conv5, conv5_biases)
conv5_relu = tf.nn.relu(conv5)
conv5_pool = tf.nn.max_pool(conv5_relu, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID')

# fc layer 1
fc1_weights = tf.Variable(tf.random_normal([256 * 3 * 3, 4096], dtype=tf.float32, stddev=0.01))
fc1_biases = tf.Variable(tf.constant(1.0, shape=[4096], dtype=tf.float32))
conv5_reshape = tf.reshape(conv5_pool, [-1, fc1_weights.get_shape().as_list()[0]])
fc1 = tf.matmul(conv5_reshape, fc1_weights)
fc1 = tf.nn.bias_add(fc1, fc1_biases)
fc1_relu = tf.nn.relu(fc1)
fc1_drop = tf.nn.dropout(fc1_relu,keep_prob=0.5 )


# fc layer 2
fc2_weights = tf.Variable(tf.random_normal([4096, 4096], dtype=tf.float32, stddev=0.01))
fc2_biases = tf.Variable(tf.constant(1.0, shape=[4096], dtype=tf.float32))
fc2 = tf.matmul(fc1_drop, fc2_weights)
fc2 = tf.nn.bias_add(fc2, fc2_biases)
fc2_relu = tf.nn.relu(fc2)
fc2_drop = tf.nn.dropout(fc2_relu, keep_prob=0.5)

# fc layer 3 - output
fc3_weights = tf.Variable(tf.random_normal([4096, num_label], dtype=tf.float32, stddev=0.01))
fc3_biases = tf.Variable(tf.constant(1.0, shape=[num_label], dtype=tf.float32))
fc3 = tf.matmul(fc2_drop, fc3_weights)
logits = tf.nn.bias_add(fc3, fc3_biases)

# source loss
source_logits = logits[:per_batch_size, ...]
source_predict = tf.cast(tf.arg_max(source_logits,dimension=1), tf.float32)
source_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(source_predict, inputs_label[:per_batch_size]))

# l2 regularization
regularizers = (tf.nn.l2_loss(conv1_weights) + tf.nn.l2_loss(conv1_biases) +
                tf.nn.l2_loss(conv2_weights) + tf.nn.l2_loss(conv2_biases) +
                tf.nn.l2_loss(conv3_weights) + tf.nn.l2_loss(conv3_biases) +
                tf.nn.l2_loss(conv4_weights) + tf.nn.l2_loss(conv4_biases) +
                tf.nn.l2_loss(conv5_weights) + tf.nn.l2_loss(conv5_biases) +
                tf.nn.l2_loss(fc1_weights) + tf.nn.l2_loss(fc1_biases) +
                tf.nn.l2_loss(fc2_weights) + tf.nn.l2_loss(fc2_biases) +
                tf.nn.l2_loss(fc3_weights) + tf.nn.l2_loss(fc3_biases))
# MMD loss
fc1_MMD = tf.abs(tf.reduce_mean(fc1_drop[:per_batch_size, ...]
                                -fc1_drop[per_batch_size:, ...]))/per_batch_size
fc2_MMD = tf.abs(tf.reduce_mean(fc2_drop[:per_batch_size, ...]
                                -fc2_drop[per_batch_size:, ...]))/per_batch_size
class_MMD = tf.abs(tf.reduce_mean(logits[:per_batch_size, ...]
                                  -logits[per_batch_size:, ...]))/per_batch_size
MMD = fc1_MMD + fc2_MMD + class_MMD

loss = source_loss + regularizers_penals * regularizers + MMD * MMD_penals
# -----------------------------------------------------------------------------------------------------

# train
train = tf.train.AdamOptimizer(learning_rate=0.01).minimize(loss)

# accuracy
predict = tf.cast(tf.argmax(logits[per_batch_size:,...], 1),tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predict, inputs_label[per_batch_size:,...]), tf.float32))

# session
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)

    # create batch data
    batch_all = num_source // 50
    for epoch in range(100):
        start = 0
        batch_generator = create_batch(start,num_source,source_data, source_labels, target_data, target_labels, per_batch_size=50)

        for i in range(batch_all):
        # batch_source_img, batch_source_label = tf.train.shuffle_batch([source_data, source_labels],
        #                                                        batch_size, capacity=1000,
        #                                                               min_after_dequeue=500
        #                                         )
        # batch_target_img, batch_target_label = tf.train.shuffle_batch([target_data, target_labels],
        #                                                               batch_size, capacity=1000,
        #                                                               min_after_dequeue=500
        #                                         )
        #
        # batch_img = tf.concat(0, [batch_source_img, batch_target_img])
        # batch_label = tf.concat(0, [batch_source_label, batch_target_label])

            batch_img, batch_label = batch_generator.next()

            print ('epoch :%i, accuracy is : %f, loss:%f' % (i, sess.run(accuracy, feed_dict={inputs_data: batch_img,
                                                                             inputs_label: batch_label}),
                                                         sess.run(loss,feed_dict={inputs_data: batch_img,
                                                                             inputs_label: batch_label})))
            sess.run(train, feed_dict={inputs_data : batch_img, inputs_label : batch_label})


#  train is finished and test target accuracy
print 'train is finished'



