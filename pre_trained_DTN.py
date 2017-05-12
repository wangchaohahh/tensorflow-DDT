"""
this is a demo about my deep transfer net by tensorflow:Alexnet+MMD
@author: wangchao
@date : 2016.12
"""

from loadData import *
import tensorflow as tf
import numpy as np
import os
import time

# -------------------------load data-------------------------------------------------------
source_data, source_labels = loaddata("/home/wangchao/deeplearning/transfer_learning/dataset/amazon/images")
target_data, target_labels = loaddata("/home/wangchao/deeplearning/transfer_learning/dataset/webcam/images")

# num_source_img = source_labels.shape[0]
# num_target_img = target_labels.shape[0]
# all_img = np.concatenate((source_data,target_data))
# all_labels = np.concatenate((source_labels,target_labels))
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

# create batch data
def create_batch(start,source_data, source_labels,target_data,target_labels,num_all_img,per_batch_size):
    for i in range(start, num_all_img, per_batch_size):
        batch_source_img = source_data[i:i+per_batch_size, ...]
        batch_source_label = source_labels[i:i+per_batch_size]
        batch_target_img = target_data[i:i+per_batch_size, ...]
        batch_target_label = target_labels[i:i+per_batch_size]
        batch_img = np.concatenate((batch_source_img,batch_target_img),axis=0)
        batch_label = np.concatenate((batch_source_label,batch_target_label))
        yield batch_img, batch_label

print 'load data is finished'
# ----------------------------------pre_trained_alexnet---------------------------------------------------------
# load pre_trained net weights
net_data = np.load("bvlc_alexnet.npy").item()

train_x = np.zeros((1, 227,227,3)).astype(float)
train_y = np.zeros((1, 1000))
xdim = train_x.shape[1:]
ydim = train_y.shape[1]

# (self.feed('data')
#         .conv(11, 11, 96, 4, 4, padding='VALID', name='conv1')
#         .lrn(2, 2e-05, 0.75, name='norm1')
#         .max_pool(3, 3, 2, 2, padding='VALID', name='pool1')
#         .conv(5, 5, 256, 1, 1, group=2, name='conv2')
#         .lrn(2, 2e-05, 0.75, name='norm2')
#         .max_pool(3, 3, 2, 2, padding='VALID', name='pool2')
#         .conv(3, 3, 384, 1, 1, name='conv3')
#         .conv(3, 3, 384, 1, 1, group=2, name='conv4')
#         .conv(3, 3, 256, 1, 1, group=2, name='conv5')
#         .fc(4096, name='fc6')
#         .fc(4096, name='fc7')
#         .fc(1000, relu=False, name='fc8')
#         .softmax(name='prob'))


def conv(input, kernel, biases, k_h, k_w, c_o, s_h, s_w,  padding="VALID", group=1):
    '''From https://github.com/ethereon/caffe-tensorflow
    '''
    c_i = input.get_shape()[-1]
    assert c_i%group==0
    assert c_o%group==0
    convolve = lambda i, k: tf.nn.conv2d(i, k, [1, s_h, s_w, 1], padding=padding)


    if group==1:
        conv = convolve(input, kernel)
    else:
        input_groups = tf.split(3, group, input)
        kernel_groups = tf.split(3, group, kernel)
        output_groups = [convolve(i, k) for i,k in zip(input_groups, kernel_groups)]
        conv = tf.concat(3, output_groups)
    return  tf.reshape(tf.nn.bias_add(conv, biases), [-1]+conv.get_shape().as_list()[1:])

x = tf.placeholder(tf.float32, (None,) + xdim)
y = tf.placeholder(tf.float32,shape=[None])
learning_rate = 0.0001
MMD_penals = 0.01

#conv1
#conv(11, 11, 96, 4, 4, padding='VALID', name='conv1')
k_h = 11; k_w = 11; c_o = 96; s_h = 4; s_w = 4
conv1W = tf.Variable(net_data["conv1"][0])
conv1b = tf.Variable(net_data["conv1"][1])
conv1_in = conv(x, conv1W, conv1b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=1)
conv1 = tf.nn.relu(conv1_in)

#lrn1
#lrn(2, 2e-05, 0.75, name='norm1')
radius = 2; alpha = 2e-05; beta = 0.75; bias = 1.0
lrn1 = tf.nn.local_response_normalization(conv1,
                                                  depth_radius=radius,
                                                  alpha=alpha,
                                                  beta=beta,
                                                  bias=bias)

#maxpool1
#max_pool(3, 3, 2, 2, padding='VALID', name='pool1')
k_h = 3; k_w = 3; s_h = 2; s_w = 2; padding = 'VALID'
maxpool1 = tf.nn.max_pool(lrn1, ksize=[1, k_h, k_w, 1], strides=[1, s_h, s_w, 1], padding=padding)


#conv2
#conv(5, 5, 256, 1, 1, group=2, name='conv2')
k_h = 5; k_w = 5; c_o = 256; s_h = 1; s_w = 1; group = 2
conv2W = tf.Variable(net_data["conv2"][0])
conv2b = tf.Variable(net_data["conv2"][1])
conv2_in = conv(maxpool1, conv2W, conv2b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
conv2 = tf.nn.relu(conv2_in)


#lrn2
#lrn(2, 2e-05, 0.75, name='norm2')
radius = 2; alpha = 2e-05; beta = 0.75; bias = 1.0
lrn2 = tf.nn.local_response_normalization(conv2,
                                                  depth_radius=radius,
                                                  alpha=alpha,
                                                  beta=beta,
                                                  bias=bias)

#maxpool2
#max_pool(3, 3, 2, 2, padding='VALID', name='pool2')
k_h = 3; k_w = 3; s_h = 2; s_w = 2; padding = 'VALID'
maxpool2 = tf.nn.max_pool(lrn2, ksize=[1, k_h, k_w, 1], strides=[1, s_h, s_w, 1], padding=padding)

#conv3
#conv(3, 3, 384, 1, 1, name='conv3')
k_h = 3; k_w = 3; c_o = 384; s_h = 1; s_w = 1; group = 1
conv3W = tf.Variable(net_data["conv3"][0])
conv3b = tf.Variable(net_data["conv3"][1])
conv3_in = conv(maxpool2, conv3W, conv3b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
conv3 = tf.nn.relu(conv3_in)

#conv4
#conv(3, 3, 384, 1, 1, group=2, name='conv4')
k_h = 3; k_w = 3; c_o = 384; s_h = 1; s_w = 1; group = 2
conv4W = tf.Variable(net_data["conv4"][0])
conv4b = tf.Variable(net_data["conv4"][1])
conv4_in = conv(conv3, conv4W, conv4b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
conv4 = tf.nn.relu(conv4_in)


#conv5
#conv(3, 3, 256, 1, 1, group=2, name='conv5')
k_h = 3; k_w = 3; c_o = 256; s_h = 1; s_w = 1; group = 2
conv5W = tf.Variable(net_data["conv5"][0])
conv5b = tf.Variable(net_data["conv5"][1])
conv5_in = conv(conv4, conv5W, conv5b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
conv5 = tf.nn.relu(conv5_in)

#maxpool5
#max_pool(3, 3, 2, 2, padding='VALID', name='pool5')
k_h = 3; k_w = 3; s_h = 2; s_w = 2; padding = 'VALID'
maxpool5 = tf.nn.max_pool(conv5, ksize=[1, k_h, k_w, 1], strides=[1, s_h, s_w, 1], padding=padding)

#fc6
#fc(4096, name='fc6')
fc6W = tf.Variable(net_data["fc6"][0])
fc6b = tf.Variable(net_data["fc6"][1])
conv5_reshape = tf.reshape(maxpool5, [-1, fc6W.get_shape().as_list()[0]])
fc6 = tf.matmul(conv5_reshape, fc6W)
fc6 = tf.nn.bias_add(fc6, fc6b)
fc6 = tf.nn.relu(fc6)

#fc7
#fc(4096, name='fc7')
fc7W = tf.Variable(net_data["fc7"][0])
fc7b = tf.Variable(net_data["fc7"][1])
fc7 = tf.nn.relu_layer(fc6, fc7W, fc7b)

#fc8
#fc(1000, relu=False, name='fc8')
fc8W = tf.Variable(tf.random_normal([4096, 10], dtype=tf.float32, stddev=0.01))
fc8b = tf.Variable(tf.constant(1.0, shape=[10], dtype=tf.float32))
fc8 = tf.nn.xw_plus_b(fc7, fc8W, fc8b)

#prob
#softmax(name='prob'))
prob = tf.nn.softmax(fc8)

# source loss
source_prob = prob[:10,...]
source_predict = tf.cast(tf.arg_max(source_prob,dimension=1), tf.float32)
source_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(source_predict, y[:10]))

# MMD loss
fc6_MMD = tf.abs(tf.reduce_mean(fc6[:10])-
                 tf.reduce_mean(fc6[10:]))/num_source

fc7_MMD = tf.abs(tf.reduce_mean(fc7[:10]) -
                 tf.reduce_mean(fc7[10:])) / num_source

fc8_MMD = tf.abs(tf.reduce_mean(fc8[:10]) -
                 tf.reduce_mean(fc8[10:])) / num_source

MMD_loss = fc6_MMD + fc7_MMD +fc8_MMD

loss = source_loss

# train
train = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

# accuracy
predict = tf.cast(tf.argmax(prob[10:,...], 1),tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predict, y[10:]), tf.float32))

# session
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    batch_all = num_source // 50
    for epoch in range(100):
        start = 0
        batch_generator = create_batch(start,source_data, source_labels, target_data, target_labels, num_source, 10)

        for i in range(2):

            batch_img, batch_label = batch_generator.next()
            print ('epoch :%i, accuracy is : %f, loss:%f' % (i, sess.run(accuracy, feed_dict={x: batch_img,
                                                                                          y: batch_label}),
                                                         sess.run(loss, feed_dict={x: batch_img,
                                                                                   y: batch_label})))
            sess.run(train, feed_dict={x: batch_img, y: batch_label})

print 'train is finished'