# -*- coding: utf-8 -*-

import tensorflow as tf

from pre_alexnet import alexnet,load_initial_weights
from input_data_alex import read_and_decode
from mmd import mix_rbf_mmd2
from mmd import mix_rbf_mmd2_and_ratio
source_dir = './data/amazon.tfrecords'
target_dir = './data/dslr.tfrecords'
pretrained_weights = './data/bvlc_alexnet.npy'
checkpoint_path = 'checkpoint/'
learning_rate=0.003
num_epochs= 30000
batch_size=64
lamda=0.1
beta=1.0
# Network params
prob = 0.5
num_classes = 10
skip_layers = ['fc8','fc7']

# TF placeholder for graph input and output
keep_prob = tf.placeholder(tf.float32)

## train data
xs, ys = read_and_decode(source_dir, batch_size, is_batch=True)
xt, yt = read_and_decode(target_dir, batch_size, is_batch=True)
X_train = tf.concat([xs, xt],0)

## test data
X_test, Y_test = read_and_decode(target_dir, batch_size, is_batch=False)
X_test = tf.convert_to_tensor(X_test)
Y_test = tf.convert_to_tensor(Y_test)
## model
fc6, fc7, source_logits = alexnet(X_train,num_classes,keep_prob,is_reuse=False)
_,_,logits = alexnet(X_test[:300,...],num_classes,keep_prob,is_reuse=True)
source_fc6 = fc6[:batch_size,...]
target_fc6 = fc6[batch_size:,...]
source_fc7 = fc7[:batch_size,...]
target_fc7 = fc7[batch_size:,...]
## source loss
source_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                        logits = source_logits[:batch_size,...], labels = ys))

## MMD
bandwidths = [2.0, 5.0, 10.0, 20.0, 40.0, 80.0]
loss_fc6,var6,MMD_fc6 = mix_rbf_mmd2_and_ratio(source_fc6, target_fc6,sigmas=bandwidths)
loss_fc7,var7,MMD_fc7 = mix_rbf_mmd2_and_ratio(source_fc7, target_fc7,sigmas=bandwidths)
#loss_fc8,var8,MMD_fc8 = mix_rbf_mmd2_and_ratio(source_logits[:batch_size,...], source_logits[batch_size:,...],sigmas=bandwidths)
#MMD_fc8 = mix_rbf_mmd2_and_ratio(source_logits[:batch_size,...], source_logits[batch_size:,...],sigmas=bandwidths)
#MMD_fc6 = tf.reduce_mean(tf.square(tf.reduce_mean(source_fc6,0)
#                        -tf.reduce_mean(target_fc6,0)))
#MMD_fc7 = tf.reduce_mean(tf.square(tf.reduce_mean(source_fc7,0)
#                        -tf.reduce_mean(target_fc7,0)))

#MMD_fc8 = tf.reduce_mean(tf.square(tf.reduce_mean(source_logits[:batch_size,...],0)
#                                -tf.reduce_mean(source_logits[batch_size:,...],0)))
## discriminate distance loss
def source_distance(x,y):
    y = tf.cast(tf.argmax(y,axis=1),tf.float32)
    y1,_,_ = tf.unique_with_counts(y)
    TensorArr = tf.TensorArray(tf.float32,size=1, dynamic_size=True,clear_after_read=False)
    x_array = TensorArr.unstack(y1)
    size = x_array.size()
    initial_outputs = tf.TensorArray(dtype=tf.float32,size=size)
    i = tf.constant(0)
    def should_continue(i, *args):
        return i < size
    def loop(i,output):
        y_class = x_array.read(i)
        idx_i = tf.where(tf.equal(y,y_class))
        xi = tf.gather_nd(x,idx_i)
        initial_outputs1 = tf.TensorArray(dtype=tf.float32,size=size)
        j = tf.constant(0)
        def should_continue1(j,*args):
            return j<size
        def loop1(j,output1):
            y2=x_array.read(j)
            idx_j = tf.where(tf.equal(y,y2))
            xj = tf.gather_nd(x,idx_j)
            dis = tf.reduce_mean (tf.square(tf.reduce_mean(xi,0)
                        -tf.reduce_mean(xj,0)))
            output1 = output1.write(j,dis)
            return j+1,output1
        j,r1=tf.while_loop(should_continue1,loop1,[j,initial_outputs1])
        output = output.write(i,r1.stack())
        return i+1,output
    i,r = tf.while_loop(should_continue,loop,[i,initial_outputs])
    out = r.stack()
    return out
fc6_result = source_distance(source_fc6,ys)
fc6_dis = tf.reduce_mean(fc6_result)
fc7_result = source_distance(source_fc7,ys)
fc7_dis = tf.reduce_mean(fc7_result)
dis_all = fc6_dis + fc7_dis
dis_all = tf.sqrt(dis_all)
## loss
loss = tf.add( source_loss , lamda* (loss_fc6/(var6+beta*fc6_dis)+loss_fc7/(var7+beta*fc7_dis)))## accuracy
correct_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(Y_test[:300,...], 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

var2 = tf.trainable_variables()[6:]
## train
#train = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)
train = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss,var_list=var2)
## session
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

# Initialize an saver for store model checkpoints
#saver = tf.train.Saver()
#save_path = os.path.join(checkpoint_path, 'best_validation')

# Load the pretrained weights into the non-trainable layer
load_initial_weights(pretrained_weights,skip_layers,sess)

coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess=sess, coord=coord)

try:
    for step in range(num_epochs):
        if coord.should_stop():
            break
        sess.run([train],  feed_dict={keep_prob:prob})
        if (step+1)%50 == 0 or (step+1) == num_epochs:
            acc= sess.run([accuracy], feed_dict={keep_prob:1.0})
            msg = 'Epoch : {} '.format(step+1) + '  >>>>>>>  '+ 'Acc : {} '.format(acc)
            print(msg)
   # saver.save(sess, save_path=save_path)
except tf.errors.OutOfRangeError:
    print('done')
finally:
    coord.request_stop()
coord.join(threads)












