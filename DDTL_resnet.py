# -*- coding: utf-8 -*-

import tensorflow as tf

from pre_alexnet import alexnet,load_initial_weights
from input_data_alex import read_and_decode
from mmd import mix_rbf_mmd2
from mmd import mix_rbf_mmd2_and_ratio
import tensorflow.contrib.slim as slim
from tensorflow.contrib.slim.nets import resnet_v1
import numpy as np
## --------------parameters---------------------------------
source_dir='./data/webcam31.tfrecords'
target_dir='./data/amazon31.tfrecords'
pretrained_weights = './data/bvlc_alexnet.npy'
checkpoint_path = 'checkpoint/'
learning_rate=0.0003
num_epochs=50000
batch_size=32
lamda=1.3
beta=0.2
num_classes = 31
num_test=2817

# TF placeholder for graph input and output
#keep_prob = tf.placeholder(tf.float32)

## train data
xs, ys = read_and_decode(source_dir, batch_size, is_batch=True)
xt, yt = read_and_decode(target_dir, batch_size, is_batch=True)
X_train = tf.concat([xs, xt],0)

## test data
X_test, Y_test = read_and_decode(target_dir, batch_size, is_batch=False)
X_test = tf.convert_to_tensor(X_test)
Y_test = tf.convert_to_tensor(Y_test)

#------- resnet model ---------------
def resnet_model(image,reuse):
    with tf.variable_scope("model",reuse=reuse):
        with slim.arg_scope(resnet_v1.resnet_arg_scope()):
            outputs,_ = resnet_v1.resnet_v1_50(image)
            #outputs,_ = inception_resnet_v2(image)
            outputs = slim.flatten(outputs)
            outputs = slim.fully_connected(outputs,256)
            logits = slim.fully_connected(outputs,num_classes,activation_fn=None)
    return outputs,logits

# -------- train -----------------------------------
features,logits=resnet_model(X_train,reuse=False)
source_fc = features[:batch_size,...]
target_fc = features[batch_size:,...]
source_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                    logits = logits[:batch_size,...], labels = ys))

# --------- test ----------------------------------
#_,_,test_logit = resnet_model(X_test,reuse=True)
#correct_pred = tf.equal(tf.argmax(test_logit, 1), tf.argmax(Y_test, 1))
#accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

##---------- MMD -----------------------------------
bandwidths = [2.0, 5.0, 10.0, 20.0, 40.0, 80.0]
#bandwidths=[2.0]
#MMD_fc = mix_rbf_mmd2(source_fc,target_fc,sigmas=bandwidths)
loss_fc,var,MMD_fc = mix_rbf_mmd2_and_ratio(source_fc, target_fc,sigmas=bandwidths)
##---------- discriminate distance loss ------------
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
fc_result = source_distance(source_fc,ys)
fc_dis = tf.reduce_mean(fc_result)
dis_all = tf.sqrt(fc_dis)

##-------------loss-------------------------------------------------
loss = source_loss+lamda*loss_fc-beta*(dis_all+var)
#loss = loss*loss
##------------ train------------------------------------------------
# adaptive learning rate

#var1 = tf.trainable_variables()[:20]
var2 = tf.trainable_variables()[60:]
#print var2
global_step = tf.Variable(0, trainable=False)
learning_rate = tf.train.exponential_decay(learning_rate, global_step, 1000, 0.95)
#train = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss,var_list=var2)
#train = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(loss)
train = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(loss,var_list=var2,global_step=global_step)
#train = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(loss)
#train = tf.group(train1,train2)
#train = tf.train.AdadeltaOptimizer(learning_rate, 0.95, 1e-6).minimize(loss,var_list=var2)
##------------ session -------------------------------------------
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.allocator_type='BFC'
config.gpu_options.polling_inactive_delay_msecs=10
init = tf.global_variables_initializer()
sess = tf.Session(config=config)
sess.run(init)

##------------ restore --------------------------------------------
def name_in_checkpoint(var):
    if "model" in var.op.name:
        return var.op.name.replace("model/","")

variables_to_restore = slim.get_model_variables()
variables_to_restore = {name_in_checkpoint(var):var for var in variables_to_restore if 'resnet' in var.op.name}
#restorer = tf.train.Saver(variables_to_restore)
#restorer.restore(sess, "./data/resnet_v1_50.ckpt")
saver = tf.train.Saver()
coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess=sess, coord=coord)
ckpt= tf.train.get_checkpoint_state('./checkpoint')
saver.restore(sess,ckpt.model_checkpoint_path)
try:
    for step in range(num_epochs):
        if coord.should_stop():
            break
        _=sess.run([train])
        if (step+1)%200 == 0:
            i = 0
            y_guess = np.zeros(num_test, dtype=np.int)
            all_sum=0
            while i< num_test:
                j=min(i+1000,num_test)
                images_batch=X_test[i:j,...]
                #labels_batch=Y_test[i:j,...]
                _,test_logit = resnet_model(images_batch,reuse=True)
                test_logit = tf.nn.softmax(test_logit)
                test_logit = tf.argmax(test_logit,1)
                #correct=tf.equal(tf.argmax(test_logit,1),tf.argmax(labels_batch,1))
                #correct_sum = tf.reduce_sum(tf.cast(correct,tf.float32))
                #correct_sums = sess.run(correct_sum)
                pred,y_true=sess.run([test_logit,Y_test])
                y_guess[i:j]=pred
               # all_sum += correct_sums
                #cls_pred[i:j] = sess.run(predict_label)
                i=j
            #correct_pred = tf.equal(cls_pred, tf.argmax(Y_test[:num_test,...], 1))
            #accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
            #acc= sess.run([accuracy])
            #correct = (np.argmax(Y_test[:num_test,...],1) == cls_pred)
            #print (correct)
            #correct_sum = correct.sum()
            #acc = np.double(all_sum) / num_test
            acc=float(sum(y_guess == y_true))/len(y_guess)
            #msg = 'Epoch : {} '.format(step+1) + '  >>  '+'source_loss:{0:.5}'.format(sourceloss)+\
            #    ' >>  '+'dis_all:{0:.5}'.format(disall)+' >>  '+'mmd:{0:.6}'.format(mmd_)+\
            #    ' >>  '+'var:{0:.5}'.format(var_)+' >>  '+'loss:{0:.5}'.format(loss_all)+\
            #    ' >>  ' + 'Acc : {0:.5} '.format(acc)
            msg = 'Step:{}'.format(step+1)+ ' >>>  '+'Acc:{0:.5}'.format(acc)
            print(msg)
            saver.save(sess, './checkpoint/w2amodel',global_step=step)
except tf.errors.OutOfRangeError:
    print('done')
finally:
    coord.request_stop()
coord.join(threads)












