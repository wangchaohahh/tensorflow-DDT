"""
this is a file to load large images and labels with
TFRecord type, which can be more effective
@author: wangchao
@date : 2016.12
"""

import os
from scipy.misc import imread, imresize
import tensorflow as tf


def create_record(filePath):
    """
    :param filePath: your dataset path
    :return: ...
    """
    writer = tf.python_io.TFRecordWriter("train.tfrecords")

    index = 0
    for datapath in os.listdir(filePath):
        for filename in os.listdir(filePath + '/' + datapath):
            filename = filePath + '/' + datapath + '/' + filename
            img = imread(filename)
            img = imresize(img, (28, 28))
            img_raw = img.tobytes()
            example = tf.train.Example(features=tf.train.Features(
                feature = {"label": tf.train.Feature(int64_list=tf.train.Int64List(value=[index])),
                           "img_raw":tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))}
            ))
            writer.write(example.SerializeToString())
        index += 1
    writer.close()
    print "create record is finished"


def read_and_decode(filename):
    filename_queue = tf.train.string_input_producer([filename])

    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'label': tf.FixedLenFeature([], tf.int64),
                                           'img_raw': tf.FixedLenFeature([], tf.string),
                                       })

    img = tf.decode_raw(features['img_raw'], tf.uint8)
    img = tf.reshape(img, [28, 28, 3])
    img = tf.cast(img, tf.float32) * (1. / 255) - 0.5
    label = tf.cast(features['label'], tf.int32)

    return img, label

img, labels = read_and_decode('train.tfrecords')

img_batch, label_batch = tf.train.shuffle_batch([img, labels],batch_size=30, capacity=1000,
                                                min_after_dequeue=500
                                                )
init = tf.initialize_all_variables()
with tf.Session() as sess:
    sess.run(init)
    threads = tf.train.start_queue_runners(sess=sess)
    for i in range(3):
        val, l= sess.run([img_batch, label_batch])
        print (val.shape, l)

