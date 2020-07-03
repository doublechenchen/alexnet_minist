import os
from PIL import Image
from array import *
#
import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import time
import matplotlib.pyplot as plt
import cv2

dropout = 0.8 # Dropout

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
#

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

sess = tf.InteractiveSession()

x = tf.placeholder('float32', [None, 784])
y_ = tf.placeholder('float32', [None, 10])
x_image = tf.reshape(x, [-1, 28, 28, 1])
keep_prob = tf.placeholder(tf.float32)
W_conv1 = weight_variable([3, 3, 1, 32])
b_conv1 = bias_variable([32])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
lrn1 = tf.nn.lrn(h_conv1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='lrn1')
h_pool1 = max_pool_2x2(lrn1)
#
#
W_conv2 = weight_variable([3, 3, 32, 64])
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1 , W_conv2) + b_conv2)
lrn2 = tf.nn.lrn(h_conv2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='lrn1')
h_pool2 = max_pool_2x2(lrn2)
#
#
W_conv3 = weight_variable([3, 3, 64, 64])
b_conv3 = bias_variable([64])
h_conv3 = tf.nn.relu(conv2d(h_pool2 , W_conv3) + b_conv3)
#
W_conv4 = weight_variable([3, 3, 64, 72])
b_conv4 = bias_variable([72])
h_conv4 = tf.nn.relu(conv2d(h_conv3, W_conv4) + b_conv4)
#
W_conv5 = weight_variable([3, 3, 72, 96])
b_conv5 = bias_variable([96])
h_conv5 = tf.nn.relu(conv2d(h_conv4, W_conv5) + b_conv5)
#

W_fc1 = weight_variable([7 * 7 * 96, 1024])
#
b_fc1 = bias_variable([1024])
# #
h_pool2_flat = tf.reshape(h_conv5, [-1, 7 * 7 * 96])
#
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
#
W_fc2 = weight_variable([1024, 128])
b_fc2 = bias_variable([128])
h_fc2 = tf.nn.relu(tf.matmul(h_fc1, W_fc2) + b_fc2)
#
W_fc3 = weight_variable([128, 10])
b_fc3 = bias_variable([10])
y_conv = tf.nn.softmax(tf.matmul(h_fc2, W_fc3) + b_fc3)
# #
cross_entropy = -tf.reduce_sum(y_ * tf.log(y_conv))
#
train_step = tf.train.GradientDescentOptimizer(1e-5).minimize(cross_entropy)
#
# #
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float32"))
#
#
sess.run(tf.initialize_all_variables())
#
# #
mnist_data_set = input_data.read_data_sets('MNIST_data', one_hot=True)
c = []
#
#
start_time = time.time()
for i in range(2000):
#
    batch_xs, batch_ys = mnist_data_set.train.next_batch(128)
#
#     #
    if i % 2 == 0:
        train_accuracy = accuracy.eval(feed_dict={x: batch_xs, y_: batch_ys})
        c.append(train_accuracy)
        print("step %d, training accuracy %g" % (i, train_accuracy))
#
        end_time = time.time()
        print('time: ', (end_time - start_time))
        start_time = end_time
#     #
    train_step.run(feed_dict={x: batch_xs, y_: batch_ys})


FileList = []
data_image = array('B')
for dirname in os.listdir('./MNIST_data/valid_images')[0:]:

    path = os.path.join('./MNIST_data/valid_images',dirname)
    for filename in os.listdir(path):
        if filename.endswith('.png'):
            FileList.append(os.path.join(dirname, filename))

data_image = np.zeros([28,28])

for filename in FileList:
    label =(filename.split('/')[0])
    name = (filename.split('/')[1])
    Im = Image.open(os.path.join('./MNIST_data/valid_images', filename))
    image_arr = np.array(Im)
    image_arr = (255 - image_arr)/255
    width, height = Im.size
    for y_1 in range(0, height):
        for x_1 in range(0,width):
            data_image[x_1][y_1] = (image_arr[x_1,y_1])

    #data_Image = np.expand_dims(data_image,axis=0)
    data_Image = data_image.reshape(-1,784)
    yyy = np.zeros((1,10))
    yyy[0,int(label)]=1
    # print("Validating Result:", sess.run(y_conv, feed_dict={x: data_Image, y_: np.array([[0,0,0,0,1,0,0,0,0,0]]), keep_prob: 1.}))

    print("Validating Result:", sess.run(y_conv, feed_dict={x: data_Image, y_: yyy, keep_prob: 1.}))
    print(filename)

sess.close()
plt.plot(c)
