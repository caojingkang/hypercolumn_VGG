import math
import time
import os
import tensorflow as tf
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import input_data


def conv_op(input_op, name, kh, kw, n_out, dh, dw):
	n_in = input_op.get_shape()[-1].value
	with tf.variable_scope(name) as scope:
		kernel = tf.get_variable("w", shape=[kh, kw, n_in, n_out], dtype=tf.float32,
		                         initializer=tf.contrib.layers.xavier_initializer_conv2d())
		conv = tf.nn.conv2d(input_op, kernel, strides=[1, dh, dw, 1], padding='SAME')
		biases = tf.get_variable('b', shape=[n_out], dtype=tf.float32, initializer=tf.constant_initializer(0.0))
		activation = tf.nn.relu(tf.nn.bias_add(conv, biases), name=scope.name)
		return activation

def fc_op(input_op, name, n_out):
	n_in = input_op.get_shape()[-1].value
	with tf.variable_scope(name) as scope:
		kernel = tf.get_variable("w", shape=[n_in, n_out], dtype=tf.float32,
		                         initializer=tf.contrib.layers.xavier_initializer())
		biases = tf.get_variable('b', shape=[n_out], dtype=tf.float32, initializer=tf.constant_initializer(0.1))
		activation = tf.nn.relu_layer(input_op, kernel, biases, name=scope.name)
		return activation

def softmax_op(input_op, name, n_out):
	with tf.variable_scope(name) as scope:
		n_in = input_op.get_shape()[-1].value
		kernel = tf.get_variable('w', initializer=tf.truncated_normal(shape=[n_in, n_out], stddev=0.1), dtype=tf.float32)
		biases = tf.get_variable('b', shape=[n_out], dtype=tf.float32, initializer=tf.constant_initializer(0.1))
		activation = tf.nn.softmax(tf.matmul(input_op, kernel) + biases, name=scope.name)
		return activation

def maxpool_op(input_op, name, kh, kw, dh, dw):
	return tf.nn.max_pool(input_op, ksize=[1, kh, kw, 1], strides=[1, dh, dw, 1], padding='SAME', name=name)

def VGG16(image_holder, num_class, keep_prob, img_h, img_w):
	conv1_1 = conv_op(image_holder, name='conv1_1', kh=3, kw=3, n_out=64, dh=1, dw=1)
	conv1_2 = conv_op(conv1_1, name='conv1_2', kh=3, kw=3, n_out=64, dh=1, dw=1)
	pool1 = maxpool_op(conv1_2, name='pool1', kh=2, kw=2, dh=2, dw=2)
	img_h_2 = int(img_h / 2)
	img_w_2 = int(img_w / 2)

	conv2_1 = conv_op(pool1, name='conv2_1', kh=3, kw=3, n_out=128, dh=1, dw=1)
	conv2_2 = conv_op(conv2_1, name='conv2_2', kh=3, kw=3, n_out=128, dh=1, dw=1)
	pool2 = maxpool_op(conv2_2, name='pool2', kh=2, kw=2, dh=2, dw=2)
	img_h_3 = int(img_h_2 / 2)
	img_w_3 = int(img_w_2 / 2)

	conv3_1 = conv_op(pool2, name='conv3_1', kh=3, kw=3, n_out=256, dh=1, dw=1)
	conv3_2 = conv_op(conv3_1, name='conv3_2', kh=3, kw=3, n_out=256, dh=1, dw=1)
	conv3_3 = conv_op(conv3_2, name='conv3_3', kh=3, kw=3, n_out=256, dh=1, dw=1)
	pool3 = maxpool_op(conv3_3, name='pool3', kh=2, kw=2, dh=2, dw=2)
	img_h_4 = int(img_h_3 / 2)
	img_w_4 = int(img_w_3 / 2)

	conv4_1 = conv_op(pool3, name='conv4_1', kh=3, kw=3, n_out=512, dh=1, dw=1)
	conv4_2 = conv_op(conv4_1, name='conv4_2', kh=3, kw=3, n_out=512, dh=1, dw=1)
	conv4_3 = conv_op(conv4_2, name='conv4_3', kh=3, kw=3, n_out=512, dh=1, dw=1)
	pool4 = maxpool_op(conv4_3, name='pool4', kh=2, kw=2, dh=2, dw=2)
	img_h_5 = int(img_h_4 / 2)
	img_w_5 = int(img_w_4 / 2)

	conv5_1 = conv_op(pool4, name='conv5_1', kh=3, kw=3, n_out=512, dh=1, dw=1)
	conv5_2 = conv_op(conv5_1, name='conv5_2', kh=3, kw=3, n_out=512, dh=1, dw=1)
	conv5_3 = conv_op(conv5_2, name='conv5_3', kh=3, kw=3, n_out=512, dh=1, dw=1)
	pool5 = maxpool_op(conv5_3, name='pool5', kh=2, kw=2, dh=2, dw=2)
	img_h_6 = int(img_h_5 / 2)
	img_w_6 = int(img_w_5 / 2)

	resh = tf.concat([image_holder[:, img_h, img_w, :],
	                   conv1_1[:, img_h, img_w, :],
	                   conv1_2[:, img_h, img_w, :],
	                   pool1[:, img_h_2, img_w_2, :],
	                   conv2_1[:, img_h_2, img_w_2, :],
	                   conv2_2[:, img_h_2, img_w_2, :],
	                   pool2[:, img_h_3, img_w_3, :],
	                   conv3_1[:, img_h_3, img_w_3, :],
	                   conv3_2[:, img_h_3, img_w_3, :],
	                   conv3_3[:, img_h_3, img_w_3, :],
	                   pool3[:, img_h_4, img_w_4, :],
	                   conv4_1[:, img_h_4, img_w_4, :],
	                   conv4_2[:, img_h_4, img_w_4, :],
	                   conv4_3[:, img_h_4, img_w_4, :],
	                   pool4[:, img_h_5, img_w_5, :],
	                   conv5_1[:, img_h_5, img_w_5, :],
	                   conv5_2[:, img_h_5, img_w_5, :],
	                   conv5_3[:, img_h_5, img_w_5, :],
	                   pool5[:, img_h_6, img_w_6, :]], 1)

	fc6 = fc_op(resh, name='fc6', n_out=4096)

	fc6_drop = tf.nn.dropout(fc6, keep_prob=keep_prob, name='fc6_drop')

	fc7 = fc_op(fc6_drop, name='fc7', n_out=4096)
	fc7_drop = tf.nn.dropout(fc7, keep_prob=keep_prob, name='fc7_drop')

	fc8 = fc_op(fc7_drop, name='fc8', n_out=1000)
	fc8_drop = tf.nn.dropout(fc8, keep_prob=keep_prob, name='fc8_drop')
	softmax = softmax_op(fc8_drop, name='softmax', n_out=num_class)
	return softmax

def data_train_mine(tfrecord_path, model_path, img_size, iteration, batch_size):
	file = open('./my_model_saver/accuracy&loss.data', 'w')
	img, label = input_data.train_data_read_and_decode(tfrecord_path, img_size)
	img_batch, label_batch = tf.train.shuffle_batch(
		[img, label], batch_size=batch_size, capacity=5 * batch_size, min_after_dequeue=2 * batch_size
	)
	label_batch = tf.one_hot(indices=label_batch, depth=2, on_value=1.0, off_value=0.0)

	keep_prob = tf.placeholder('float')

	softmax = VGG16(img_batch, 2, keep_prob, int(img_size / 2 - 1), int(img_size / 2 - 1))

	cross_entropy = -tf.reduce_sum(label_batch * tf.log(softmax + 1e-10))
	cross_entropy_mean = tf.reduce_mean(cross_entropy)
	train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy_mean)
	correct_prediction = tf.equal(tf.argmax(softmax, 1), tf.argmax(label_batch, 1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))

	sess = tf.InteractiveSession()
	saver = tf.train.Saver()
	sess.run(tf.initialize_all_variables())
	threads = tf.train.start_queue_runners(sess=sess)

	for i in range(iteration):
		if i % 10 == 0:
			train_accuracy, loss = sess.run([accuracy, cross_entropy_mean], feed_dict={keep_prob: 1.0})
			file.write('%.3f %.10f\n' % (train_accuracy, loss))
			file.flush()
			print("step %d, training accuracy %.3f, loss %.10f" % (i, train_accuracy, loss))
		sess.run(train_step, feed_dict={keep_prob: 0.5})
		# print(sess.run(label_batch))
		# time.sleep(300)
	file.close()
	saver_path = saver.save(sess, model_path)

def single_img_test(ModelPath, tfrecordName, size, batch_size):
	path = './Pre-treatment/test_data/'
	img, axis_x, axis_y = input_data.test_data_read_and_decode(path + tfrecordName, size)
	# num_example = 0
	# for record in tf.python_io.tf_record_iterator(tfrecordPath):
	# 	num_example += 1
	string = tfrecordName[22:-9].split('-')
	real_height = int(string[0])
	real_width = int(string[1])
	num_example = int(real_height * real_width)
	iteration = int(num_example / batch_size) + 1
	img_batch, axisx, axisy = tf.train.batch(
		[img, axis_x, axis_y],
		batch_size=batch_size,
		capacity=batch_size * 2)
	image = np.zeros([real_height - 1 + size, real_width - 1 + size, 3], dtype=np.uint8)

	softmax = VGG16(img_batch, 2, 1, int(size / 2 - 1), int(size / 2 - 1))
	result = tf.one_hot(indices=tf.argmax(softmax, 1), depth=2, on_value=1.0, off_value=0.0)
	label = tf.argmax(result, 1)
	sess = tf.Session()
	sess.run(tf.initialize_all_variables())
	saver = tf.train.Saver()
	saver.restore(sess, ModelPath)
	threads = tf.train.start_queue_runners(sess=sess)
	for i in range(iteration):
		value, x, y = sess.run([label, axisx, axisy])
		for j in range(batch_size):
			if value[j] == 1:
				image[y[j] + int(size / 2 - 1), x[j] + int(size / 2 - 1)] = [255, 255, 255]
		print('step %.2f%%' % ((i + 1) / iteration * 100))
	cv2.imwrite(path + tfrecordName[0:20] + '_result' + '.bmp', image)

def img_totfrecord_and_test(test_data_path, ModelPath='./my_model_saver/model.ckpt', size=32, batch_size=100):
	tfrecordname = input_data.test_single_data_to_tfrecord(test_data_path, size=size)
	single_img_test(ModelPath=ModelPath, tfrecordName=tfrecordname, size=size, batch_size=batch_size)
	os.remove(test_data_path[0:26] + tfrecordname)

def img_loss_show():
	file = open('./my_model_saver/accuracy&loss.data')
	lines = file.readlines()
	plt.figure()
	accuracy = [float(i.split()[0]) for i in lines]
	loss = [float(i.split()[1]) for i in lines]
	a = []
	l = []
	for i in range(0, len(lines), 30):
		a.append(accuracy[i])
		l.append(loss[i])
	plt.plot(l, 'r-', linewidth=1)
	plt.xlabel('iterations(×30)')
	plt.ylabel('loss')
	plt.ylim(-10, 100)
	plt.show()

if __name__ == '__main__':
	# data_train_mine(tfrecord_path='./Pre-treatment/train_data/tfrecord',
	#                 model_path='./my_model_saver/model.ckpt',
	#                 img_size=32, iteration=60000, batch_size=100)
	# single_img_test(ModelPath='./my_model_saver/model.ckpt',
	#                     tfrecordName='2016-10-20 10-53-28_1-983-417.tfrecord',
	#                     size=32, batch_size=50)
	# img_totfrecord_and_test(test_data_path='./Pre-treatment/test_data/2016-10-14 04-55-13_2.bmp', ModelPath='./my_model_saver/model.ckpt',
	#              size=32, batch_size=100)
	img_loss_show()
