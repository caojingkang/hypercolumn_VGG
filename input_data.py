import tensorflow as tf
from PIL import Image
import random
import os

def train_data_to_tfrecord(train_data_path, tfrecord_path, scale_0, length):
	writer = tf.python_io.TFRecordWriter(tfrecord_path + '/train_data_0.tfrecord')
	list0 = os.listdir(train_data_path + '/0')
	list1 = os.listdir(train_data_path + '/1')
	random.shuffle(list0)
	random.shuffle(list0)
	random.shuffle(list1)
	random.shuffle(list1)
	list0 = list0[0 : len(list1) * scale_0]
	list0 = [train_data_path + '/0/' + i for i in list0]
	list1 = [train_data_path + '/1/' + i for i in list1]
	full_list = list0 + list1
	random.shuffle(full_list)
	random.shuffle(full_list)
	random.shuffle(full_list)
	i = 0
	for image in full_list:
		img = Image.open(image)
		img_raw = img.tobytes()
		label = int(image.split('/')[-2])
		example = tf.train.Example(features=tf.train.Features(feature={
			'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[label])),
			'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))
		}))
		writer.write(example.SerializeToString())
		i += 1
		if i % length == 0:
			writer.close()
			writer = tf.python_io.TFRecordWriter(tfrecord_path + '/train_data_' + str(int(i / length)) +'.tfrecord')
	writer.close()

def train_data_read_and_decode(file_dir, size):
	filename = os.listdir(file_dir)
	filename = [file_dir + '/' + i for i in filename]
	filename_queue = tf.train.string_input_producer(filename)
	reader = tf.TFRecordReader()
	_, serialized_example = reader.read(filename_queue) #返回文件名和文件
	features = tf.parse_single_example(serialized_example, features={
		'label': tf.FixedLenFeature([], tf.int64),
		'img_raw': tf.FixedLenFeature([], tf.string)
	})
	img = tf.decode_raw(features['img_raw'], tf.uint8)
	img = tf.reshape(img, [size, size, 3])
	img = tf.cast(img, tf.float32) * (1. / 255) - 0.5
	label = tf.cast(features['label'], tf.int32)
	return img, label

def test_single_data_to_tfrecord(image_path, size):
	img = Image.open(image_path)
	real_width = img.size[0] - size + 1
	real_height = img.size[1] - size + 1
	tfrecord_path = image_path[:-4] + '-' + str(real_height) + '-' + str(real_width) + '.tfrecord'
	writer = tf.python_io.TFRecordWriter(tfrecord_path)
	for width in range(0, real_width):
		for height in range(0, real_height):
			region = img.crop((width, height, width + size, height + size))
			img_raw = region.tobytes()
			example = tf.train.Example(features=tf.train.Features(feature={
				'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw])),
				'axis_x': tf.train.Feature(int64_list=tf.train.Int64List(value=[width])),
				'axis_y': tf.train.Feature(int64_list=tf.train.Int64List(value=[height]))
			}))
			writer.write(example.SerializeToString())
	writer.close()
	return tfrecord_path.split('/')[-1]

def test_data_read_and_decode(filename, size):
	filename_queue = tf.train.string_input_producer([filename])
	reader = tf.TFRecordReader()
	_, serialized_example = reader.read(filename_queue)
	features = tf.parse_single_example(serialized_example, features={
		'img_raw': tf.FixedLenFeature([], tf.string),
		'axis_x': tf.FixedLenFeature([], tf.int64),
		'axis_y': tf.FixedLenFeature([], tf.int64)
	})
	img = tf.decode_raw(features['img_raw'], tf.uint8)
	img = tf.reshape(img, [size, size, 3])
	img = tf.cast(img, tf.float32) * (1. / 255) - 0.5
	axis_x = tf.cast(features['axis_x'], tf.int32)
	axis_y = tf.cast(features['axis_y'], tf.int32)
	return img, axis_x, axis_y

if __name__ == '__main__':
	# train_data_to_tfrecord('./Pre-treatment/train_data', './Pre-treatment/train_data/tfrecord', scale_0=10, length=500000)
	test_single_data_to_tfrecord('./Pre-treatment/test_data/2016-10-20 10-53-28_1.bmp', 32)