# -*- coding: utf-8 -*
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import argparse
import sys
import tempfile
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import mnist_model

FLAGS = None
def main(_):
  mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)
  #���������mnistͼƬ��СΪ28*28
  x = tf.placeholder(tf.float32, [None, 784])
  #���������������1-10
  y_ = tf.placeholder(tf.float32, [None, 10])
  # �������磬���롪>��һ������>��һ��ػ���>�ڶ�������>�ڶ���ػ���>��һ��ȫ���ӡ�>�ڶ���ȫ����
  y_conv, keep_prob = mnist_model.deepnn(x)
  #��һ�����������һ��������һ��softmax���ڶ�����softmax�����ʵ��������һ��������
  #cross_entropy���ص�������
  with tf.name_scope('loss'):
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y_,                                                            logits=y_conv)
  #��cross_entropy������ƽ��ֵ�õ�������
  cross_entropy = tf.reduce_mean(cross_entropy)
  #AdamOptimizer��Adam�Ż��㷨��һ��Ѱ��ȫ�����ŵ���Ż��㷨��������η��ݶ�У��
  with tf.name_scope('adam_optimizer'):
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
  #�ڲ��Լ��ϵľ�ȷ��
  with tf.name_scope('accuracy'):
    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
    correct_prediction = tf.cast(correct_prediction, tf.float32)
  accuracy = tf.reduce_mean(correct_prediction)
  #��������ͼģ�ͱ��汾�أ�����ͨ��������鿴���ӻ�����ṹ
  graph_location = tempfile.mkdtemp()
  print('Saving graph to: %s' % graph_location)
  train_writer = tf.summary.FileWriter(graph_location)
  train_writer.add_graph(tf.get_default_graph())

  #��ѵ�������籣������
  saver = tf.train.Saver()
  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(5000):
      batch = mnist.train.next_batch(50)
      if i % 100 == 0:
        train_accuracy = accuracy.eval(feed_dict={
            x: batch[0], y_: batch[1], keep_prob: 1.0})#�������ֵ䣬��ʾtensorflow��feed��ֵ
        print('step %d, training accuracy %g' % (i, train_accuracy))
      train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

    test_accuracy = 0
    for i in range(200):
      batch = mnist.test.next_batch(50)
      test_accuracy += accuracy.eval(feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0}) / 200;
    print('test accuracy %g' % test_accuracy)
    save_path = saver.save(sess,"mnist_cnn_model.ckpt")

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--data_dir', type=str, default='./', help = 'Directory for storing input data')
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
