import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

#׼��ѵ������
x_data = np.linspace(-1, 1, 300, dtype=np.float64)[:, np.newaxis]
noise = np.random.normal(0, 0.05, x_data.shape).astype(np.float64)
y_data = 2 * np.power(x_data, 3) + np.power(x_data, 2) + noise

#����ռλ����
xs = tf.placeholder(tf.float32, [None, 1])
ys = tf.placeholder(tf.float32, [None, 1])

#�����񾭲㣺���ز��Ԥ���
# ����1
Weights1 = tf.Variable(tf.random_normal([1, 5]))
biases1 = tf.Variable(tf.zeros([1, 5]) + 0.1)
Wx_plus_b1 = tf.matmul(xs, Weights1) + biases1
l1 = tf.nn.relu(Wx_plus_b1)
# ����2
Weights2 = tf.Variable(tf.random_normal([5, 10]))
biases2 = tf.Variable(tf.zeros([1, 10]) + 0.1)
Wx_plus_b2 = tf.matmul(l1, Weights2) + biases2
l2 = tf.nn.relu(Wx_plus_b2)
# �����
Weights3 = tf.Variable(tf.random_normal([10, 1]))
biases3 = tf.Variable(tf.zeros([1, 1]) + 0.1)
prediction = tf.matmul(l2, Weights3) + biases3

#���� loss ���ʽ��������þ����mean squared error��
loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction), reduction_indices=[1]))

#ѵ�����Ż����ԣ�һ�����ݶ��½���AdamOptimizer�ȡ�.minimize(loss)���� loss �ﵽ��С��
train_step = tf.train.AdamOptimizer(0.1).minimize(loss)

# ��ʼ�����б���
init = tf.global_variables_initializer()
# ����Ự
with tf.Session() as sess:
    sess.run(init)
    # ����ԭʼx-yɢ��ͼ��
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.scatter(x_data, y_data)
    plt.ion()
    #plt.show()
    # �������� = 10000
    for i in range(10000):
        # ѵ��
        sess.run(train_step, feed_dict={xs: x_data, ys: y_data})
        # ÿ50����ͼ����ӡ�����
        if i % 50 == 0:
            # ���ӻ�ģ������Ľ����
            try:
                ax.lines.remove(lines[0])
            except Exception:
                pass
            prediction_value = sess.run(prediction, feed_dict={xs: x_data})
            # ����ģ��Ԥ��ֵ��
            lines = ax.plot(x_data, prediction_value, 'r-', lw=5)
    plt.show()
    # ��ӡ��ʧ
    print(sess.run(loss, feed_dict={xs: x_data, ys: y_data}))
