#!/usr/local/bin/python3
# -*- coding: utf-8 -*-

import os
import sys
import numpy as np
import tensorflow as tf
from xzc_tools import tools
import matplotlib.pyplot as plt

if __name__ == '__main__':
    try:
        #
        tf.set_random_seed(5)
        # np.random.seed(42)

        #
        batch_size = 50
        a1 = tf.Variable(tf.random_normal(shape=[1,1]), dtype=tf.float32)
        b1 = tf.Variable(tf.random_uniform(shape=[1,1]), dtype=tf.float32)
        a2 = tf.Variable(tf.random_normal(shape=[1,1]), dtype=tf.float32)
        b2 = tf.Variable(tf.random_uniform(shape=[1,1]), dtype=tf.float32)

        x = np.random.normal(2, 0.1, 500) # (均值，标准差，大小)
        x_data = tf.placeholder(shape=[None, 1], dtype=tf.float32)

        #
        sigmoid_activation = tf.sigmoid(tf.add(tf.matmul(x_data, a1), b1))
        relu_activation = tf.nn.relu(tf.add(tf.matmul(x_data, a2), b2))

        #
        loss_1 = tf.reduce_mean(tf.square(tf.subtract(sigmoid_activation, 0.75)))
        loss_2 = tf.reduce_mean(tf.square(tf.subtract(relu_activation, 0.75)))

        #
        train_step_1 = tf.train.GradientDescentOptimizer(0.01).minimize(loss_1)
        train_step_2 = tf.train.GradientDescentOptimizer(0.01).minimize(loss_2)

        #
        with tf.Session() as sess:
            init = tf.global_variables_initializer()
            sess.run(init)

            loss_sigmoid = []
            loss_relu = []
            activation_sigmoid = []
            activation_relu = []
            for i in range(750):
                rand_index = np.random.choice(len(x), size=batch_size)
                x_value = np.transpose([x[rand_index]])
                sess.run(train_step_1, feed_dict={x_data: x_value})
                sess.run(train_step_2, feed_dict={x_data: x_value})

                loss_sigmoid_temp = sess.run(loss_1, feed_dict={x_data:x_value})
                loss_relu_temp = sess.run(loss_2, feed_dict={x_data:x_value})
                loss_sigmoid.append(loss_sigmoid_temp)
                loss_relu.append(loss_relu_temp)

                activation_sigmoid_temp = sess.run(tf.reduce_mean(sess.run(sigmoid_activation, feed_dict={x_data:x_value})))
                activation_relu_temp = sess.run(tf.reduce_mean(sess.run(relu_activation, feed_dict={x_data:x_value})))
                activation_sigmoid.append(activation_sigmoid_temp)
                activation_relu.append(activation_relu_temp)

                if i % 50 ==0:
                    print('step:{0},s_loss:{1},r_loss:{2},s_acc:{3},r_acc:{4}'.format(
                        str(i), str(loss_sigmoid_temp), str(loss_relu_temp), str(activation_sigmoid_temp), str(activation_relu_temp)
                    ))

            plt.subplot(1,2,1)
            plt.plot(activation_sigmoid, 'k-', label='sigmoid activation')
            plt.plot(activation_relu, 'r--', label='relu activation')
            plt.ylim([0, 1.0])
            plt.xlim([0, 800])
            plt.title('activation outputs')
            plt.xlabel('generation')
            plt.ylabel('outputs')
            plt.legend(loc='lower right')

            plt.subplot(1,2,2)
            plt.plot(loss_sigmoid, 'k-', label='sigmoid loss')
            plt.plot(loss_relu, 'r--', label='relu loss')
            plt.ylim([0, 1.0])
            plt.xlim([0, 800])
            plt.title('loss per generation')
            plt.xlabel('generation')
            plt.ylabel('loss')
            plt.legend(loc='lower right')
            plt.show()

        with tf.Session() as sess:
            init = tf.global_variables_initializer()
            sess.run(init)
            print(sess.run(a1), sess.run(b1), sess.run(a2), sess.run(b2))

    except Exception as msg:
        tools.printInfo(2,msg)