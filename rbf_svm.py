#!/usr/local/bin/python3
# -*- coding: utf-8 -*-

import os
import sys
import numpy as np
import tensorflow as tf
from xzc_tools import tools
import matplotlib.pyplot as plt
from sklearn import datasets

if __name__ == '__main__':
    try:
        #  生成模拟数据
        (X,Y) = datasets.make_circles(n_samples=500, factor=0.5, noise=0.1) # 生成环形数据  factor(外圈与内圈的尺度因子<1) noise(异常点的比例)
        Y = np.array([1 if y == 1 else -1 for y in Y])

        class_1_x = [x[0] for i,x in enumerate(X) if Y[i] == 1]
        class_1_y = [x[1] for i,x in enumerate(X) if Y[i] == 1]
        class_2_x = [x[0] for i,x in enumerate(X) if Y[i] == -1]
        class_2_y = [x[1] for i,x in enumerate(X) if Y[i] == -1]

        # 声明批量大小，占位符，创建模型变量
        batch_size = 500
        x_data = tf.placeholder(shape=[None, 2], dtype=tf.float32)
        y_data = tf.placeholder(shape=[None, 1], dtype=tf.float32)
        predict_grid = tf.placeholder(shape=[None, 2], dtype=tf.float32)
        alpha = tf.Variable(tf.random_normal(shape=[1, batch_size]))

        # 创建高斯函数
        gamma = tf.constant(-45.0)
        dist = tf.reduce_sum(tf.square(x_data), 1, keep_dims=True)
        sq_dists = tf.add(tf.subtract(dist, tf.multiply(2.0, tf.matmul(x_data, tf.transpose(x_data)))), tf.transpose(dist))
        svm_kernel = tf.exp(tf.multiply(gamma, tf.abs(sq_dists)))

        #
        loss_1 = tf.reduce_sum(alpha)
        b_vce_cross = tf.matmul(tf.transpose(alpha), alpha)
        y_data_cross = tf.matmul(y_data, tf.transpose(y_data))
        loss_2 = tf.reduce_sum(tf.multiply(svm_kernel, tf.multiply(b_vce_cross, y_data_cross)))
        loss = tf.negative(tf.subtract(loss_1, loss_2))

        # 创建预测函数，准确度函数
        rA = tf.reduce_sum(tf.square(x_data), 1, keep_dims=True)
        rB = tf.reduce_sum(tf.square(predict_grid), 1, keep_dims=True)

        pred_sq_dist = tf.add(tf.subtract(rA, tf.multiply(2.0, tf.matmul(x_data, tf.transpose(predict_grid)))), tf.transpose(rB))
        pred_kernel = tf.exp(tf.multiply(gamma, tf.abs(pred_sq_dist)))

        prediction_output = tf.matmul(tf.multiply(tf.transpose(y_data), alpha), pred_kernel)
        prediction = tf.sign(prediction_output + tf.divide(tf.subtract(tf.reduce_sum(y_data), tf.reduce_sum(prediction_output)), tf.to_float(tf.shape(y_data)[0])))
        accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.squeeze(prediction), tf.squeeze(y_data)), tf.float32))

        # 创建优化器
        train_step = tf.train.GradientDescentOptimizer(0.001).minimize(loss)

        with tf.Session() as sess:
            init = tf.global_variables_initializer()
            sess.run(init)

            loss_values = []
            batch_accuracy = []
            for i in range(2000):
                rand_index = np.random.choice(len(X), size=batch_size)
                rand_x = X[rand_index]
                rand_y = np.transpose([Y[rand_index]])

                sess.run(train_step, feed_dict={x_data:rand_x, y_data:rand_y})

                temp_loss = sess.run(loss, feed_dict={x_data:rand_x, y_data:rand_y})
                loss_values.append(temp_loss)

                acc_temp = sess.run(accuracy, feed_dict={x_data:rand_x, y_data:rand_y, predict_grid:rand_x})
                batch_accuracy.append(acc_temp)

                if i%100 == 0:
                    print('step:{0},loss:{1},accuracy:{2}'.format(
                        str(i),str(temp_loss),str(acc_temp)
                    ))

            x_min, x_max = X[:,0].min() -1, X[:,0].max() + 1
            y_min, y_max = X[:,1].min() -1, X[:,1].max() + 1
            xx,yy = np.meshgrid(np.arange(x_min, x_max, 0.02), np.arange(y_min, y_max, 0.02))
            grid_points = np.c_[xx.ravel(), yy.ravel()]
            [grid_predictions] = sess.run(prediction, feed_dict={x_data:rand_x, y_data:rand_y,predict_grid:grid_points})
            grid_predictions = grid_predictions.reshape(xx.shape)

            plt.subplot(2,2,1)
            plt.contourf(xx,yy,grid_predictions, cmap=plt.cm.Paired,alpha=0.8)
            plt.plot(class_1_x, class_1_y, 'ro', label='class 1')
            plt.plot(class_2_x, class_2_y, 'kx', label='class 2')
            plt.legend(loc='lower right')
            plt.ylim([-1.5, 1.5])
            plt.xlim([-1.5, 1.5])

            plt.subplot(2,2,2)
            plt.plot(batch_accuracy, 'k-', label='accuracy')
            plt.title('batch accuracy')
            plt.xlabel('generation')
            plt.ylabel('accuracy')
            plt.legend('lower eight')

            plt.subplot(2,2,3)
            plt.plot(loss_values, 'k-')
            plt.title('loss per generation')
            plt.xlabel('generation')
            plt.ylabel('loss')

            plt.show()
    except Exception as msg:
        tools.printInfo(2,msg)
        sys.exit()