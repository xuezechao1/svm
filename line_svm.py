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
        # 导入数据
        iris = datasets.load_iris()
        X = np.array([[x[0],x[3]] for x in iris.data])
        Y = np.array([1 if y==0 else -1 for y in iris.target])

        # 分隔数据集为训练集和测试集
        train_index = np.random.choice(len(X), round(len(X) * 0.8), replace=False)
        test_index = np.array(list(set(range(len(X))) - set(train_index)))
        x_train = X[train_index]
        x_test = X[test_index]
        y_train = Y[train_index]
        y_test = Y[test_index]

        # 设置批量大小，占位符，模型变量
        batch_size = 100
        x_data = tf.placeholder(shape=[None, 2], dtype=tf.float32)
        y_data = tf.placeholder(shape=[None, 1], dtype=tf.float32)

        W = tf.Variable(tf.random_normal(shape=[2,1]))
        b=  tf.Variable(tf.random_normal(shape=[1,1]))
        C = tf.constant([9.0])

        # 声明模型输出
        model_output = tf.subtract(tf.matmul(x_data, W), b)

        # 声明损失函数
        l2_norm = tf.reduce_sum(tf.square(W))

        loss_1 = tf.reduce_mean(tf.maximum(0., tf.subtract(1., tf.multiply(model_output, y_data))))
        loss = tf.add(l2_norm, tf.multiply(C, loss_1))

        # 声明预测函数
        predict = tf.sign(model_output)
        accuracy = tf.reduce_mean(tf.cast(tf.equal(predict, y_data), tf.float32))

        # 声明优化器函数
        train_step = tf.train.GradientDescentOptimizer(0.01).minimize(loss)

        with tf.Session() as sess:
            init = tf.global_variables_initializer()
            sess.run(init)

            loss_value = []
            train_acc_value = []
            test_acc_value = []
            for i in range(1000):
                rand_index = np.random.choice(len(x_train), size=batch_size)
                rand_x = x_train[rand_index]
                rand_y = np.transpose([y_train[rand_index]])

                sess.run(train_step, feed_dict={x_data:rand_x, y_data:rand_y})

                temp_loss = sess.run(loss, feed_dict={x_data:rand_x, y_data:rand_y})
                loss_value.append(temp_loss)

                train_acc_temp = sess.run(accuracy, feed_dict={x_data:x_train, y_data:np.transpose([y_train])})
                train_acc_value.append(train_acc_temp)

                test_acc_temp = sess.run(accuracy, feed_dict={x_data:x_test, y_data:np.transpose([y_test])})
                test_acc_value.append(test_acc_temp)

                if i % 100 == 0:
                    print('step:{0}, loss:{1}, train_acc:{2}, test_acc:{3}'.format(
                        str(i), str(temp_loss), str(train_acc_temp), str(test_acc_temp)
                    ))

            [[a1],[a2]] = sess.run(W)
            [[b]] = sess.run(b)

            slope = -a2/a1
            y_intercept = b/a1

            x_value = X[:,1]

            pre_value = [slope * x + y_intercept for x in x_value]

            setosa_x = [x[1] for i,x in enumerate(X) if Y[i]==1]
            setosa_y = [x[0] for i, x in enumerate(X) if Y[i] == 1]
            no_setosa_x = [x[1] for i,x in enumerate(X) if Y[i]==-1]
            no_setosa_y = [x[0] for i, x in enumerate(X) if Y[i] == -1]

            plt.subplot(2,2,1)
            plt.plot(setosa_x, setosa_y, 'o', label='setosa')
            plt.plot(no_setosa_x, no_setosa_y, 'x', label='np_setosa')
            plt.plot(x_value, pre_value, 'r-', label='svm result', linewidth=3)
            plt.ylim(0, 10) # 设置坐标轴范围
            plt.legend(loc='lower right') # 显示位置
            plt.title('line svm')
            plt.xlabel('X')
            plt.ylabel('Y')

            plt.subplot(2,2,2)
            plt.plot(train_acc_value, 'k-', label='train accuracy')
            plt.plot(test_acc_value, 'r--', label='test accuracy')
            plt.title('train and test accuracy')
            plt.xlabel('generation')
            plt.ylabel('accuracy')
            plt.legend(loc='lower right')

            plt.subplot(2,2,3)
            plt.plot(loss_value, 'k-')
            plt.title('loss per generation')
            plt.xlabel('generation')
            plt.ylabel('loss')
            plt.show()
    except Exception as msg:
        tools.printInfo(2, msg)
        sys.exit()