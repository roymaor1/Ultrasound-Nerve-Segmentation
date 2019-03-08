
import numpy as np
import tensorflow as tf
import glob
from natsort import natsorted
import cv2 as cv
import matplotlib.pyplot as plt

smooth = 1


def dice_coeff(y_true, y_pred):
    intersection = tf.reduce_sum(tf.multiply(y_true, y_pred))
    return (2. * intersection + smooth) / (tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) + smooth)


# create list of labeled images (matlab output)
im_list = natsorted(glob.glob('train_labeled/*.tif'))
num_of_files = 11270
label_list = [-1] * num_of_files
i = 1

# check file order
'''
for index in range(num_of_files):
    print(im_list[index])
'''

# create label matrix
for index in range(1, num_of_files, 2):
    label = int(im_list[index][-5:][:1])
    if label == 1:  # valid mask
        label_list[index-1] = 1  # valid
        label_list[index] = 0  # non-valid
    elif label == 0:  # black mask
        label_list[index-1] = 0  # valid
        label_list[index] = 1  # non-valid
    i = i+1

label_arr = np.array(label_list)
label_mat = np.reshape(label_arr, (int(num_of_files/2), 2)).astype(dtype=np.float32)

images_num = 11270
train_num = 9984

train_x = tf.placeholder(tf.float32, [None, 64*64])
train_x_reshaped = tf.reshape(train_x, [-1, 64, 64, 1])
train_y = tf.placeholder(tf.float32, [None, 64*64])
train_y_C = tf.placeholder(tf.float32, [None, 2])
step_size = tf.placeholder(tf.float32)

initializer = tf.contrib.layers.xavier_initializer()





W1_C = tf.Variable(initializer([3, 3, 1, 4]))
b1_C = tf.Variable(initializer([1]))
y1_C = tf.nn.relu(tf.nn.conv2d(train_x_reshaped, W1_C, strides=[1, 2, 2, 1], padding='SAME') + b1_C)

W2_C = tf.Variable(initializer([3, 3, 4, 8]))
b2_C = tf.Variable(initializer([1]))
y2_C = tf.nn.relu(tf.nn.conv2d(y1_C, W2_C, strides=[1, 2, 2, 1], padding='SAME') + b2_C)

W3_C = tf.Variable(initializer([3, 3, 8, 16]))
b3_C = tf.Variable(initializer([1]))
y3_C = tf.nn.relu(tf.nn.conv2d(y2_C, W3_C, strides=[1, 2, 2, 1], padding='SAME') + b3_C)
y3_C_reshaped = tf.reshape(y3_C, [-1, 8*8*16])

W4_C = tf.Variable(initializer([8*8*16, 16*16]))
b4_C = tf.Variable(initializer([16*16]))
y4_C = tf.nn.relu(tf.matmul(y3_C_reshaped, W4_C) + b4_C)

W5_C = tf.Variable(initializer([16*16, 2]))
b5_C = tf.Variable(initializer([2]))
y5_C = tf.nn.sigmoid(tf.matmul(y4_C, W5_C) + b5_C)

reg_C = tf.nn.l2_loss(W1_C) + tf.nn.l2_loss(W2_C) + tf.nn.l2_loss(W3_C) + tf.nn.l2_loss(W4_C) + tf.nn.l2_loss(W5_C)
loss_train_C = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=train_y_C, logits=y5_C) + 0.06*reg_C)
loss_test_C = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=train_y_C, logits=y5_C))
train_C = tf.train.RMSPropOptimizer(step_size).minimize(loss_train_C, var_list=[W1_C, b1_C, W2_C, b2_C, W3_C, b3_C, W4_C, b4_C, W5_C, b5_C])






W1 = tf.Variable(initializer([3, 3, 1, 16]))
b1 = tf.Variable(initializer([1]))
y1 = tf.nn.relu(tf.nn.conv2d(train_x_reshaped, W1, strides=[1, 1, 1, 1], padding='SAME') + b1)
y1_pool = tf.nn.max_pool(y1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

W2 = tf.Variable(initializer([3, 3, 16, 32]))
b2 = tf.Variable(initializer([1]))
y2 = tf.nn.relu(tf.nn.conv2d(y1_pool, W2, strides=[1, 1, 1, 1], padding='SAME') + b2)
y2_pool = tf.nn.max_pool(y2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

W3 = tf.Variable(initializer([3, 3, 32, 64]))
b3 = tf.Variable(initializer([1]))
y3 = tf.nn.relu(tf.nn.conv2d(y2_pool, W3, strides=[1, 1, 1, 1], padding='SAME') + b3)   # 32*32*32
y3_pool = tf.nn.max_pool(y3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')   # 16*16*32

W4 = tf.Variable(initializer([3, 3, 64, 128]))
b4 = tf.Variable(initializer([1]))
y4 = tf.nn.relu(tf.nn.conv2d(y3_pool, W4, strides=[1, 1, 1, 1], padding='SAME') + b4)   # 12*12*64

up5 = tf.image.resize_images(y4, [16, 16], method=tf.image.ResizeMethod.BILINEAR)
W5 = tf.Variable(initializer([3, 3, 128, 64]))
b5 = tf.Variable(initializer([1]))
y5 = tf.nn.relu(tf.nn.conv2d(up5, W5, strides=[1, 1, 1, 1], padding='SAME') + b5)

up6 = tf.image.resize_images(y5, [32, 32], method=tf.image.ResizeMethod.BILINEAR)
W6 = tf.Variable(initializer([3, 3, 64, 32]))
b6 = tf.Variable(initializer([1]))
y6 = tf.nn.relu(tf.nn.conv2d(up6, W6, strides=[1, 1, 1, 1], padding='SAME') + b6)

up7 = tf.image.resize_images(y6, [64, 64], method=tf.image.ResizeMethod.BILINEAR)
W7 = tf.Variable(initializer([3, 3, 32, 16]))
b7 = tf.Variable(initializer([1]))
y7 = tf.nn.relu(tf.nn.conv2d(up7, W7, strides=[1, 1, 1, 1], padding='SAME') + b7)

W8 = tf.Variable(initializer([3, 3, 16, 1]))
b8 = tf.Variable(initializer([1]))
y8 = tf.nn.sigmoid(tf.nn.conv2d(y7, W8, strides=[1, 1, 1, 1], padding='SAME') + b8)

y8_reshaped = tf.reshape(y8, [-1, 64*64])



loss_train = 1 - dice_coeff(train_y, y8_reshaped)
res_C = tf.placeholder(tf.float32, [None, 2])
res_C_float = tf.to_float(tf.argmin(res_C, 1))
res_C_reshaped = tf.reshape(res_C_float, [int((images_num - train_num) / 2), 1])
res_C_expanded = tf.tile(res_C_reshaped, [1, 64*64])
y8_test = tf.multiply(res_C_expanded, y8_reshaped)
y8_test = tf.cast(tf.greater(y8_test, 0.5), tf.float32)

loss_test = 1 - dice_coeff(train_y, y8_test)
train = tf.train.RMSPropOptimizer(step_size).minimize(loss_train, var_list=[W1, b1, W2, b2, W3, b3, W4, b4, W5, b5, W6, b6, W7, b7, W8, b8])

init = tf.global_variables_initializer()






with tf.Session() as sess:

    batch_size = 32
    step = 5e-5
    sess.run(init)
    raw_images_preprocessed = [cv.resize(cv.threshold(cv.imread(image, 0), 85, 255, cv.THRESH_TOZERO)[1], dsize=(64, 64)) for image in im_list]
    input_images = np.array([im.flatten() for im in raw_images_preprocessed], dtype=np.float32)




    input_x = input_images[::2]
    input_y = label_mat
    x_test = input_x[int(train_num/2):int(images_num/2)]
    y_test = input_y[int(train_num/2):int(images_num/2)]

    for epoch in range(500):
        for batch in range(int(train_num / (2*batch_size))):
            x = input_x[batch_size*batch:batch_size*batch+batch_size]
            y = input_y[batch_size*batch:batch_size*batch+batch_size]
            sess.run(train_C, feed_dict={train_x: x, train_y_C: y, step_size: step})

        if epoch % 25 == 0 or epoch == 499:
            print('epoch: %g' % epoch)
            entropy_loss = loss_train_C.eval(feed_dict={train_x: input_x[:int(train_num/2)], train_y_C: label_mat[:int(train_num/2)], step_size: step})
            print('entropy loss - train: %g' % entropy_loss)
            entropy_loss = loss_test_C.eval(feed_dict={train_x: x_test, train_y_C: y_test, step_size: step})
            print('entropy loss - test: %g' % entropy_loss)
            correct_prediction = tf.equal(tf.argmax(y5_C, 1), tf.argmax(train_y_C, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            accuracy_cal = sess.run(accuracy, feed_dict={train_x: x_test, train_y_C: y_test, step_size: step})
            print('accuracy: %g' % accuracy_cal)
            if accuracy_cal > 0.78:
                break

    pred = y5_C.eval(feed_dict={train_x: x_test, train_y_C: y_test})






    batch_size = 16
    step = 1e-5

    input_x = input_images[::2]
    input_y = input_images[1::2] / 255
    x_test = input_x[int(train_num/2):int(images_num/2)]
    y_test = input_y[int(train_num/2):int(images_num/2)]
    for run_try in range(5):
        sess.run(init)
        print('run number: %g' % run_try)
        for epoch in range(30):
            for batch in range(int(train_num / (2*batch_size))):
                x = input_x[batch_size*batch:batch_size*batch+batch_size]
                y = input_y[batch_size*batch:batch_size*batch+batch_size]
                sess.run(train, feed_dict={train_x: x, train_y: y, step_size: step})

            print('   epoch: %g' % epoch)
            print('   loss test round: %g' % loss_test.eval(feed_dict={train_x: x_test, train_y: y_test, res_C: pred}))

