import time

import sys
import random
import numpy as np
import pandas as pd
import tensorflow as tf
from klangner import helpers, artificial

TRAIN_PATH = '../data/artificial-train/'
TEST_PATH = '../data/artificial-test/'
MODEL_PATH = '../model/artificial'

PIXEL_COUNT = 64 * 128
LABEL_COUNT = 4


def load_data(folder):
    df = pd.read_csv(folder + 'plates.csv')
    images = [helpers.load_image(folder + fname) for fname in df['image']]
    X_train = np.array(images)
    Y_train = df[['left', 'top', 'right', 'bottom']].as_matrix()
    return X_train, Y_train


# Mean square error
def mse(expected, predicted):
    se = tf.square(expected - predicted)
    return tf.reduce_mean(se)


def train(X_train, Y_train, X_test, Y_test, neural_net, epoch):
    X2_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1]*X_train.shape[2]))
    Y2_train = Y_train / (64, 32, 64, 32) - 1
    X2_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1]*X_test.shape[2]))
    Y2_test = Y_test / (64, 32, 64, 32) - 1
    y_placeholder = tf.placeholder(tf.float32, shape=[None, LABEL_COUNT])
    loss = mse(y_placeholder, neural_net.build())
    dataset = helpers.Dataset(X2_train, Y2_train)
    saver = tf.train.Saver()
    with tf.Session() as session:
        start_time = time.time()
        best_score = 1
        session.run(tf.initialize_all_variables())
        saver.restore(session, MODEL_PATH)
        train_step = tf.train.GradientDescentOptimizer(5e-5).minimize(loss)
        last_epoch = -1
        while dataset.epoch_completed() < epoch:
            (batch_x, batch_y) = dataset.next_batch(20)
            train_step.run(feed_dict={neural_net.x_placeholder: batch_x, y_placeholder: batch_y})
            if dataset.epoch_completed() > last_epoch:
                last_epoch = dataset.epoch_completed()
                score_test = loss.eval(feed_dict={neural_net.x_placeholder: X2_test, y_placeholder: Y2_test})
                if score_test < best_score:
                    best_score = score_test
                    saver.save(session, MODEL_PATH)
                    print('Epoch: %d, Score: %f saved' % (dataset.epoch_completed(), score_test))
                else:
                    print('Epoch: %d, Score: %f' % (dataset.epoch_completed(), score_test))

    dt = (time.time()-start_time) / 60
    eph = 60 * dataset.epoch_completed() / dt
    print('Trained %d epoch in %f minutes. %f epoches per hour' % (dataset.epoch_completed(), dt, eph))


def test(X_test, neural_net):
    X2_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1]*X_test.shape[2]))
    model = neural_net.build()
    saver = tf.train.Saver()
    with tf.Session() as session:
        saver.restore(session, MODEL_PATH)
        ids = [random.randint(0, X2_test.shape[0]) for _ in range(9)]
        Y2_test = model.eval(feed_dict={neural_net.x_placeholder: X2_test[ids]})
        helpers.plot_images(X_test[ids], (Y2_test+1) * (64, 32, 64, 32))


def main(epoch=0):
    X_train, Y_train = load_data(TRAIN_PATH)
    X_test, Y_test = load_data(TEST_PATH)
    network = artificial.CNN()
    if epoch > 0:
        train(X_train, Y_train, X_test, Y_test, network, epoch)
    else:
        test(X_test, network)
    print('Done.')


if __name__ == "__main__":
    if len(sys.argv) > 1:
        epoch_count = int(sys.argv[1])
        main(epoch_count)
    else:
        main()
