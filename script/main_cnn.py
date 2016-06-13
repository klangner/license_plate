import time

import sys
import random
import numpy as np
import pandas as pd
import tensorflow as tf
from klangner import helpers, artificial

DATA_PATH = '../data/'
MODEL_PATH = '../model/artificial'

PIXEL_COUNT = 64 * 128
LABEL_COUNT = 4


def load_data(folder):
    df = pd.read_csv(folder + 'plates.csv')
    images = [helpers.load_image(folder + fname) for fname in df['image']]
    x_train = np.array(images)
    y_train = df[['left', 'top', 'right', 'bottom']].as_matrix()
    return x_train, y_train


# Mean square error
def mse(expected, predicted):
    se = tf.square(expected - predicted)
    return tf.reduce_mean(se)


def train(x_train, y_train, x_test, y_test, neural_net, epoch):
    x2_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1] * x_train.shape[2]))
    y2_train = y_train / (64, 32, 64, 32) - 1
    x2_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1] * x_test.shape[2]))
    y2_test = y_test / (64, 32, 64, 32) - 1
    y_placeholder = tf.placeholder(tf.float32, shape=[None, LABEL_COUNT])
    loss = mse(y_placeholder, neural_net.build())
    dataset = helpers.Dataset(x2_train, y2_train)
    saver = tf.train.Saver()
    with tf.Session() as session:
        start_time = time.time()
        best_score = 1
        session.run(tf.initialize_all_variables())
        saver.restore(session, MODEL_PATH)
        train_step = tf.train.GradientDescentOptimizer(5e-4).minimize(loss)
        last_epoch = -1
        while dataset.epoch_completed() < epoch:
            (batch_x, batch_y) = dataset.next_batch(20)
            train_step.run(feed_dict={neural_net.x_placeholder: batch_x, y_placeholder: batch_y})
            if dataset.epoch_completed() > last_epoch:
                last_epoch = dataset.epoch_completed()
                score_test = loss.eval(feed_dict={neural_net.x_placeholder: x2_test, y_placeholder: y2_test})
                score_train = loss.eval(feed_dict={neural_net.x_placeholder: x2_train[:1000],
                                                   y_placeholder: y2_train[:1000]})
                if score_test < best_score:
                    best_score = score_test
                    saver.save(session, MODEL_PATH)
                    print('Epoch=%d, Score=%f (train=%f) saved' % (dataset.epoch_completed(), score_test, score_train))
                else:
                    print('Epoch=%d, Score=%f (train=%f)' % (dataset.epoch_completed(), score_test, score_train))

    dt = (time.time()-start_time) / 60
    eph = 60 * dataset.epoch_completed() / dt
    print('Trained %d epoch in %f minutes. %f epoches per hour' % (dataset.epoch_completed(), dt, eph))


def test(x_test, neural_net):
    ids = [random.randint(0, x_test.shape[0]) for _ in range(9)]
    x2_test = np.reshape(x_test[ids], (len(ids), x_test.shape[1]*x_test.shape[2]))
    model = neural_net.build()
    saver = tf.train.Saver()
    with tf.Session() as session:
        saver.restore(session, MODEL_PATH)
        y2_test = model.eval(feed_dict={neural_net.x_placeholder: x2_test})
        helpers.plot_images(x_test[ids], (y2_test+1) * (64, 32, 64, 32))


def main(epoch, train_path):
    print('Train %d epochs on %s dataset' % (epoch, train_path))
    x_train, y_train = load_data(DATA_PATH + train_path + '/')
    x_test, y_test = load_data(DATA_PATH + 'artificial-test/')
    network = artificial.CNN()
    if epoch > 0:
        train(x_train, y_train, x_test, y_test, network, epoch)
    else:
        test(x_test, network)
    print('Done.')


if __name__ == "__main__":
    epoch_count = 0
    train_dataset = 'artificial-train-1'
    if len(sys.argv) > 1:
        epoch_count = int(sys.argv[1])
    if len(sys.argv) > 2:
        train_dataset = sys.argv[2]
    main(epoch_count, train_dataset)
