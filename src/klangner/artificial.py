import tensorflow as tf

PIXEL_COUNT = 64 * 128
LABEL_COUNT = 4


class CNN:

    def __init__(self, ):
        self.x_placeholder = tf.placeholder(tf.float32, shape=[None, PIXEL_COUNT])

    def build(self):
        x_image = tf.reshape(self.x_placeholder, [-1, 64, 128, 1])
        # Convolution Layer 1
        W_conv1 = self.weight_variable([3, 3, 1, 32])
        b_conv1 = self.bias_variable([32])
        h_conv1 = tf.nn.relu(self.conv2d(x_image, W_conv1) + b_conv1)
        h_pool1 = self.max_pool_2x2(h_conv1)
        # Convolution Layer 2
        W_conv2 = self.weight_variable([2, 2, 32, 64])
        b_conv2 = self.bias_variable([64])
        h_conv2 = tf.nn.relu(self.conv2d(h_pool1, W_conv2) + b_conv2)
        h_pool2 = self.max_pool_2x2(h_conv2)
        # Convolution Layer 3
        W_conv3 = self.weight_variable([2, 2, 64, 128])
        b_conv3 = self.bias_variable([128])
        h_conv3 = tf.nn.relu(self.conv2d(h_pool2, W_conv3) + b_conv3)
        h_pool3 = self.max_pool_2x2(h_conv3)
        # Dense layer 1
        h_pool3_flat = tf.reshape(h_pool3, [-1, 8*16*128])
        W_fc1 = self.weight_variable([8*16*128, 500])
        b_fc1 = self.bias_variable([500])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool3_flat, W_fc1) + b_fc1)
        # Dense layer 2
        W_fc2 = self.weight_variable([500, 500])
        b_fc2 = self.bias_variable([500])
        h_fc2 = tf.nn.relu(tf.matmul(h_fc1, W_fc2) + b_fc2)
        # Output layer
        W_out = self.weight_variable([500, LABEL_COUNT])
        b_out = self.bias_variable([LABEL_COUNT])
        return tf.matmul(h_fc2, W_out) + b_out

    @staticmethod
    def weight_variable(shape):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

    @staticmethod
    def bias_variable(shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)

    @staticmethod
    def conv2d(x, W):
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

    @staticmethod
    def max_pool_2x2(x):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

