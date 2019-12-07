import tensorflow as tf

SEED = 66478
NUM_CHANNELS = 3


def unet(data, n_filter, filter_size, in_channel, leaky=False, dropout=None):
    """Definition for the U-Net model"""

    def activation(x, leaky):
        if leaky:
            return tf.nn.relu(x)
        else:
            return tf.nn.leaky_relu(x)

    def conv_block(data, filter_size, n_filter, in_channel, leaky):
        conv1_weights = tf.Variable(
            tf.truncated_normal([filter_size, filter_size, in_channel, n_filter],
                                stddev=0.1,
                                seed=SEED))
        conv2_weights = tf.Variable(
            tf.truncated_normal([filter_size, filter_size, n_filter, n_filter],
                                stddev=0.1,
                                seed=SEED))

        conv1_inner = tf.nn.conv2d(data,
                                   conv1_weights,
                                   strides=[1, 1, 1, 1],
                                   padding='SAME')

        relu1 = activation(conv1_inner, leaky)

        conv2_inner = tf.nn.conv2d(relu1,
                                   conv2_weights,
                                   strides=[1, 1, 1, 1],
                                   padding='SAME')
        relu2 = activation(conv2_inner, leaky)

        return relu2

    def down_sample(data, n_filter, filter_size, in_channel, leaky, dropout):
        conv = conv_block(data, n_filter, filter_size, in_channel, leaky)
        pool = tf.nn.max_pool(conv,
                              ksize=[1, 2, 2, 1],
                              strides=[1, 2, 2, 1],
                              padding='SAME')
        if dropout is not None:
            pool = tf.nn.dropout(pool,
                                 keep_prob=dropout,
                                 seed=SEED)
        return conv, pool

    def up_sample(data, n_filter, filter_size, in_channel, leaky, dropout):
        conv = conv_block(data, n_filter, filter_size, in_channel, leaky)
        up = tf.nn.conv2d_transpose(conv,
                                    tf.Variable(
                                        tf.truncated_normal(
                                            [filter_size, filter_size, n_filter / 2, n_filter])),
                                    strides=(2, 2),
                                    padding='SAME')
        if dropout is not None:
            up = tf.nn.dropout(up,
                               keep_prob=dropout,
                               seed=SEED)
        return up

    conv1, pool1 = down_sample(data, n_filter, filter_size, in_channel, leaky, dropout)
    conv2, pool2 = down_sample(pool1, n_filter * 2, filter_size, n_filter, leaky, dropout)
    conv3, pool3 = down_sample(pool2, n_filter * 4, filter_size, n_filter * 2, leaky, dropout)
    conv4, pool4 = down_sample(pool3, n_filter * 8, filter_size, n_filter * 4, leaky, dropout)
    conv5, _ = down_sample(pool4, n_filter * 16, filter_size, n_filter * 8, leaky, dropout)

    up6 = tf.nn.conv2d_transpose(conv5,
                                 tf.Variable(
                                     tf.truncated_normal(
                                         [filter_size, filter_size, n_filter * 8, n_filter * 16])),
                                 strides=(2,2),
                                 padding='SAME'
                                 )
    up6 = tf.concat([up6, conv4], axis=3)

    up7 = up_sample(up6, n_filter * 8, filter_size, n_filter * 16, leaky, dropout)
    up7 = tf.concat([up7, conv3], axis=3)

    up8 = up_sample(up7, n_filter * 4, filter_size, n_filter * 8, leaky, dropout)
    up8 = tf.concat([up8, conv2], axis=3)

    up9 = up_sample(up8, n_filter * 2, filter_size, n_filter * 4, leaky, dropout)
    up9 = tf.concat([up9, conv1], axis=3)

    conv9 = conv_block(up9, filter_size, n_filter, n_filter * 2, leaky)

    output_weights = tf.Variable(
        tf.truncated_normal([1, 1, n_filter, 1],
                            stddev=0.1,
                            seed=SEED))
    output = tf.nn.conv2d(conv9,
                          output_weights,
                          strides=[1, 1, 1, 1],
                          padding='SAME')
    output = tf.math.sigmoid(output)

    return output
