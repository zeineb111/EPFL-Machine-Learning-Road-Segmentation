import tensorflow as tf


def activation(data, leaky):
    if leaky:
        return tf.keras.layers.LeakyReLU()(data)
    else:
        return tf.keras.layers.ReLU()(data)


def conv_block(data, n_filter, filter_size, leaky):
    conv1_inner = tf.keras.layers.Conv2D(n_filter,
                                         filter_size,
                                         padding='same',
                                         kernel_initializer='he_normal')(data)
    conv1_inner = tf.keras.layers.BatchNormalization()(conv1_inner)
    x1 = activation(conv1_inner, leaky)

    conv2_inner = tf.keras.layers.Conv2D(n_filter,
                                         filter_size,
                                         padding='same',
                                         kernel_initializer='he_normal')(x1)
    conv2_inner = tf.keras.layers.BatchNormalization()(conv2_inner)
    x2 = activation(conv2_inner, leaky)
    return x2


def down_sample(data, n_filter, filter_size, leaky, dropout):
    conv = conv_block(data, n_filter, filter_size, leaky)
    pool = tf.keras.layers.MaxPool2D((2, 2))(conv)
    if dropout is not None:
        pool = tf.keras.layers.Dropout(dropout)(pool)
    return conv, pool


def up_sample(data, n_filter, filter_size, leaky, dropout):
    conv = conv_block(data, n_filter, filter_size, leaky)
    up = tf.keras.layers.Conv2DTranspose(n_filter,
                                         filter_size,
                                         (2, 2),
                                         padding='same',
                                         kernel_initializer='he_normal')(conv)
    if dropout is not None:
        up = tf.keras.layers.Dropout(dropout)(up)
    return up


def unet(data, n_filter, filter_size, leaky=False, dropout=None):
    return None
