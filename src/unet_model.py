import tensorflow as tf
import numpy as np
from utils.metrics import f1_scores


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


def unet(data, n_filter, filter_size, leaky, dropout):
    """Definition for the U-Net model"""

    conv1, pool1 = down_sample(data, n_filter, filter_size, leaky, dropout)
    conv2, pool2 = down_sample(pool1, n_filter * 2, filter_size, leaky, dropout)
    conv3, pool3 = down_sample(pool2, n_filter * 4, filter_size, leaky, dropout)
    conv4, pool4 = down_sample(pool3, n_filter * 8, filter_size, leaky, dropout)
    conv5 = conv_block(pool4, n_filter * 16, filter_size, leaky)

    up6 = tf.keras.layers.Conv2DTranspose(n_filter * 8,
                                          filter_size,
                                          (2, 2),
                                          padding='same',
                                          kernel_initializer='he_normal')(conv5)
    up6 = tf.keras.layers.concatenate([up6, conv4], axis=3)
    up7 = up_sample(up6, n_filter * 4, filter_size, leaky, dropout)
    up7 = tf.keras.layers.concatenate([up7, conv3], axis=3)
    up8 = up_sample(up7, n_filter * 2, filter_size, leaky, dropout)
    up8 = tf.keras.layers.concatenate([up8, conv2], axis=3)
    up9 = up_sample(up8, n_filter, filter_size, leaky, dropout)
    up9 = tf.keras.layers.concatenate([up9, conv1], axis=3)

    up9 = conv_block(up9, n_filter, filter_size, leaky)
    output = tf.keras.layers.Conv2D(filters=1, kernel_size=1, activation='sigmoid')(up9)

    return output


def unet_model(img_size, n_channel, n_filter, filter_size, leaky=False, dropout=None):
    inputs = tf.keras.layers.Input((img_size, img_size, n_channel))
    outputs = unet(inputs, n_filter, filter_size, leaky, dropout)
    model = tf.keras.Model(inputs=[inputs], outputs=[outputs])
    model.compile(optimizer=tf.keras.optimizers.Adam(),
                  loss='binary_crossentropy',
                  metrics=['Precision', 'Recall'])
    return model


def train_model(model, x_train, y_train, batch_size, n_epochs):
    # path_checkpoint = '/content/drive/My Drive/Road_Segmentation/weights.{epoch:02d}-{loss:.2f}.hdf5'
    # checkpoint = tf.keras.callbacks.ModelCheckpoint(path_checkpoint, monitor='loss', save_best_only=True, mode='min')

    history = model.fit(x_train, y_train, batch_size=batch_size, epochs=n_epochs, validation_split=0.0)
    print('\nhistory dict:', history.history)

    scores = f1_scores(history)

    return model, scores
