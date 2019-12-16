import tensorflow.keras as keras
from tensorflow.keras import layers
from utils.helpers import *


class CnnModel(keras.Model):

    def __init__(self):
        super(CnnModel, self).__init__()
        self.patch_size = 16
        self.window_size = 72
        self.nb_channels = 3
        self.leak_alpha = 0.1
        self.dropout_prob = 0.25
        self.regularization_value = 1e-6
        self.nb_classes = 2
        self.batch_size = 256
        keras.backend.set_image_data_format('channels_last')
        self.create_model()
        

    def create_model(self):
        self.model = keras.Sequential()
        input_shape = (self.window_size, self.window_size, self.nb_channels)
        self.model.add(layers.InputLayer(input_shape))

        self.model.add(layers.Conv2D(64, 5, padding='same'))
        self.model.add(layers.LeakyReLU(alpha=self.leak_alpha))
        self.model.add(layers.MaxPool2D(padding = 'same'))
        self.model.add(layers.Dropout(self.dropout_prob))

        self.model.add(layers.Conv2D(128, 3, padding='same'))
        self.model.add(layers.LeakyReLU(alpha=self.leak_alpha))
        self.model.add(layers.MaxPool2D(padding='same'))
        self.model.add(layers.Dropout(self.dropout_prob))

        self.model.add(layers.Conv2D(256, 3, padding='same'))
        self.model.add(layers.LeakyReLU(alpha=self.leak_alpha))
        self.model.add(layers.MaxPool2D(padding='same'))
        self.model.add(layers.Dropout(self.dropout_prob))

        self.model.add(layers.Flatten())
        self.model.add(layers.Dense(128, kernel_regularizer=keras.regularizers.l2(self.regularization_value)))
        self.model.add(layers.LeakyReLU(alpha=self.leak_alpha))
        self.model.add(layers.Dropout(self.dropout_prob * 2))

        self.model.add(
            layers.Dense(self.nb_classes, kernel_regularizer=keras.regularizers.l2(self.regularization_value),
                         activation='softmax'))
        optimizer = keras.optimizers.Adam()


        self.model.compile(optimizer=optimizer, loss=keras.losses.binary_crossentropy,
                       metrics=['accuracy'])

        self.model.summary()

    def train_model(self, gt_imgs, tr_imgs, nb_epochs=100):

        np.random.seed(1)  # for reproducibility

        def generate_minibatch():
            """
            Procedure for real-time minibatch creation and image augmentation.
            This runs in a parallel thread while the model is being trained.
            """
            while True:
                # Generate one minibatch
                X_batch = np.empty((self.batch_size, self.window_size, self.window_size, 3))
                Y_batch = np.empty((self.batch_size, 2))
                for i in range(self.batch_size):
                    # Select a random image
                    idx = np.random.choice(tr_imgs.shape[0])
                    shape = tr_imgs[idx].shape

                    # Sample a random window from the image
                    center = np.random.randint(self.window_size // 2, shape[0] - self.window_size // 2, 2)

                    sub_image = tr_imgs[idx][center[0] - self.window_size // 2:center[0] + self.window_size // 2,
                                center[1] - self.window_size // 2:center[1] + self.window_size // 2]

                    gt_sub_image = gt_imgs[idx][center[0] - self.patch_size // 2:center[0] + self.patch_size // 2,
                                   center[1] - self.patch_size // 2:center[1] + self.patch_size // 2]

                    threshold = 0.25
                    label = (np.array([np.mean(gt_sub_image)]) > threshold) * 1
                    label = keras.utils.to_categorical(label, self.nb_classes)
                    X_batch[i] = sub_image
                    Y_batch[i] = label
                yield (X_batch, Y_batch)

        samples_per_epoch = tr_imgs.shape[0] * tr_imgs.shape[1] * tr_imgs.shape[2] / (self.patch_size**2 *self.batch_size)
        print("samples per epoch %d" % samples_per_epoch)

        # This callback reduces the learning rate when the training accuracy does not improve any more
        lr_callback = keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5, patience=1,
                                                        verbose=1, mode='auto', epsilon=0.0001, cooldown=0, min_lr=0)

        # Stops the training process upon convergence
        stop_callback = keras.callbacks.EarlyStopping(monitor='loss', min_delta=0.0001, patience=1, verbose=1,
                                                      mode='auto')

        try:
            self.model.fit_generator(generate_minibatch(),
                                     steps_per_epoch=samples_per_epoch,
                                     epochs=nb_epochs,
                                     verbose=1,
                                     callbacks=[lr_callback, stop_callback], workers=1)
        except KeyboardInterrupt:
            # Do not throw away the model in case the user stops the training process
            pass

        print('Training completed')
        save_weights('../weights_.h5')

    def save(self,
             filepath,
             overwrite=True,
             include_optimizer=True,
             save_format=None,
             signatures=None,
             options=None):
        self.model.save_weights(filepath)
        print("model saved !")

    def predict(self,
                x,
                batch_size=None,
                verbose=0,
                steps=None,
                callbacks=None,
                max_queue_size=10,
                workers=1,
                use_multiprocessing=False):
        X_patches = gen_patches(x, self.window_size,
                                self.patch_size)
        Y_pred = self.model.predict(X_patches)
        Y_pred = (Y_pred[:, 0] < Y_pred[:, 1]) * 1
        return group_patches(Y_pred,x.shape[0])
