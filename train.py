
import os
import matplotlib.pyplot as plt

from src.unet_keras import *
from utils.helpers import *

# Loaded a set of images
PATH_ROOT = "/content/drive/My Drive/Road_Segmentation/"
PATH_TRAINING = PATH_ROOT + "data/training/"

IMG_SIZE = 400
NUM_CHANNELS = 3
NUM_FILTER = 32
FILTER_SIZE = 3

BATCH_SIZE = 8
NUM_EPOCHS = 1000


def main(argv=None):

    image_dir = PATH_TRAINING + "images/"
    gt_dir = PATH_TRAINING + "groundtruth/"

    files = os.listdir(image_dir)
    n = len(files)

    print("Loading " + str(n) + " images")
    imgs = [load_image(image_dir + files[i]) for i in range(n)]

    print("Loading " + str(n) + " groundtruth images")
    gt_imgs = [load_image(gt_dir + files[i]) for i in range(n)]

    x_train = np.asarray(imgs)
    y_train = np.expand_dims(np.asarray(gt_imgs), axis=3)

    # Create Model
    model = unet_model(IMG_SIZE, NUM_CHANNELS, NUM_FILTER, FILTER_SIZE, leaky=True, dropout=0.5)

    # Run Model
    model, f1_scores = train_model(model, x_train, y_train, BATCH_SIZE, NUM_EPOCHS)

    # Save the trained model
    print('Saving trained model')
    new_model_filename = 'unet_leaky_0val_50drop_1000epoch_evo.h5'
    model.save(PATH_ROOT + new_model_filename)

    plt.plot(f1_scores)
    plt.xlabel('# epochs')
    plt.ylabel('F1-Score')
    plt.title('F1-Score for every epochs')
    plt.savefig(PATH_ROOT + 'F1-Scores.png')


if __name__ == '__main__':
    tf.compat.v1.app.run()
