from utils.helpers import *
from cnn_model import CnnModel

data_dir = '../data/training/'
train_data_filename = data_dir + 'images/'
train_labels_filename = data_dir + 'groundtruth/'
PATH_WEIGHTS = '..models/weights.h5'
num_images = 1000
nb_epochs = 100


def main():
    cnn = CnnModel()
    imgs, gt_imgs = load_images(train_data_filename, train_labels_filename, num_images)
    history = cnn.train_model(gt_imgs, imgs, nb_epochs)
    cnn.save_weights(PATH_WEIGHTS)


if __name__ == '__main__':
    main()
