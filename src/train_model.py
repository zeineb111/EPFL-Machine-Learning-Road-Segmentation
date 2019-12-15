from src.utils.helpers import *
from src.CnnModel import CnnModel

data_dir = '../data/training/'
train_data_filename = data_dir + 'images/'
train_labels_filename = data_dir + 'groundtruth/'
num_images = 100


def main():
    cnn = CnnModel()
    imgs, gt_imgs = load_images(train_data_filename, train_labels_filename, num_images)
    imgs = preprocess_imgs(imgs, cnn.window_size, cnn.patch_size)
    gt_imgs = preprocess_imgs(gt_imgs, cnn.window_size, cnn.patch_size)
    # cnn.build((cnn.window_size,cnn.window_size,cnn.nb_channels))
    cnn.train_model(gt_imgs, imgs)
    # cnn.summary()
    cnn.save_weights('weights.h5')


if __name__ == '__main__':
    main()
