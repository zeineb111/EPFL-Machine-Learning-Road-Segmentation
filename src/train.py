from utils.helpers import *
from cnn_model import CnnModel
from utils.metrics import f1_scores

data_dir = '../data/training/'
train_data_filename = data_dir + 'images/'
train_labels_filename = data_dir + 'groundtruth/'
PATH_WEIGHTS = '..models/weights.h5'
num_images = 1000
nb_epochs = 100


def main():
    # Instanciate the model
    cnn = CnnModel()
    # Load data
    tr_imgs, gt_imgs = load_images(train_data_filename, train_labels_filename, num_images)
    # Train the model
    history = cnn.train_model(gt_imgs, tr_imgs, nb_epochs)
    # Save the weights
    cnn.save_weights(PATH_WEIGHTS)

    # Generate plots
    if history is not None:
        plot_metric_history(f1_scores=f1_scores(history))


if __name__ == '__main__':
    main()
