import tensorflow as tf
from src.utils.helpers import *
from src.utils.submission.mask_to_submission import *
from src.cnn_model import CnnModel
from PIL import Image

PATH_WEIGHTS = '../weights.h5'
PATH_TEST_DATA = '../data/test_set_images/'
PATH_PREDICTION_DIR = '../predictions'
PATH_SUBMISSION = 'submission.csv'

TEST_SIZE = 50


def main(argv=None):
    print('Loading Model...')
    model = CnnModel()
    model.load_weights(PATH_WEIGHTS)

    # We add all test images to an array, used later for generating a submission
    image_filenames = []
    for i in range(1, 51):
        image_filenames = [PATH_TEST_DATA + 'test_' + str(i + 1) + '/' + 'test_' + str(i + 1) + '.png' for i
                           in range(TEST_SIZE)]

    # Set-up submission filename
    submission_filename = 'final_submission.csv'

    # Generates the submission
    generate_submission(model, submission_filename, *image_filenames)


if __name__ == '__main__':
    main()
