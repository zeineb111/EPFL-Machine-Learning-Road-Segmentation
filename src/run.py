import tensorflow as tf
from src.utils.helpers import *
from src.utils.submission.mask_to_submission import *
from src.CnnModel import CnnModel
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

    images_filenames = []

    predictions = [model.predict(load_image(PATH_TEST_DATA + 'test_' + str(i + 1) + '/' + 'test_' + str(i + 1))) for i
                   in range(TEST_SIZE)]
    try:
        # Create target Directory
        os.mkdir(PATH_PREDICTION_DIR)
        print('Directory ', '"predictions"', ' Created ')
    except FileExistsError:
        print('Directory ', '"predictions"', ' already exists')

    for i in range(len(predictions)):
        pimp = np.squeeze(predictions[i]).round()
        pimp = img_float_to_uint8(pimp, 3)

        image_filename = PATH_PREDICTION_DIR + 'pred_' + str(i + 1) + '.png'
        Image.fromarray(pimp).save(image_filename)
        images_filenames.append(image_filename)

    # Create submission
    submission_filename = 'submission.csv'
    masks_to_submission(submission_filename, *images_filenames)


if __name__ == '__main__':
    main()
