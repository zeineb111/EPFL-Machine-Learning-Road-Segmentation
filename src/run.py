import tensorflow as tf
from utils.helpers import *
from cnn_model import CnnModel
import tensorflow.keras as keras
import sys

PATH_WEIGHTS = '../models/weights.h5'
PATH_UNET = '../models/unet.h5'
PATH_TEST_DATA = '../data/test_set_images/'
PATH_PREDICTION_DIR = '../data/predictions/'
PATH_SUBMISSION = '../final_submission.csv'
TEST_SIZE = 50


def main(argv):
    # We add all test images to an array, used later for generating a submission
    image_filenames_test = [PATH_TEST_DATA + 'test_' + str(i + 1) + '/' + 'test_' + str(i + 1) + '.png' for i
                            in range(TEST_SIZE)]
    image_filenames_predict = [PATH_PREDICTION_DIR + 'pred_' + str(i + 1) + '_unet.png' for i in range(TEST_SIZE)]
    print('Loading Model...')
    if len(argv) != 2:
        print(len(argv))
        raise Exception('Please pass only one argument to the script')
    else:
        if argv[1] == '-unet':

            # Run the UNET model
            model_unet = keras.models.load_model(PATH_UNET)
            gen_image_predictions_unet(model_unet, PATH_PREDICTION_DIR, *image_filenames_test)

            # Generates the submission
            generate_submission(model_unet, PATH_SUBMISSION, True, *image_filenames_predict)

        elif argv[1] == '-normal':
            # Run the normal CNN model
            model = CnnModel()
            model.load_weights(PATH_WEIGHTS)

            # Generates the submission
            generate_submission(model, PATH_SUBMISSION, False, *image_filenames_test)

        else:
            raise Exception('Please pass only "unet" or "normal" as argument to the script')


if __name__ == '__main__':
    main(sys.argv)
