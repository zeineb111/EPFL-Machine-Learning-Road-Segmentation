import tensorflow as tf
from utils.helpers import *
from utils.mask_to_submission import *
from PIL import Image

PATH = '/content/drive/My Drive/Road_Segmentation'
PATH_MODEL = '/content/unet_leaky_0val_0drop_200epo.h5'
PATH_DATA = PATH + '/data/test_set_images/'
PATH_PREDICTION_DIR = '/predictions'
PATH_SUBMISSION = PATH + '/submission.csv'

TEST_SIZE = 50


def main(argv=None):

    print('Loading Model...')
    model = tf.keras.models.load_model(PATH_MODEL)

    images_filenames = []

    predictions = []
    for i in range(TEST_SIZE):
        path_folder = PATH_DATA + '/test_' + str(i + 1)
        img = load_data(path_folder, test=True)
        prediction = img_predict(np.squeeze(img), model)
        predictions.append(prediction)

    try:
        # Create target Directory
        os.mkdir(PATH + PATH_PREDICTION_DIR)
        print('Directory ', '"predictions"', ' Created ')
    except FileExistsError:
        print('Directory ', '"predictions"', ' already exists')

    for i in range(len(predictions)):
        pimp = np.squeeze(predictions[i]).round()
        pimp = img_float_to_uint8(pimp)

        image_filename = PATH + PATH_PREDICTION_DIR + '/pred_' + str(i + 1) + '.png'
        Image.fromarray(pimp).save(image_filename)
        images_filenames.append(image_filename)

    # Create submission
    submission_filename = 'submission.csv'
    masks_to_submission(submission_filename, *images_filenames)


if __name__ == '__main__':
    tf.app.run()
