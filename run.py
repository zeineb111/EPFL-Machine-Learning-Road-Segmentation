import tensorflow as tf
from utils.helpers import *
from PIL import Image

PATH = '/content/drive/My Drive/Road_Segmentation'
PATH_MODEL = '/content/unet_model.h5'
PATH_DATA = PATH + '/data/training/images/'
PATH_PREDICTION_DIR = '/predictions'
PATH_SUBMISSION = PATH + '/submission.csv'


def main(argv=None):

    print('Loading Model...')
    model = tf.keras.models.load_model(PATH_MODEL)

    print('Loading Data...')
    data = load_data(PATH_DATA)

    images_filenames = []
    predictions = model.predict(data)

    for i in range(predictions.shape[0]):
        pimp = np.squeeze(predictions[i]).round()
        pimp = img_float_to_uint8(pimp)

        image_filename = PATH + PATH_PREDICTION_DIR + '/pred_' + str(i + 1) + '.png'
        Image.fromarray(pimp).save(image_filename)
        images_filenames.append(image_filename)


if __name__ == '__main__':
    tf.app.run()
