
import os

from src.unet_keras import *
from utils.helpers import *

# Loaded a set of images
root_dir = "/content/drive/My Drive/Road_Segmentation/data/training/"

image_dir = root_dir + "images/"
files = os.listdir(image_dir)
n = len(files) # Load maximum 20 images
print("Loading " + str(n) + " images")
imgs = [load_image(image_dir + files[i]) for i in range(n)]
print(files[0])

gt_dir = root_dir + "groundtruth/"
print("Loading " + str(n) + " images")
gt_imgs = [load_image(gt_dir + files[i]) for i in range(n)]
print(files[0])

x_train = np.asarray(imgs)
y_train = np.expand_dims(np.asarray(gt_imgs), axis=3)

IMG_SIZE = 400
NUM_CHANNELS = 3
NUM_FILTER = 32
FILTER_SIZE = 3

BATCH_SIZE = 16
NUM_EPOCHS = 200

# Create Model
model = unet_model(IMG_SIZE, NUM_CHANNELS, NUM_FILTER, FILTER_SIZE, leaky=True)

# Run Model
model = train_model(model, x_train, y_train, BATCH_SIZE, NUM_EPOCHS)

# Save the trained model
print('Saving trained model')
new_model_filename = 'unet_leaky_0val_0drop_200epo.h5'
model.save(new_model_filename)
