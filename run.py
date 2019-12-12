
import os

from src.unet_keras import *
from utils.helpers import *

# Loaded a set of images
root_dir = "data/training/"

image_dir = root_dir + "images/"
files = os.listdir(image_dir)
n = min(100, len(files)) # Load maximum 20 images
print("Loading " + str(n) + " images")
imgs = [load_image(image_dir + files[i]) for i in range(n)]
print(files[0])

gt_dir = root_dir + "groundtruth/"
print("Loading " + str(n) + " images")
gt_imgs = [load_image(gt_dir + files[i]) for i in range(n)]
print(files[0])

x = np.asarray(imgs)
y = np.expand_dims(np.asarray(gt_imgs), axis=3)

x_train = x[:50]
y_train = y[:50]

x_val = x[50:70]
y_val = y[50:70]

x_test = x[70:]
y_test = y[70:]

IMG_SIZE = 400
NUM_CHANNELS = 3
NUM_FILTER = 16
FILTER_SIZE = 3

BATCH_SIZE = 16
NUM_EPOCHS = 5

# Create Model
model = unet_model(IMG_SIZE, NUM_CHANNELS, NUM_FILTER, FILTER_SIZE)

# Run Model
model = train_model(model, x_train, y_train, x_val, y_val, BATCH_SIZE, NUM_EPOCHS)

# Evaluate the model on the test data using `evaluate`
print('\n# Evaluate on test data')
results = model.evaluate(x_test, y_test)
print('test loss, test acc:', results)

# Generate predictions (probabilities -- the output of the last layer)
# on new data using `predict`
print('\n# Generate predictions for 1 sample')
predictions = model.predict(x_test[0])
print('predictions shape:', predictions.shape)

print('My prediction: ', predictions)
