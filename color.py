from keras.layers import Input, Dense, Flatten, Reshape, Conv2D, UpSampling2D, MaxPooling2D
from keras.models import Model, Sequential
from keras.datasets import mnist
from keras.callbacks import Callback
import random
import glob
import wandb
from wandb.keras import WandbCallback
import subprocess
import os
from PIL import Image
import numpy as np
import cv2
from keras import backend as K
from skimage import io, color

run = wandb.init(project='superres')
config = run.config

config.num_epochs = 1000
config.batch_size = 4





config.img_dir = "images"
config.input_height = 32
config.input_width = 32
config.output_height = 256
config.output_width = 256


val_dir = 'test'
train_dir = 'train'

# automatically get the data if it doesn't exist
if not os.path.exists("train"):
    print("Downloading flower dataset...")
    subprocess.check_output("curl https://storage.googleapis.com/l2kzone/flowers.tar | tar xz", shell=True)

# please don't change this function
def image_generator(batch_size, img_dir):
    """A generator that returns small images and large images"""
    image_filenames = glob.glob(img_dir + "/*")
    counter = 0
    while True:
        small_images = np.zeros((batch_size, config.input_width, config.input_height, 3))
        large_images = np.zeros((batch_size, config.output_width, config.output_height, 3))
        random.shuffle(image_filenames) 
        if ((counter+1)*batch_size>=len(image_filenames)):
              counter = 0
        for i in range(batch_size):
              img = Image.open(image_filenames[counter + i])
              small_images[i] = np.array(img.resize((config.input_width, config.input_height)))
              large_images[i] = np.array(img.resize((config.output_width, config.output_height)))
        yield (small_images, large_images)
        counter += batch_size

# please don't change this function
def perceptual_distance(y_true, y_pred):
    rmean = ( y_true[:,:,:,0] + y_pred[:,:,:,0] ) / 2;
    r = y_true[:,:,:,0] - y_pred[:,:,:,0]
    g = y_true[:,:,:,1] - y_pred[:,:,:,1]
    b = y_true[:,:,:,2] - y_pred[:,:,:,2]
    
    return K.mean(K.sqrt((((512+rmean)*r*r)/256) + 4*g*g + (((767-rmean)*b*b)/256)));

(val_small_images, val_large_images) = next(image_generator(145, val_dir))


model = Sequential()
model.add(Conv2D(3, (3, 3), activation='relu', padding='same', input_shape=(config.input_width,config.input_height, 3)))
model.add(UpSampling2D())
model.add(Conv2D(3, (3, 3), activation='relu', padding='same'))
model.add(UpSampling2D())
model.add(Conv2D(3, (3, 3), activation='relu', padding='same'))
model.add(UpSampling2D())
model.add(Conv2D(3, (3, 3), activation='relu', padding='same'))


# please don't change metrics=[perceptual_distance] 
model.compile(optimizer='adam', loss='mse', metrics=[perceptual_distance])


model.fit_generator( image_generator(config.batch_size, train_dir),
                     steps_per_epoch=2,
                     epochs=config.num_epochs, callbacks=[WandbCallback(data_type='image', predictions=16)],
                     validation_data=(val_small_images, val_large_images))


