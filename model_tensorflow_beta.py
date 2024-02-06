import tensorflow as tf 
import os
import random
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from keras.models import Sequential
from keras.layers import Conv2D, Conv2DTranspose, Activation, BatchNormalization, ReLU
from keras.utils import load_img, img_to_array


def gaussian_blur(image, probability=0.5, max_dev=0.5):
    if random.random() < probability:
        standard_dev = random.uniform(0, max_dev)
        return tf.image.random_brightness(image, standard_dev)
    return image


class SISRDataSet(tf.data.Dataset):
    def __init__(self, hr_dir, lr_dir, training=False):
        self.hr_dir = hr_dir
        self.lr_dir = lr_dir
        self.hr_images = os.listdir(hr_dir)
        self.lr_images = os.listdir(lr_dir)
        self.training = training

    def read_images(self, hr_path, lr_path):
        hr_image = Image.open(hr_path).convert('L')
        lr_image = Image.open(lr_path).convert('L')
        if self.training:
            lr_image = gaussian_blur(lr_image)
        return np.array(hr_image), np.array(lr_image)
    
    def __call__(self):
        dataset = tf.data.Dataset.from_tensor_slices((self.hr_images, self.lr_images))
        dataset = dataset.map(lambda hr, lr: tf.py_function(self.read_images, [hr, lr], [tf.float32, tf.float32]))
        return dataset
    
class SISRCNN(tf.keras.Model):
    def __init__(self):
        super(SISRCNN, self).__init__()
        self.relu = ReLU()
        self.scale_factor = 2
        
        # Define layers
        self.conv1 = Conv2D(32, kernel_size=3, padding='same', activation='relu')
        self.conv2 = Conv2D(32, kernel_size=3, padding='same', activation='relu')
        self.conv3 = Conv2D(32, kernel_size=3, padding='same', activation='relu')
        self.conv4 = Conv2D(32, kernel_size=3, padding='same', activation='relu')
        self.conv5 = Conv2D(32, kernel_size=3, padding='same', activation='relu')
        self.conv6 = Conv2D(32 * (self.scale_factor ** 2), kernel_size=3, padding='same')
        self.conv7 = Conv2D(32, kernel_size=3, padding='same', activation='relu')
        self.conv8 = Conv2D(32, kernel_size=3, padding='same', activation='relu')
        self.conv9 = Conv2D(32, kernel_size=3, padding='same', activation='relu')
        self.conv10 = Conv2D(1, kernel_size=3, padding='same')

    def call(self, x):
        x = self.f1_compute(x)
        return self.f2_compute(x)
    
    def f1_compute(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))
        x = self.relu(self.conv5(x))
        x = tf.nn.depth_to_space(self.conv6(x), self.scale_factor)
        return x
    
    def f2_compute(self, f1_output):
        x = self.relu(self.conv7(f1_output))
        x = self.relu(self.conv8(x))
        x = self.relu(self.conv9(x))
        return self.conv10(x)
    
def load_and_process_image(image_path):
    image = load_img(image_path, color_mode='grayscale')
    image = img_to_array(image)
    image = tf.convert_to_tensor(image, dtype=tf.float32)
    image = image / 255.0 
    return image

def create_dataset(hr_dir, lr_dir):
    hr_image_files = [os.path.join(hr_dir, filename) for filename in sorted(os.listdir(hr_dir))]
    lr_image_files = [os.path.join(lr_dir, filename) for filename in sorted(os.listdir(lr_dir))]

    dataset = tf.data.Dataset.from_tensor_slices((lr_image_files, hr_image_files))
    dataset = dataset.map(lambda lr, hr: (load_and_process_image(lr), load_and_process_image(hr)),
                          num_parallel_calls=tf.data.experimental.AUTOTUNE)

    return dataset

# Usage
hr_dir = '.\\ProcessedImages\\HR'
lr_dir = '.\\ProcessedImages\\LR'
dataset = create_dataset(hr_dir, lr_dir)

# Optionally add shuffling, batching, and prefetching for training
dataset = dataset.shuffle(buffer_size=1000).batch(32).prefetch(tf.data.experimental.AUTOTUNE)


def main():
    epochs = 10
    batch_size = 32
    learning_rate = 0.001

    hr_dir = '.\\ProcessedImages\\HR'
    lr_dir = '.\\ProcessedImages\\LR'
    val_hr_dir = '.\\ProcessedImage\\HR\\Validation'
    val_lr_dir = '.\\ProcessedImage\\LR\\Validation'

    dataset = create_dataset(hr_dir, lr_dir)
    dataset = dataset.shuffle(buffer_size=1000).batch(32).prefetch(tf.data.experimental.AUTOTUNE)
    val_dataset = create_dataset(val_hr_dir, val_lr_dir)
    val_dataset = val_dataset.batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
    

    model = SISRCNN()

    model.compile(optimizer=tf.optimizers.Adam(learning_rate=learning_rate),
                  loss='mean_squared_error',  
                  metrics=['accuracy'])  
    
    model.fit(dataset, epoch=epochs, validation_data=val_dataset)


if __name__ == '__main__':
    main()