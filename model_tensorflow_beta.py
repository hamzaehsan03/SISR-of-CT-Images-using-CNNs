import tensorflow as tf 
import os
import random
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from keras.callbacks import Callback
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
        self.scale_factor = 4
        
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
    
def load_image(file_path):
    image = tf.io.read_file(file_path)
    image = tf.image.decode_png(image, channels=1)  
    image = tf.image.convert_image_dtype(image, tf.float32)
    return image


def create_dataset(hr_dir, lr_dir):
    hr_image_files = [os.path.join(hr_dir, filename) for filename in sorted(os.listdir(hr_dir))]
    lr_image_files = [os.path.join(lr_dir, filename) for filename in sorted(os.listdir(lr_dir))]

    dataset = tf.data.Dataset.from_tensor_slices((lr_image_files, hr_image_files))
    dataset = dataset.map(lambda lr, hr: (load_image(lr), load_image(hr)),
                          num_parallel_calls=tf.data.experimental.AUTOTUNE)

    return dataset

class SaveImage(Callback):
      
    def __init__(self, model, dataset, output_dir, num_images=3):
        super(SaveImage, self).__init__()
        self.model = model
        self.dataset = dataset
        self.output_dir = output_dir
        self.num_images = num_images

    def on_epoch_end(self, epoch, logs=None):
        # Create output directory if it doesn't exist
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        
        # Iterate over the dataset
        for i, (lr_images, hr_images) in enumerate(self.dataset.take(self.num_images)):
            # Generate super-resolved images
            sr_images = self.model.predict(lr_images)

            # Save LR, HR, SR images
            for j in range(len(lr_images)):
                lr_img = tf.squeeze(lr_images[j]).numpy()
                hr_img = tf.squeeze(hr_images[j]).numpy()
                sr_img = tf.squeeze(sr_images[j]).numpy()

                # Scale images to 0-255 range and convert to uint8
                lr_img = (lr_img * 255).astype('uint8')
                hr_img = (hr_img * 255).astype('uint8')
                sr_img = (sr_img * 255).astype('uint8')

                # Use PIL to save the images
                Image.fromarray(lr_img).save(os.path.join(self.output_dir, f'epoch_{epoch+1}_img_{i}_lr.png'))
                Image.fromarray(hr_img).save(os.path.join(self.output_dir, f'epoch_{epoch+1}_img_{i}_hr.png'))
                Image.fromarray(sr_img).save(os.path.join(self.output_dir, f'epoch_{epoch+1}_img_{i}_sr.png'))

def psnr_loss(y_true, y_pred):
    max_val = 1.0
    return -tf.image.psnr(y_true, y_pred, max_val)

def ssim_loss(y_true, y_pred):
    max_val = 1.0
    return 1 - tf.image.ssim(y_true, y_pred, max_val)

def combined_loss(y_true, y_pred):
    alpha = 0.84 # ssim
    beta = 0.16 # psnr
    combined = tf.keras.losses.MeanSquaredError()(y_true, y_pred) + alpha * ssim_loss(y_true, y_pred) + beta * psnr_loss(y_true, y_pred)

    return combined


def main():
    epochs = 10
    batch_size = 16
    learning_rate = 0.001

    hr_dir = '.\\ProcessedImages\\HR'
    lr_dir = '.\\ProcessedImages\\LR'
    val_hr_dir = '.\\ProcessedImages\\HR\\Validation'
    val_lr_dir = '.\\ProcessedImages\\LR\\Validation'

    dataset = create_dataset(hr_dir, lr_dir)
    dataset = dataset.shuffle(buffer_size=500).batch(16).prefetch(tf.data.experimental.AUTOTUNE)
    val_dataset = create_dataset(val_hr_dir, val_lr_dir)
    val_dataset = val_dataset.batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)

    model = SISRCNN()

    model.compile(optimizer=tf.optimizers.Adam(learning_rate=learning_rate),
                  loss='mean_squared_error',  
                  metrics=['accuracy'])  
    output_dir = './output_images'  
    save_images = SaveImage(model, dataset, output_dir)
    model.fit(dataset, epochs=epochs, validation_data=val_dataset, callbacks=[save_images])


if __name__ == '__main__':
    main()