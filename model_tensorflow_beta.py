import tensorflow as tf 
import os
import random
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
from keras.callbacks import Callback, ModelCheckpoint, LearningRateScheduler, CSVLogger
from keras.models import Sequential
from keras.layers import Conv2D, Conv2DTranspose, Activation, BatchNormalization, ReLU
from keras.utils import load_img, img_to_array


def gaussian_blur(image, kernel_size=5, sigma=1.0, probability=0.5):
    # Apply blur only with a certain probability
    if random.random() < probability:
        def gauss_kernel(channels, kernel_size, sigma):
            axis = tf.range(-kernel_size // 2 + 1.0, kernel_size // 2 + 1.0)
            x, y = tf.meshgrid(axis, axis)
            kernel = tf.exp(-(x**2 + y**2) / (2.0 * sigma**2))
            kernel = kernel / tf.reduce_sum(kernel)
            return kernel[:, :, tf.newaxis, tf.newaxis]

        gaussian_kernel = gauss_kernel(tf.shape(image)[-1], kernel_size, sigma)
        blurred_image = tf.nn.depthwise_conv2d(image[tf.newaxis, ...], gaussian_kernel, strides=[1, 1, 1, 1], padding="SAME")[0]
        return blurred_image
    else:
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
            lr_image = np.array(lr_image)
            lr_image = gaussian_blur(lr_image, probability=0.5)  # Apply Gaussian blur with 50% probability
            lr_image = Image.fromarray(lr_image.numpy().astype(np.uint8))
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
        #residual = x
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))
        x = self.relu(self.conv5(x))
        #x += residual
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
      
    def __init__(self, model, dataset, output_dir, num_images=3, ksize=3):
        super(SaveImage, self).__init__()
        self.model = model
        self.dataset = dataset
        self.output_dir = output_dir
        self.num_images = num_images
        self.ksize = ksize
    
    def apply_filters(self, image, median_ksize=3, unsharp_amount=1.5, unsharp_threshold=5):

        for i in range(2):
            image = cv2.medianBlur(image, median_ksize)
        
        # Apply unsharp mask to enhance edges
        gaussian_blurred = cv2.GaussianBlur(image, (0, 0), unsharp_threshold)
        unsharp_image = cv2.addWeighted(image, unsharp_amount, gaussian_blurred, -0.5, 0)

        return unsharp_image

    # def apply_filters(self, image, median_ksize=3, bilateral_d=5, bilateral_sigmaColor=50, bilateral_sigmaSpace=50):
    #     # Apply median filter
    #     image = cv2.medianBlur(image, median_ksize)
    #     # Apply bilateral filter
    #     image = cv2.bilateralFilter(image, bilateral_d, bilateral_sigmaColor, bilateral_sigmaSpace)
    #     return image

    def on_epoch_end(self, epoch, logs=None):
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        
        # Iterate over the dataset
        for i, (lr_images, hr_images) in enumerate(self.dataset.take(self.num_images)):
            sr_images = self.model.predict(lr_images)

            for j in range(len(lr_images)):
                lr_img = tf.squeeze(lr_images[j]).numpy()
                hr_img = tf.squeeze(hr_images[j]).numpy()
                sr_img = tf.squeeze(sr_images[j]).numpy()

                # Scale images to range [0, 255] and convert to uint8
                lr_img = (lr_img * 255).astype('uint8')
                hr_img = (hr_img * 255).astype('uint8')
                sr_img = (sr_img * 255).astype('uint8')

                # Apply median filter to SR image
                filtered_sr_img = self.apply_filters(sr_img, median_ksize=3, unsharp_amount=1.5, unsharp_threshold=5)
                #filtered_sr_img = self.apply_filters(sr_img, median_ksize=5, bilateral_d=9, bilateral_sigmaColor=75, bilateral_sigmaSpace=75)
                #filtered_sr_img = self.apply_median_filter(sr_img)

                # Save images
                Image.fromarray(lr_img).save(os.path.join(self.output_dir, f'epoch_{epoch+1}_img_{i}_lr.png'))
                Image.fromarray(hr_img).save(os.path.join(self.output_dir, f'epoch_{epoch+1}_img_{i}_hr.png'))
                Image.fromarray(sr_img).save(os.path.join(self.output_dir, f'epoch_{epoch+1}_img_{i}_sr.png'))
                Image.fromarray(filtered_sr_img).save(os.path.join(self.output_dir, f'epoch_{epoch+1}_img_{i}_sr_filtered.png'))

def scheduler(epoch, lr):
    if epoch < 10:
        return lr
    else:
        return lr * tf.math.exp(-0.1)


def psnr_metric(y_true, y_pred):
    return tf.reduce_mean(tf.image.psnr(y_true, y_pred, max_val=1.0))

def ssim_metric(y_true, y_pred):
    return tf.reduce_mean(tf.image.ssim(y_true, y_pred, max_val=1.0))

def psnr_loss(y_true, y_pred):
    max_val = 1.0
    return -tf.clip_by_value(tf.image.psnr(y_true, y_pred, max_val), -100, 100)

def ssim_loss(y_true, y_pred):
    max_val = 1.0
    return 1 - tf.image.ssim(y_true, y_pred, max_val)

def combined_loss(y_true, y_pred):
    alpha = 0.6  # Weight for SSIM
    mse_loss = tf.keras.losses.MeanSquaredError()(y_true, y_pred)
    ssim_loss_val = 1 - tf.image.ssim(y_true, y_pred, max_val=1.0)

    return mse_loss + alpha * ssim_loss_val

#     psnr_loss_val = -tf.image.psnr(y_true, y_pred, max_val=1.0)

#     return mse_loss + alpha * ssim_loss_val + beta * psnr_loss_val


def main():
    epochs = 40
    batch_size = 12
    learning_rate = 0.001

    hr_dir = '.\ProcessedImages\HR\Train'
    lr_dir = '.\ProcessedImages\LR\Train'
    val_hr_dir = '.\ProcessedImages\HR\Validation'
    val_lr_dir = '.\ProcessedImages\LR\Validation'

    dataset = create_dataset(hr_dir, lr_dir)
    dataset = dataset.shuffle(buffer_size=500).batch(8).prefetch(tf.data.experimental.AUTOTUNE)
    val_dataset = create_dataset(val_hr_dir, val_lr_dir)
    val_dataset = val_dataset.batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)

    lr_scheduler = LearningRateScheduler(scheduler)
    csv_logger = CSVLogger('training_log.csv', append=True, separator=';')


    model = SISRCNN()

    model.compile(optimizer=tf.optimizers.Adam(learning_rate=learning_rate),
                  loss=combined_loss,  
                  metrics=[psnr_metric, ssim_metric])
    
    checkpoint_path = "./model_checkpoints/model_epoch_{epoch:02d}"
    checkpoint = ModelCheckpoint(filepath=checkpoint_path,
                                 save_weights_only=False,
                                 verbose=1,
                                 save_best_only=False,
                                 save_freq='epoch',
                                 save_format='tf')  
    output_dir = './output_images'  
    save_images = SaveImage(model, dataset, output_dir)
    model.fit(dataset, epochs=epochs, validation_data=val_dataset, callbacks=[save_images, checkpoint, lr_scheduler, csv_logger]) # , callbacks=[save_images]


if __name__ == '__main__':
    main()