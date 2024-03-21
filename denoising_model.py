import os
import tensorflow as tf
from keras.models import Model
from keras import layers, models
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, Callback
import matplotlib.pyplot as plt
import numpy as np

class DenoiseCallback(Callback):
    def __init__(self, dataset, output_dir, freq=1):
        super(DenoiseCallback, self).__init__()
        self.dataset = dataset  # A tf.data.Dataset object
        self.output_dir = output_dir
        self.freq = freq  # Frequency in epochs to save the denoised image
        os.makedirs(output_dir, exist_ok=True)  # Ensure the output directory exists

    def on_epoch_end(self, epoch, logs=None):
        if epoch % self.freq == 0:
            # Take one batch from the dataset
            for img_batch, _ in self.dataset.take(1):
                img_to_denoise = img_batch[0:1]  # Take the first image of the batch
                break
            
            # Denoise the image
            denoised_img = self.model.predict(img_to_denoise)
            # Remove batch dimension and scale back to [0, 255]
            denoised_img = np.squeeze(denoised_img) * 255.0
            denoised_img = denoised_img.astype(np.uint8)
            # Save the denoised image
            output_path = os.path.join(self.output_dir, f"denoised_epoch_{epoch+1}.png")
            plt.imsave(output_path, denoised_img, cmap='gray')
            print(f"Denoised image saved to: {output_path}")


def DnCNN(depth=8, filters=32, image_channels=1):
    """
    Creates a DnCNN model based on the specified parameters.

    Parameters:
    - depth: Number of convolutional layers.
    - filters: Number of filters in each convolutional layer.
    - image_channels: Number of image input channels.

    Returns:
    A Keras model representing the DnCNN architecture.
    """
    input_shape = (None, None, image_channels)  # Dynamic spatial dimensions, fixed channel dimension
    inputs = tf.keras.Input(shape=input_shape)

    # Initial convolutional layer without Batch Normalization
    x = layers.Conv2D(filters, kernel_size=3, padding='same', activation='relu')(inputs)
    
    # Intermediate layers with Batch Normalization and ReLU activation
    for _ in range(depth-2):
        x = layers.Conv2D(filters, kernel_size=3, padding='same', use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
    
    # Final convolutional layer without activation to output the denoised image
    outputs = layers.Conv2D(image_channels, kernel_size=3, padding='same')(x)

    # Define the model
    model = models.Model(inputs=inputs, outputs=outputs)
    
    return model

dncnn_model = DnCNN(image_channels=1)
dncnn_model.summary()

sr_dir = './SuperResolvedImages'
hr_dir = './ProcessedImages/HR/Train'

sr_image_filenames = sorted(os.listdir(sr_dir))
hr_image_filenames = sorted(os.listdir(hr_dir))

assert all(sr.lstrip('SR_') == hr for sr, hr in zip(sr_image_filenames, hr_image_filenames)), "Filenames do not match."

sr_image_paths = [os.path.join(sr_dir, filename) for filename in sr_image_filenames]
hr_image_paths = [os.path.join(hr_dir, filename.lstrip('SR_')) for filename in hr_image_filenames]

def load_image(file_path):
    # Load the image file
    img = tf.io.read_file(file_path)
    img = tf.image.decode_png(img, channels=1)  
    img = tf.image.convert_image_dtype(img, tf.float32)  # Convert to float32
    return img

def load_image_pair(sr_path, hr_path):
    sr_img = load_image(sr_path)
    hr_img = load_image(hr_path)
    return sr_img, hr_img

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
    alpha = 0.5  # Weight for SSIM
    mse_loss = tf.keras.losses.MeanSquaredError()(y_true, y_pred)
    ssim_loss_val = 1 - tf.image.ssim(y_true, y_pred, max_val=1.0)

    return mse_loss + alpha * ssim_loss_val

dataset = tf.data.Dataset.from_tensor_slices((sr_image_paths, hr_image_paths))
dataset = dataset.map(load_image_pair)  # Load the actual image pairs

# Shuffle and split the dataset here
DATASET_SIZE = len(sr_image_paths)
TRAIN_SIZE = int(0.8 * DATASET_SIZE)
VAL_SIZE = DATASET_SIZE - TRAIN_SIZE

dataset = dataset.shuffle(buffer_size=DATASET_SIZE)  # Shuffle the dataset

train_dataset = dataset.take(TRAIN_SIZE)
val_dataset = dataset.skip(TRAIN_SIZE)

# Batch and prefetch both datasets
train_dataset = train_dataset.batch(8).prefetch(tf.data.experimental.AUTOTUNE)
val_dataset = val_dataset.batch(8).prefetch(tf.data.experimental.AUTOTUNE)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss', 
    factor=0.2,  # Reduction factor; new_lr = lr * factor
    patience=5, 
    min_delta=1e-4, 
    verbose=1
)

output_dir = './DenoisedImages'

denoise_callback = DenoiseCallback(dataset=train_dataset, output_dir=output_dir, freq=1)



# Define the DnCNN model
dncnn_model = DnCNN(image_channels=1)

early_stopper = EarlyStopping(
    monitor='val_loss', 
    patience=20,  # Increased patience
    min_delta=1e-4,  # Smaller threshold for improvement
    verbose=1, 
    restore_best_weights=True
)
# Compile the model with an optimizer and loss function appropriate for denoising
dncnn_model.compile(optimizer='adam', loss=combined_loss, metrics=[psnr_metric, ssim_metric])

history = dncnn_model.fit(
    train_dataset, 
    epochs=100, 
    validation_data=val_dataset, 
    callbacks=[early_stopper, reduce_lr, denoise_callback]
)

dncnn_model.save('dncnn_model.h5')