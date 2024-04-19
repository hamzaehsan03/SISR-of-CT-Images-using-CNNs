import tensorflow as tf
from PIL import Image
import numpy as np

def load_image(image_path, color_mode='grayscale', size=(None, None), scale=True):
    """Load and preprocess an image."""
    img = tf.io.read_file(image_path)
    if color_mode == 'grayscale':
        img = tf.image.decode_png(img, channels=1)
    else:
        img = tf.image.decode_png(img, channels=3)

    # Convert image to tf.float32 after decoding
    img = tf.image.convert_image_dtype(img, tf.float32)
    
    if size != (None, None):
        img = tf.image.resize(img, size)

    if scale:
        # Image is already in tf.float32, so we can safely scale it
        img = img / 255.0

    img = tf.expand_dims(img, 0)  # Add batch dimension
    return img



def save_image(image, file_path):
    """Save an image to the specified file path."""
    if image.ndim == 4:
        image = tf.squeeze(image, axis=0)
    image = image * 255.0
    image = np.array(image, dtype=np.uint8)
    if image.shape[-1] == 1:
        image = image[:,:,0]
    img = Image.fromarray(image)
    img.save(file_path)

def load_model(model_path):
    """Load the trained DnCNN model."""
    return tf.keras.models.load_model(model_path)

def denoise_image(input_image_path, output_image_path, model_path):
    """Load an image, denoise it using the trained model, and save the result."""
    model = load_model(model_path)
    image = load_image(input_image_path)
    denoised_image = model.predict(image)
    save_image(denoised_image, output_image_path)

if __name__ == "__main__":
    input_image_path = './output_images/epoch_45_img_2_sr.png'  # Specify the path to your noisy image
    output_image_path = './image.png'  # Specify where to save the denoised image
    model_path = './dncnn_model.h5'  # Specify the path to your trained model file

    denoise_image(input_image_path, output_image_path, model_path)
