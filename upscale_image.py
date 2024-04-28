import tensorflow as tf
import os
from PIL import Image
import numpy as np
import cv2

def load_and_process_image(file_path):

    """
    Load an image and process it to be compatible with the model input.
    """

    img = tf.io.read_file(file_path)
    img = tf.image.decode_png(img, channels=1) 
    img = tf.image.convert_image_dtype(img, tf.float32)  
    img = tf.expand_dims(img, axis=0)  
    return img

def save_image(image, file_path):

    if image.ndim == 3 and image.shape[-1] == 1:
        image = np.squeeze(image, axis=-1)
        
    """
    Save an image (numpy array) to the specified file path.
    """

    image = Image.fromarray((image * 255).astype(np.uint8), 'L')  
    image.save(file_path)

model_checkpoint_path = './model_checkpoints/model_epoch_53'
model = tf.keras.models.load_model(model_checkpoint_path, compile=False)

lr_dir = './ProcessedImages/LR/Test'
sr_dir = './SuperResolvedImages'
os.makedirs(sr_dir, exist_ok=True)

# Process each low-resolution image and save the super-resolved output
for filename in os.listdir(lr_dir):
    lr_image_path = os.path.join(lr_dir, filename)
    sr_image_path = os.path.join(sr_dir, 'SR_' + filename)
    
    lr_image = load_and_process_image(lr_image_path)
    
    # Use the model to create a super-resolved image
    sr_image = model.predict(lr_image)[0] 
    
    sr_image = cv2.medianBlur(sr_image, ksize=3)
    
    # Save the super-resolved image
    save_image(sr_image, sr_image_path)

print("All super-resolved images have been saved.")
