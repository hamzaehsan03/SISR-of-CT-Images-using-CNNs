import os
import numpy as np
from PIL import Image

# check for whether the images provided fit the standardised format
def check_image(image_path, file_extension='png'):
    if not image_path.lower().endswith(file_extension.lower()):
        raise ValueError("Image format is incorrect")
    
    # with Image.open(image_path) as img:
    #     if (img.size != (512, 512) or img.size != (128, 128)):
    #         raise ValueError ("Image dimensions are not 512x512 or 128x128")
    # return
import os
from PIL import Image

def process_folders(hr_folder, lr_folder):
    # Process HR folder
    for filename in os.listdir(hr_folder):
        if filename.lower().endswith('.png') or filename.lower().endswith('.jpg'):
            file_path = os.path.join(hr_folder, filename)
            size_ok = False
            try:
                with Image.open(file_path) as img:
                    size_ok = img.size == (512, 512)
            except IOError:
                print(f"Error opening {file_path}. Skipping.")
                continue

            if not size_ok:
                try:
                    os.remove(file_path)
                    print(f"Deleted {file_path} as its dimensions were not 512x512.")
                except OSError as e:
                    print(f"Error deleting {file_path}: {e}")

    # Process LR folder
    for filename in os.listdir(lr_folder):
        if filename.lower().endswith('.png') or filename.lower().endswith('.jpg'):
            file_path = os.path.join(lr_folder, filename)
            size_ok = False
            try:
                with Image.open(file_path) as img:
                    size_ok = img.size == (256, 256)
            except IOError:
                print(f"Error opening {file_path}. Skipping.")
                continue

            if not size_ok:
                try:
                    os.remove(file_path)
                    print(f"Deleted {file_path} as its dimensions were not 128x128.")
                except OSError as e:
                    print(f"Error deleting {file_path}: {e}")

# pixel values are stored as integers, however these can to be mapped back to HU due to the image data containing the HU
# this is through the images being 16 bit unsigned, and an offset of 32768 allows for the HU to be found
# it's important to note that this level of accuracy isn't necessary for general SISR but can provide better model results
def intensity_scale(image_path):
    image = Image.open(image_path)
    image_array = np.array(image, dtype=np.int16)
    houndsfield_units = image_array - 32768
    return houndsfield_units
    
# transform the pixel intensity of an image from the HU scale to a normalised range
# use numpy to limit the values to the min and max bounds
# scale the values to between 0 and 1
def normalise_image(houndsfield_units, min=-1000, max=400):

    houndsfield_units = np.clip(houndsfield_units, min, max)
    normalised = (houndsfield_units - min) / (max - min)
    normalised = normalised.astype(np.float32)

    return normalised

