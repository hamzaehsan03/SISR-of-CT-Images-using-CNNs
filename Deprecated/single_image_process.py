'''
this file is primarily for documentation purposes
its use is to compare the images after preprocessing steps have been taken and output them visually
'''
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from preprocess_pipeline import preprocess_image

print("starting preprocessing")

image_path = "C:\\Users\\Hamza\\Documents\\University\\Dissertation stuff\\Code\\final-year-project\\DeepLesion\\Images_png_01\\Images_png\\000001_01_01\\103.png"

# process the image
original_image = Image.open(image_path)

# convert to HU
original_image_array = np.array(original_image, dtype=np.int16) - 32768 
preprocessed_image = preprocess_image(image_path)

if preprocessed_image is not None:
    # Create a figure with two subplots, 1 row 2 columns
    fig, axes = plt.subplots(1, 2, figsize=(12, 6)) 

    # display the original image
    axes[0].imshow(original_image_array, cmap='gray')
    axes[0].set_title("Original Image")

    # display the processed image
    axes[1].imshow(preprocessed_image, cmap='gray')
    axes[1].set_title("Processed Image")

    plt.show()
else:
    print("image processing failed")