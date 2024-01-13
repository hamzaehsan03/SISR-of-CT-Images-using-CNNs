import os
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from preprocess_pipeline import preprocess_image


def process_images(image_dir):
    for sub_directory, directory, images in os.walk(image_dir):
        for image in images:
            if image.endswith('.png'):
                image_path = os.path.join(sub_directory, image)
                print("processing", image_path)
                preprocessed_image = preprocess_image(image_path)

if __name__ == '__main__':
    current_directory = os.getcwd()
    image_directory = os.path.join(current_directory, "DeepLesion")
    process_images(image_directory)