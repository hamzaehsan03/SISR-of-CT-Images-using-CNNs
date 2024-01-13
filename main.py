import os
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from preprocess_pipeline import preprocess_image


def process_images(image_dir, output_dir):

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for sub_directory, directory, images in os.walk(image_dir):
        for image in images:
            if image.endswith('.png'):
                image_path = os.path.join(sub_directory, image)
                print("processing", image_path)
                preprocessed_image = preprocess_image(image_path)

                if preprocess_image is not None:
                    rel_path = os.path.relpath(sub_directory, image_dir)
                    output_subdir = os.path.join(output_dir, rel_path)

                    if not os.path.exists(output_subdir):
                        os.makedirs(output_subdir)

                    output_path = os.path.join(output_subdir, image)
                    processed_image = Image.fromarray((preprocessed_image * 255).astype('uint8'))
                    processed_image.save(output_path)
                    print("saved image to ", output_path)


if __name__ == '__main__':
    current_directory = os.getcwd()
    image_directory = os.path.join(current_directory, "DeepLesion")
    output_directory = os.path.join(current_directory, "ProcessedImages")
    process_images(image_directory, output_directory)