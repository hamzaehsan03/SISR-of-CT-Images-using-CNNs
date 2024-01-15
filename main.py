import os
from process_image import save_process_image
from preprocess_pipeline import preprocess_image
from multiprocess import parallel_process

def process_images(image_dir, output_dir):
    argument_list = []

    for sub_directory, directory, images in os.walk(image_dir):
        for image in images:
            if image.endswith('.png'):
                image_path = os.path.join(sub_directory, image)
                print("processing", image_path)

                rel_path = os.path.relpath(sub_directory, image_dir)
                output_subdir = os.path.join(output_dir, rel_path)

                if not os.path.exists(output_subdir):
                    os.makedirs(output_subdir)

                argument_list.append((image_path, output_subdir))

    results = parallel_process(save_process_image, argument_list)
    for result in results:
        print(result)

if __name__ == '__main__':
    current_directory = os.getcwd()
    image_directory = os.path.join(current_directory, "DeepLesion")
    output_directory = os.path.join(current_directory, "ProcessedImages")
    process_images(image_directory, output_directory)