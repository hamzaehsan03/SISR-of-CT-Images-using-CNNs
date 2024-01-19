import os
from process_image import save_process_image
from preprocess_pipeline import process_folders
from multiprocess import parallel_process

def process_images(image_dir, output_dir_hr, output_dir_lr):
    argument_list = []

    if not os.path.exists(output_dir_hr):
        os.makedirs(output_dir_hr)
    if not os.path.exists(output_dir_lr):
        os.makedirs(output_dir_lr)

    for sub_directory, directory, images in os.walk(image_dir):
        for image in images:
            if image.endswith('.png'):
                image_path = os.path.join(sub_directory, image)
                print("processing", image_path)
                argument_list.append((image_path, output_dir_hr, output_dir_lr))

    results = parallel_process(save_process_image, argument_list)
    for result in results:
        print(result)

if __name__ == '__main__':
    current_directory = os.getcwd()
    image_directory = os.path.join(current_directory, "DeepLesion")
    output_directory_hr = os.path.join(current_directory, "ProcessedImages\\HR")
    output_directory_lr = os.path.join(current_directory, "ProcessedImages\\LR")
    process_images(image_directory, output_directory_hr, output_directory_lr)
    process_folders(output_directory_hr, output_directory_lr)
