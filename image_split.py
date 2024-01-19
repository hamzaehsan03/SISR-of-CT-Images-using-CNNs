import os
import shutil
from multiprocess import parallel_process
from sklearn.model_selection import train_test_split



def copy_images(src, dst):
    shutil.copy(src, dst)


def parallel_copy(files, src_dir, dst_dir):
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)
    
    arguments = []
    for f in files:
        src_path = os.path.join(src_dir, f)
        dst_path = os.path.join(dst_dir, f)
        if src_path != dst_path:
            arguments.append((src_path, dst_path))  # Arguments as a tuple
    
    parallel_process(copy_images, arguments)




def main():

    current_directory = os.getcwd()
    hr_dir = os.path.join(current_directory, "ProcessedImages\\HR")
    lr_dir = os.path.join(current_directory, "ProcessedImages\\LR")

    hr_images = [os.path.join(hr_dir, f) for f in os.listdir(hr_dir) if f.endswith('.png')]

    # Ensure each HR image has a corresponding LR image
    lr_images_set = set(os.listdir(lr_dir))
    missing_lr_files = [f for f in os.listdir(hr_dir) if f not in lr_images_set]

    if missing_lr_files:
        print("Missing LR files for:", missing_lr_files)
    # Handle the missing files, e.g., by removing them from hr_images or taking other actions
    hr_images_set = set(os.listdir(hr_dir))
    missing_hr_files = [f for f in os.listdir(lr_dir) if f not in hr_images_set]

    if missing_hr_files:
        print("Missing LR files for:", missing_hr_files)


    # Split the dataset into training, validation, and test sets
    train_files, test_files = train_test_split(hr_images, test_size=0.15, random_state=727) # wysi
    train_files, val_files = train_test_split(train_files, test_size=0.2, random_state=727)  # 70/15/15 split

    for file_set, set_name in zip([train_files, val_files, test_files], ['Train', 'Validation', 'Test']):
        hr_set_dir = os.path.join(hr_dir, set_name)
        lr_set_dir = os.path.join(lr_dir, set_name)
        
        # Parallel copy HR images
        hr_set_files = [os.path.basename(f) for f in file_set]
        lr_set_files = [os.path.basename(f) for f in file_set]
        parallel_copy(hr_set_files, hr_dir, hr_set_dir)
        parallel_copy(lr_set_files, lr_dir, lr_set_dir)

       

if __name__ == '__main__':
    main()

#70178
#68176