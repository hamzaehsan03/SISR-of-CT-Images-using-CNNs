import os
from PIL import Image
from preprocess_pipeline import check_image, intensity_scale, normalise_image


def preprocess_image(image_path):
    # check image format
    try:
        check_image(image_path)
    except ValueError:
        print ("debug")
        return None

    image_hu = intensity_scale(image_path)
    image_normalised = normalise_image(image_hu)

    return image_normalised

def save_process_image(image_path, output_dir_hr, output_dir_lr, scale_factor=2):
    preprocessed_image = preprocess_image(image_path)
    if preprocessed_image is not None:
        filename = os.path.basename(os.path.dirname(image_path)) + '_' + os.path.basename(image_path)

        output_path_hr = os.path.join(output_dir_hr, filename)
        processed_image = Image.fromarray((preprocessed_image * 255).astype('uint8'))
        processed_image.save(output_path_hr)

        output_path_lr = os.path.join(output_dir_lr, filename)
        lr_image = processed_image.resize((processed_image.width // scale_factor, processed_image.height // scale_factor))
        lr_image.save(output_path_lr)

        return f"HR image saved to {output_path_hr}, LR image saved to {output_path_lr}"
