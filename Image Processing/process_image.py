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

def save_process_image(image_path, output_dir):
    preprocessed_image = preprocess_image(image_path)
    if preprocessed_image is not None:
        filename = os.path.basename(os.path.dirname(image_path)) + '_' + os.path.basename(image_path)
        output_path = os.path.join(output_dir, filename)

        # convert array back to an image
        processed_image = Image.fromarray((preprocessed_image * 255).astype('uint8'))
        processed_image.save(output_path)
        return f"processed image saved to {output_path}"

def create_lr_image(hr_image_path, scale_factor = 4):
    with Image.open(hr_image_path) as image:
        lr_image = image.resize(image.width // scale_factor, image.height // scale_factor)
        return lr_image