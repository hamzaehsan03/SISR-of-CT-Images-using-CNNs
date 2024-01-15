import os
from PIL import Image
from preprocess_pipeline import preprocess_image

def save_process_image(image_path, output_subdir):
    preprocessed_image = preprocess_image(image_path)
    if preprocessed_image is not None:
        image = os.path.basename(image_path)
        output_path = os.path.join(output_subdir, image)
        processed_image = Image.fromarray((preprocessed_image * 255).astype('uint8'))
        processed_image.save(output_path)
        return f"processed image saved to {output_path}"
        