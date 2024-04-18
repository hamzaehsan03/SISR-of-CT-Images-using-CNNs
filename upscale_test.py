import cv2
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

def upscale_image(image_path, method, scale_factor):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    new_size = (int(image.shape[1] * scale_factor), int(image.shape[0] * scale_factor))
    upscaled_image = cv2.resize(image, new_size, interpolation=method)
    return upscaled_image

def evaluate_quality(upscaled_image, ground_truth_path):
    ground_truth = cv2.imread(ground_truth_path, cv2.IMREAD_GRAYSCALE)

    # Calculate PSNR and SSIM
    psnr_value = psnr(ground_truth, upscaled_image, data_range=ground_truth.max() - ground_truth.min())
    ssim_value = ssim(ground_truth, upscaled_image, data_range=ground_truth.max() - ground_truth.min())
    return psnr_value, ssim_value

def main():
    image_path = 'output_images\epoch_66_img_2_lr_256.png'
    ground_truth_path = 'output_images\epoch_66_img_2_hr.png'
    scale_factor = 2.0  

    # methods
    methods = {
        'Nearest Neighbor': cv2.INTER_NEAREST,
        'Bilinear': cv2.INTER_LINEAR,
        'Bicubic': cv2.INTER_CUBIC,
        'Lanczos': cv2.INTER_LANCZOS4
    }

    # Process each method
    for name, method in methods.items():
        upscaled_image = upscale_image(image_path, method, scale_factor)
        psnr_value, ssim_value = evaluate_quality(upscaled_image, ground_truth_path)
        print(f'{name} - PSNR: {psnr_value:.2f}, SSIM: {ssim_value:.3f}')

if __name__ == "__main__":
    main()
