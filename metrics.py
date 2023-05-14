import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity as ssim
from scipy.signal import correlate
from sklearn.metrics import pairwise_distances
import torch
import torch.nn.functional as F
from skimage.util import view_as_windows

def calculate_psnr(image1, image2):
    mse = np.mean((image1 - image2) ** 2)
    max_pixel = np.max(image1)
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr

def calculate_ssim(image1, image2):
    return ssim(image1, image2, data_range=1)

def calculate_locals(image, window_size):
    # Create a sliding window view of the image
    windows = view_as_windows(image, (window_size, window_size))
    # Calculate the mean value of each local patch
    local_means = np.mean(windows, axis=(2, 3))
    local_stddevs = np.std(windows, axis=(2, 3))
    # for i in range(local_stddevs.shape[0]):
    #     for j in range(local_stddevs.shape[1]):
    #         if local_stddevs[i,j]<=0:
    #             print(i, j, local_stddevs[i,j])
    normalized_patches = (windows - local_means[:, :, np.newaxis, np.newaxis]) / (local_stddevs[:, :, np.newaxis, np.newaxis]+1e-6)
    return normalized_patches

def locally_normalized_cross_correlation(image1, image2, window_size=15):
    normalized_patches1 = calculate_locals(image1, window_size)
    normalized_patches2 = calculate_locals(image2, window_size)
    cross_corr = correlate(normalized_patches1, normalized_patches2, mode="same")
    print(cross_corr.shape)
    # Compute the LNCC as the mean of the cross-correlation
    lncc = np.mean(cross_corr)
    return lncc

def calculate_dice_coefficient(image1, image2, threshold=0.5):
    image1_binary = (image1 > threshold).astype(int)
    image2_binary = (image2 > threshold).astype(int)
    
    intersection = np.sum(image1_binary * image2_binary)
    dice_coefficient = (2.0 * intersection) / (np.sum(image1_binary) + np.sum(image2_binary))
    return dice_coefficient



# Example usage
# image1 = Image.open("/Users/lyudonghang/image_enhancement/train_datasets/breast/high_quality/1500.png")
# image2 = Image.open("/Users/lyudonghang/image_enhancement/train_datasets/breast/low_quality/1500.png")

# Convert images to float and normalize them
# image1 = np.float32(np.array(image1)) 
# image2 = np.float32(np.array(image2)) 
# print(locally_normalized_cross_correlation(image1, image2))
# print(np.max(image1))

# psnr = calculate_psnr(image1, image2)
# ssim1 = calculate_ssim(image1, image2)
# dice_coefficient = calculate_dice_coefficient(image1, image2)

# print(f"PSNR: {psnr:.2f}")
# print(f"SSIM: {ssim:.4f}")
# print(f"LNCC: {lncc:.4f}")
# print(f"Dice Coefficient: {dice_coefficient:.4f}")
# import numpy as np
# from skimage.util import view_as_windows

# def calculate_local_means(image, window_size):
#     # Create a sliding window view of the image
#     windows = view_as_windows(image, (window_size, window_size))
#     print(windows.shape)
#     # Calculate the mean value of each local patch
#     local_means = np.mean(windows, axis=(2, 3))
#     return local_means

# # Example usage
# image = np.random.rand(256, 256)  # Example image
# window_size = 15  # Window size for local patches

# local_means = calculate_local_means(image, window_size)
# print("Local Means:")
# print(local_means)
