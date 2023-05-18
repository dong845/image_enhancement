# import numpy as np
# from PIL import Image
# from skimage.metrics import structural_similarity as ssim
# from scipy.signal import correlate
# from sklearn.metrics import pairwise_distances
# import torch
# import torch.nn.functional as F
# from skimage.util import view_as_windows
#
# def calculate_psnr(image1, image2):
#     mse = np.mean((image1 - image2) ** 2)
#     max_pixel = np.max(image1)
#     psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
#     return psnr
#
# def calculate_ssim(image1, image2):
#     return ssim(image1, image2, data_range=1)
#
# def calculate_locals(image, window_size):
#     # Create a sliding window view of the image
#     windows = view_as_windows(image, (window_size, window_size))
#     # Calculate the mean value of each local patch
#     local_means = np.mean(windows, axis=(2, 3))
#     local_stddevs = np.std(windows, axis=(2, 3))
#     # for i in range(local_stddevs.shape[0]):
#     #     for j in range(local_stddevs.shape[1]):
#     #         if local_stddevs[i,j]<=0:
#     #             print(i, j, local_stddevs[i,j])
#     normalized_patches = (windows - local_means[:, :, np.newaxis, np.newaxis]) / (local_stddevs[:, :, np.newaxis, np.newaxis]+1e-6)
#     return normalized_patches
#
# def locally_normalized_cross_correlation(image1, image2, window_size=15):
#     normalized_patches1 = calculate_locals(image1, window_size)
#     normalized_patches2 = calculate_locals(image2, window_size)
#     cross_corr = correlate(normalized_patches1, normalized_patches2, mode="same")
#     print(cross_corr.shape)
#     # Compute the LNCC as the mean of the cross-correlation
#     lncc = np.mean(cross_corr)
#     return lncc
#
# def calculate_dice_coefficient(image1, image2, threshold=0.5):
#     image1_binary = (image1 > threshold).astype(int)
#     image2_binary = (image2 > threshold).astype(int)
#
#     intersection = np.sum(image1_binary * image2_binary)
#     dice_coefficient = (2.0 * intersection) / (np.sum(image1_binary) + np.sum(image2_binary))
#     return dice_coefficient

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

import numpy as np
import torch
from math import exp
import torch.nn.functional as F
from torch.autograd import Variable
import torch
from skimage.util import view_as_windows


def calculate_dice_coefficient(image1, image2, threshold=0.5):
    image1_binary = (image1 > threshold).astype(int)
    image2_binary = (image2 > threshold).astype(int)

    intersection = np.sum(image1_binary * image2_binary)
    dice_coefficient = (2.0 * intersection) / (np.sum(image1_binary) + np.sum(image2_binary))
    return dice_coefficient


def compute_measure(x, y, pred, data_range):
    original_psnr = compute_PSNR(x, y, data_range)
    original_ssim = compute_SSIM(x, y, data_range)
    original_rmse = compute_RMSE(x, y)
    pred_psnr = compute_PSNR(pred, y, data_range)
    pred_ssim = compute_SSIM(pred, y, data_range)
    pred_rmse = compute_RMSE(pred, y)
    return (original_psnr, original_ssim, original_rmse), (pred_psnr, pred_ssim, pred_rmse)


def compute_MSE(img1, img2):
    return ((img1 - img2) ** 2).mean()


def compute_RMSE(img1, img2):
    if type(img1) == torch.Tensor:
        return torch.sqrt(compute_MSE(img1, img2)).item()
    else:
        return np.sqrt(compute_MSE(img1, img2))


def compute_PSNR(img1, img2, data_range):
    if type(img1) == torch.Tensor:
        mse_ = compute_MSE(img1, img2)
        return 10 * torch.log10((data_range ** 2) / mse_).item()
    else:
        mse_ = compute_MSE(img1, img2)
        return 10 * np.log10((data_range ** 2) / mse_)


def compute_SSIM(img1, img2, data_range, window_size=11, channel=1, size_average=True):
    # referred from https://github.com/Po-Hsun-Su/pytorch-ssim
    if len(img1.size()) == 2:
        shape_ = img1.shape[-1]
        img1 = img1.view(1,1,shape_ ,shape_ )
        img2 = img2.view(1,1,shape_ ,shape_ )
    window = create_window(window_size, channel)
    window = window.type_as(img1)

    mu1 = F.conv2d(img1, window, padding=window_size//2)
    mu2 = F.conv2d(img2, window, padding=window_size//2)
    mu1_sq, mu2_sq = mu1.pow(2), mu2.pow(2)
    mu1_mu2 = mu1*mu2

    sigma1_sq = F.conv2d(img1*img1, window, padding=window_size//2) - mu1_sq
    sigma2_sq = F.conv2d(img2*img2, window, padding=window_size//2) - mu2_sq
    sigma12 = F.conv2d(img1*img2, window, padding=window_size//2) - mu1_mu2

    C1, C2 = (0.01*data_range)**2, (0.03*data_range)**2
    #C1, C2 = 0.01**2, 0.03**2

    ssim_map = ((2*mu1_mu2+C1)*(2*sigma12+C2)) / ((mu1_sq+mu2_sq+C1)*(sigma1_sq+sigma2_sq+C2))
    if size_average:
        return ssim_map.mean().item()
    else:
        return ssim_map.mean(1).mean(1).mean(1).item()


def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window