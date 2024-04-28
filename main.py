import datetime
import os
import numpy as np
import pydicom
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.data import shepp_logan_phantom
from skimage.draw import ellipse as circle, line
from skimage.color import gray2rgb, rgb2gray
from scipy.fftpack import fft, ifft, fftfreq
from multiprocessing import Pool
from functools import partial


def radon_transform(image, scan_count, detector_count, angle_range, pad=True, plot=False):
    if pad: image = circle_pad(image)
    center = center_of(image)
    width = height = image.shape[0]
    radius = width // 2
    alphas = np.linspace(0, 180, scan_count)
    results = np.zeros((scan_count, detector_count))
    if plot:
        plt.figure()
        for i, alpha in enumerate(alphas):
            results[i] = single_radon_transform(detector_count, angle_range, image, radius, center, alpha)
            plt.imshow(np.swapaxes(results, 0, 1), cmap=plt.cm.Greys_r)
            plt.show()
    else:
        # ~core times faster
        with Pool() as pool:
            results = pool.map(partial(single_radon_transform, detector_count, angle_range, image, radius, center),
                               alphas)

    return np.swapaxes(results, 0, 1)

# Helper functions
def center_pad(array, shape, *args, **kwargs):
    """Centered padding to a given shape"""
    pad = (np.array(shape) - np.array(array.shape)) / 2
    pad = np.array([np.floor(pad), np.ceil(pad)]).T.astype(int)
    return np.pad(array, pad, *args, **kwargs)


def circle_pad(array, *args, **kwargs):
    """Centered padding to a square side equal to the diameter of the circle circumscribed on a rectangle"""
    w, h = array.shape
    side = int(np.ceil(np.sqrt(w ** 2 + h ** 2)))
    return center_pad(array, (side, side), *args, **kwargs)


def center_of(array):
    """Center indices of an n-dimensional array"""
    return np.floor(np.array(array.shape) / 2).astype(int)


def rescale(array, min=0, max=1):
    """Rescale array elements to range [min, max]"""
    res = array.astype('float32')
    res -= np.min(res)
    res /= np.max(res)
    res -= min
    res *= max
    return res


def clip(array, min, max):
    """Clip array elements to range [min, max]"""
    array[array < min] = min
    array[array > max] = max
    return array


def rmse(a, b):
    """Root mean square error"""
    a, b = rescale(a), rescale(b)
    return np.sqrt(np.mean((a - b) ** 2))


def cut_pad(img, height, width):
    y, x = img.shape
    startx = x // 2 - (width // 2)
    starty = y // 2 - (height // 2)
    return img[starty:starty + height, startx:startx + width]


# Bresenham line function
def bresenham(x0, y0, x1, y1):
    if abs(y1 - y0) > abs(x1 - x0):
        swapped = True
        x0, y0, x1, y1 = y0, x0, y1, x1
    else:
        swapped = False
    m = (y1 - y0) / (x1 - x0) if x1 - x0 != 0 else 1
    q = y0 - m * x0
    if x0 < x1:
        xs = np.arange(np.floor(x0), np.ceil(x1) + 1, +1, dtype=int)
    else:
        xs = np.arange(np.ceil(x0), np.floor(x1) - 1, -1, dtype=int)
    ys = np.round(m * xs + q).astype(int)
    if swapped:
        xs, ys = ys, xs
    return np.array([xs, ys])


def radon_lines(emitters, detectors):
    return [np.array(bresenham(x0, y0, x1, y1)) for (x0, y0), (x1, y1) in zip(emitters, detectors)]


def single_radon_transform(detector_count, angle_range, image, radius, center, alpha):
    emitters = emitter_coords(alpha, angle_range, detector_count, radius, center)
    detectors = detector_coords(alpha, angle_range, detector_count, radius, center)
    lines = radon_lines(emitters, detectors)
    result = rescale(np.array([np.sum(image[tuple(line)]) for line in lines]))
    return result

# Functions to calculate emitter and detector coordinates
def circle_points(angle_shift, angle_range, count, radius=1, center=(0, 0)):
    angles = np.linspace(0, angle_range, count) + angle_shift
    cx, cy = center
    x = radius * np.cos(angles) - cx
    y = radius * np.sin(angles) - cy
    points = np.array(list(zip(x, y)))
    return np.floor(points).astype(int)


def detector_coords(alpha, angle_range, count, radius=1, center=(0, 0)):
    return circle_points(np.radians(alpha - angle_range / 2), np.radians(angle_range), count, radius, center)


def emitter_coords(alpha, angle_range, count, radius=1, center=(0, 0)):
    return circle_points(np.radians(alpha - angle_range / 2 + 180), np.radians(angle_range), count, radius, center)[::-1]


# Inverse radon transform (backprojection) with optional filtering
def filter_sinogram(sinogram):
    n = sinogram.shape[0]  # number of detectors
    filter = 2 * np.abs(fftfreq(n).reshape(-1, 1))
    result = ifft(fft(sinogram, axis=0) * filter, axis=0)
    result = clip(np.real(result), 0, 1)
    return result


def single_inverse_radon_transform(image, tmp, single_alpha_sinogram, alpha, detector_count, angle_range, radius,
                                   center):
    emitters = emitter_coords(alpha, angle_range, detector_count, radius, center)
    detectors = detector_coords(alpha, angle_range, detector_count, radius, center)
    lines = radon_lines(emitters, detectors)
    for i, line in enumerate(lines):
        image[tuple(line)] += single_alpha_sinogram[i]
        tmp[tuple(line)] += 1


def inverse_radon(shape, sinogram, angle_range, pad=True, plot=False, filtering=False):
    if filtering:
        sinogram = filter_sinogram(sinogram)
    number_of_detectors, number_of_scans = sinogram.shape
    sinogram = np.swapaxes(sinogram, 0, 1)

    result = np.zeros(shape)
    if pad: result = circle_pad(result)
    tmp = np.zeros(result.shape)

    center = center_of(result)
    width = height = result.shape[0]
    radius = width // 2
    alphas = np.linspace(0, 180, number_of_scans)

    for i, alpha in enumerate(alphas):
        single_inverse_radon_transform(result, tmp, sinogram[i], alpha, number_of_detectors, angle_range, radius,
                                       center)
        if plot:
            plt.imshow(result, cmap=plt.cm.Greys_r)
            plt.show()

    tmp[tmp == 0] = 1
    result = rescale(result / tmp)
    if pad: result = cut_pad(result, *shape)
    return result


# Test function
def test(image, scans, detectors, angle_range, filtering, plot=False):
    padded = circle_pad(image)
    sinogram = radon_transform(padded, scans, detectors, angle_range, pad=False)
    output = inverse_radon(padded.shape, sinogram, angle_range, filtering=filtering, pad=False)
    output = cut_pad(output, *image.shape)
    if plot:
        fig, axs = plt.subplots(1, 2, figsize=(10, 10))
        axs[0].imshow(image, cmap='gray')
        axs[1].imshow(output, cmap='gray')
    return output, rmse(image, output)


def test_filtering(image, scans=360, detectors=360, angle=270):
    y1, loss1 = test(image, scans, detectors, angle, filtering=False)
    y2, loss2 = test(image, scans, detectors, angle, filtering=True)
    print(f'rmse no filtering  {loss1:.6f}')
    print(f'rmse filtering     {loss2:.6f}')
    fig, axs = plt.subplots(1, 3, figsize=(10, 10))
    axs[0].imshow(image, cmap='gray')
    axs[1].imshow(y1, cmap='gray')
    axs[2].imshow(y2, cmap='gray')


def main():
    # Example usage
    input_image = shepp_logan_phantom()
    pad_image = circle_pad(input_image)

    # Radon Transform
    number_of_scans = 360
    number_of_detectors = pad_image.shape[0]
    angle_range = 270
    sinogram = radon_transform(input_image, number_of_scans, number_of_detectors, angle_range)

    # Inverse Radon Transform
    output = inverse_radon(input_image.shape, sinogram, angle_range, filtering=True)

    # Display input image, sinogram, and output image
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))

    axs[0].imshow(input_image, cmap='gray')
    axs[0].set_title('Input Image')

    axs[1].imshow(sinogram, cmap='gray')
    axs[1].set_title('Sinogram')

    axs[2].imshow(output, cmap="gray")
    axs[2].set_title('Reconstructed Image')

    plt.show()

    # Statistical Analysis
    test_filtering(input_image, scans=100, detectors=100, angle=90)

if __name__ == "__main__":
    main()
