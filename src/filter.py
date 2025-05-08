import cv2 as cv
import numpy as np
from scipy import ndimage
from skimage import filters


def window(type, shape):
    return filters.window(type, shape)


def butterworth(shape, cutoff, order, high_pass=True, squared_butterworth=True):
    ranges = [
        np.fft.ifftshift(
            (np.arange(-(dim - 1) // 2, (dim - 1) // 2 + 1) / (dim * cutoff)) ** 2
        )
        for dim in shape
    ]

    q2 = np.add(*np.meshgrid(*ranges, indexing="ij", sparse=True))
    q2 = np.power(q2, order)

    filter = 1 / (1 + q2)

    if high_pass:
        filter *= q2

    if not squared_butterworth:
        np.sqrt(filter, out=filter)

    return filter


def fresnel(
    shape, centerband, sideband, thickness=10, smoothness=1, distance=85, threshold=None
):
    mask = np.ones(shape, dtype=float)
    offset = (distance / 100) * (sideband - centerband)

    first_filter = (centerband + offset).astype(int)
    second_filter = (centerband - offset).astype(int)

    for position in [first_filter, second_filter]:
        cv.line(
            img=mask,
            pt1=tuple(centerband.astype(int)[::-1]),
            pt2=tuple(position[::-1]),
            color=0,
            thickness=thickness,
        )

    filter = ndimage.gaussian_filter(mask, sigma=smoothness)

    if threshold is not None:
        filter[filter < threshold] = 0

    return filter

    # img_fft_filtered = img_fft_shifted * filter
    # return fft.ifft2(fft.ifftshift(img_fft_filtered)).real
