import numpy as np


def rfft2_to_fft2(img_shape, img_rFFT):
    fcols = img_shape[-1]
    FFT_cols = img_rFFT.shape[-1]

    full_FFT = np.zeros(img_shape, dtype=img_rFFT.dtype)
    full_FFT[:, :FFT_cols] = img_rFFT

    top = img_rFFT[0, 1:]

    if fcols % 2 == 0:
        full_FFT[0, FFT_cols - 1 :] = top[::-1].conj()
        middle = img_rFFT[1:, 1:]
        middle = np.hstack((middle, middle[::-1, ::-1][:, 1:].conj()))
    else:
        full_FFT[0, FFT_cols:] = top[::-1].conj()
        middle = img_rFFT[1:, 1:]
        middle = np.hstack((middle, middle[::-1, ::-1].conj()))

    full_FFT[1:, 1:] = middle

    return full_FFT


def pad_image(image, output_size, **kwargs):
    pad_top = (output_size[0] - image.shape[0]) // 2
    pad_bottom = (output_size[0] - image.shape[0]) - pad_top

    pad_left = (output_size[1] - image.shape[1]) // 2
    pad_right = (output_size[1] - image.shape[1]) - pad_left

    padding = ((pad_top, pad_bottom), (pad_left, pad_right))

    return np.pad(image, padding, **kwargs)


def grayscale_convert(image):
    image = 255 * (image / image.max())
    w, h = image.shape

    image_gray = np.empty((w, h, 3), dtype=np.uint8)
    image_gray[:, :, 2] = image_gray[:, :, 1] = image_gray[:, :, 0] = image

    return image_gray
