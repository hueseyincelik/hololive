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


def butterworth_filter(shape, cutoff, order, high_pass=True, squared_butterworth=True):
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


def convert_gray(image):
    image_8bit = 255 * (image - image.min()) / (image.max() - image.min())
    image_gray = np.stack((image_8bit,) * 3, axis=-1, dtype=np.uint8, casting="unsafe")

    return image_gray
