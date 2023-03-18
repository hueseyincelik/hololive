import os
import sys

import numpy as np
import scipy.fft as sfft

os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "hide"

import pygame as pg

from . import microscope


class GUI:
    def __init__(
        self,
        ip,
        port,
        camera="CCD",
        exposure_time=0.2,
        sideband_quadrant="upper_left",
        auto_correlation_buffer=50,
        dimension=512,
    ):
        self.dimension = dimension

        self.microscope = microscope.Microscope(ip, port)
        self.microscope.configure_camera(camera, exposure_time)

        self.sideband_quadrant, self.auto_correlation_buffer = (
            sideband_quadrant,
            auto_correlation_buffer,
        )

        pg.init()

        self.screen = pg.display.set_mode((self.dimension, self.dimension))
        pg.display.set_caption("Live Phase")

        self.run()

    def run(self):
        while True:
            for event in pg.event.get():
                if event.type == pg.QUIT:
                    sys.exit()

            current_phase = self.get_phase()
            current_phase_grayscale = self.grayscale_convert(
                255 * current_phase / current_phase.max()
            )

            surface_phase_image = pg.surfarray.make_surface(current_phase_grayscale)
            self.screen.blit(surface_phase_image, (0, 0))

            pg.display.flip()

    def get_phase(self):
        img_CCD = self.microscope.acquire()

        img_fft = sfft.fft2(img_CCD)
        img_fft_shifted = sfft.fftshift(img_fft)

        if self.sideband_quadrant == "upper_left":
            img_shift_cropped = img_fft_shifted[
                : img_fft_shifted.shape[0] // 2 - self.auto_correlation_buffer,
                : img_fft_shifted.shape[1] // 2 - self.auto_correlation_buffer,
            ]
        elif self.sideband_quadrant == "upper_right":
            img_shift_cropped = img_fft_shifted[
                : img_fft_shifted.shape[0] // 2 - self.auto_correlation_buffer,
                img_fft_shifted.shape[1] // 2 + self.auto_correlation_buffer :,
            ]
        elif self.sideband_quadrant == "lower_left":
            img_shift_cropped = img_fft_shifted[
                img_fft_shifted.shape[0] // 2 + self.auto_correlation_buffer :,
                : img_fft_shifted.shape[1] // 2 - self.auto_correlation_buffer,
            ]
        elif self.sideband_quadrant == "lower_right":
            img_shift_cropped = img_fft_shifted[
                img_fft_shifted.shape[0] // 2 + self.auto_correlation_buffer :,
                img_fft_shifted.shape[1] // 2 + self.auto_correlation_buffer :,
            ]
        else:
            raise ValueError("Unsupported position!")

        sideband_position = np.argwhere(img_fft_shifted == np.amax(img_shift_cropped))[
            0
        ]

        sideband_distance = np.linalg.norm(
            np.asarray([int(p / 2) - 1 for p in img_shift_cropped.shape[::-1]])
            - np.asarray(sideband_position[::-1])
        )

        img_cut_out = img_fft_shifted[
            sideband_position[0] - int(sideband_distance / 6) : sideband_position[0]
            + int(sideband_distance / 6),
            sideband_position[1] - int(sideband_distance / 6) : sideband_position[1]
            + int(sideband_distance / 6),
        ]

        padding = np.abs(img_cut_out.shape[0] - self.dimension) // 2
        img_zero_padded = np.pad(
            img_cut_out, ((padding, padding), (padding, padding)), constant_values=0
        )

        return np.angle(sfft.ifft2(img_zero_padded)).swapaxes(0, 1)

    def grayscale_convert(self, image):
        image = 255 * (image / image.max())
        w, h = image.shape

        image_gray = np.empty((w, h, 3), dtype=np.uint8)
        image_gray[:, :, 2] = image_gray[:, :, 1] = image_gray[:, :, 0] = image

        return image_gray
