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
