import itertools as it
import os
import sys
from datetime import datetime
from threading import Thread

import numpy as np
import scipy.fft as sfft
from tifffile import imwrite

os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "hide"

import pygame as pg

from . import microscope


class GUI:
    def __init__(self, ip, port, camera="CCD", exposure_time=0.2, dimension=512):
        self.dimension = dimension

        self.microscope = microscope.Microscope(ip, port)
        self.microscope.configure_camera(camera, exposure_time)

        self.sideband_position, self.sideband_distance = (0, 0), 0
        self.sideband_quadrant = "upper_left"
        self.sideband_lock = False

        self.amplifications, self.phase_amplification = (
            it.islice(it.cycle([1, 2, 3, 4]), 1, None),
            1,
        )
        self.auto_correlation_buffer = 50

        pg.init()

        self.screen, self.font = (
            pg.display.set_mode((self.dimension, self.dimension)),
            pg.freetype.SysFont(None, 18),
        )
        pg.display.set_caption("Live Phase")

        self.run()

    def run(self):
        while True:
            for event in pg.event.get():
                if event.type == pg.QUIT:
                    sys.exit()
                elif event.type == pg.KEYDOWN:
                    if event.key == pg.K_l:
                        self.sideband_lock = not self.sideband_lock

                    if event.key == pg.K_LEFT:
                        self.sideband_quadrant = "lower_left"
                    if event.key == pg.K_RIGHT:
                        self.sideband_quadrant = "upper_right"
                    if event.key == pg.K_UP:
                        self.sideband_quadrant = "upper_left"
                    if event.key == pg.K_DOWN:
                        self.sideband_quadrant = "lower_right"

                    if event.key == pg.K_PLUS:
                        self.auto_correlation_buffer += 5
                    if event.key == pg.K_MINUS and self.auto_correlation_buffer >= 5:
                        self.auto_correlation_buffer -= 5

                    if event.key == pg.K_a:
                        self.phase_amplification = next(self.amplifications)

                    if event.key == pg.K_s:
                        self.save_screenshot_thread = Thread(
                            target=self.save_screenshot
                        )
                        self.save_screenshot_thread.start()

            self.current_phase = self.phase_amplification * self.get_phase()
            self.current_phase_grayscale = self.grayscale_convert(
                255 * self.current_phase / self.current_phase.max()
            )

            surface_phase_image = pg.surfarray.make_surface(
                self.current_phase_grayscale
            )
            self.screen.blit(surface_phase_image, (0, 0))

            for coordinate, message in zip(
                [(5, 5), (5, 25), (5, 45), (5, 65)],
                [
                    f"Quadrant: {self.sideband_quadrant}",
                    f"Amplification: {self.phase_amplification}",
                    f"Locking: {self.sideband_lock}",
                    f"Buffer: {self.auto_correlation_buffer}",
                ],
            ):
                self.font.render_to(self.screen, coordinate, message, pg.Color("RED"))

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

        if not self.sideband_lock:
            self.sideband_position = np.argwhere(
                img_fft_shifted == img_shift_cropped.max()
            )[0]
            self.sideband_distance = np.linalg.norm(
                np.asarray([int(p / 2) - 1 for p in img_shift_cropped.shape[::-1]])
                - np.asarray(self.sideband_position[::-1])
            )

        img_cut_out = img_fft_shifted[
            self.sideband_position[0]
            - int(self.sideband_distance / 6) : self.sideband_position[0]
            + int(self.sideband_distance / 6),
            self.sideband_position[1]
            - int(self.sideband_distance / 6) : self.sideband_position[1]
            + int(self.sideband_distance / 6),
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

    def save_screenshot(self, datatype=np.float32, photometric="minisblack"):
        imwrite(
            f"hololive_{format(datetime.now(), '%Y-%m-%d_%H-%M-%S')}.tif",
            self.current_phase.astype(datatype),
            photometric=photometric,
        )
