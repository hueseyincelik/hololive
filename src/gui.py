import itertools as it
import os
import sys
from datetime import datetime
from threading import Thread

import numpy as np
import scipy.fft as sfft
from skimage.filters import window
from tifffile import imwrite

os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "hide"

import pygame as pg

from . import microscope


class GUI:
    def __init__(
        self,
        ip,
        port,
        remote=True,
        camera="CCD",
        exposure_time=0.2,
        binning=4,
        dimension=512,
    ):
        self.dimension = dimension

        self.microscope = microscope.Microscope(ip, port, remote)
        self.microscope.configure_camera(camera, exposure_time, binning)

        self.sideband_position, self.sideband_distance = (0, 0), 0
        self.sideband_quadrant = "top"
        self.sideband_lock = False

        self.amplifications, self.phase_amplification = (
            it.islice(it.cycle([1, 2, 3, 4]), 1, None),
            1,
        )
        self.auto_correlation_buffer, self.hann_smoothing = 50, True

        self.reconstruct_amplitude = False
        self.pause = False

        pg.init()

        self.screen, self.font = (
            pg.display.set_mode((self.dimension, self.dimension), pg.RESIZABLE),
            pg.freetype.SysFont(None, 18),
        )
        pg.display.set_caption("HoloLive")

        self.run()

    def run(self):
        while True:
            for event in pg.event.get():
                if event.type == pg.QUIT:
                    sys.exit()
                elif event.type == pg.VIDEORESIZE:
                    self.screen = pg.display.set_mode((event.w, event.h), pg.RESIZABLE)
                elif event.type == pg.KEYDOWN:
                    if event.key == pg.K_l:
                        self.sideband_lock = not self.sideband_lock

                    if event.key == pg.K_UP:
                        self.sideband_quadrant = "top"
                    if event.key == pg.K_DOWN:
                        self.sideband_quadrant = "bottom"
                    if event.key == pg.K_LEFT:
                        self.sideband_quadrant = "left"
                    if event.key == pg.K_RIGHT:
                        self.sideband_quadrant = "right"

                    if event.key == pg.K_PLUS:
                        self.auto_correlation_buffer += 5
                    if event.key == pg.K_MINUS and self.auto_correlation_buffer >= 5:
                        self.auto_correlation_buffer -= 5

                    if event.key == pg.K_TAB:
                        self.reconstruct_amplitude = not self.reconstruct_amplitude

                    if event.key == pg.K_a:
                        self.phase_amplification = next(self.amplifications)

                    if event.key == pg.K_s:
                        self.save_screenshot_thread = Thread(
                            target=self.save_screenshot
                        )
                        self.save_screenshot_thread.start()

                    if event.key == pg.K_p:
                        self.pause = not self.pause

                    if event.key == pg.K_h:
                        self.hann_smoothing = not self.hann_smoothing

            if not self.pause:
                self.current_phase = (
                    np.angle(np.exp(1j * self.phase_amplification * self.get_phase()))
                    if self.phase_amplification != 1
                    else self.get_phase()
                )
                self.current_phase_grayscale = self.grayscale_convert(
                    255 * self.current_phase / self.current_phase.max()
                )

            surface_phase_image = pg.transform.smoothscale(
                pg.surfarray.make_surface(self.current_phase_grayscale),
                pg.display.get_surface().get_size(),
            )
            self.screen.blit(surface_phase_image, (0, 0))

            for coordinate, message in zip(
                [(5, 5), (5, 25), (5, 65), (5, 85), (5, 105)],
                [
                    f"Quadrant: {self.sideband_quadrant}",
                    f"Locking: {self.sideband_lock}",
                    f"Smoothing: {self.hann_smoothing}",
                    f"Amplification: {self.phase_amplification}",
                    f"Buffer: {self.auto_correlation_buffer}",
                ],
            ):
                self.font.render_to(self.screen, coordinate, message, pg.Color("RED"))

            pg.display.flip()

    def get_phase(self):
        img_CCD = self.microscope.acquire()

        img_fft = sfft.fft2(img_CCD)
        img_fft_shifted = sfft.fftshift(img_fft)

        if self.sideband_quadrant == "top":
            img_shift_cropped = img_fft_shifted[
                : img_fft_shifted.shape[0] // 2 - self.auto_correlation_buffer, :
            ]
        elif self.sideband_quadrant == "bottom":
            img_shift_cropped = img_fft_shifted[
                img_fft_shifted.shape[0] // 2 + self.auto_correlation_buffer :, :
            ]
        elif self.sideband_quadrant == "left":
            img_shift_cropped = img_fft_shifted[
                :, : img_fft_shifted.shape[1] // 2 - self.auto_correlation_buffer
            ]
        elif self.sideband_quadrant == "right":
            img_shift_cropped = img_fft_shifted[
                :, img_fft_shifted.shape[1] // 2 + self.auto_correlation_buffer :
            ]

        if not self.sideband_lock:
            self.sideband_position = np.argwhere(
                img_fft_shifted == img_shift_cropped.max()
            )[0]
            self.sideband_distance = np.linalg.norm(
                np.asarray(img_fft_shifted.shape) / 2 - self.sideband_position
            )

        cut_out_idx = [
            [
                int(np.clip(sb_pos - self.sideband_distance / 6, 0, img_dim - 1)),
                int(np.clip(sb_pos + self.sideband_distance / 6, 0, img_dim - 1)),
            ]
            for (sb_pos, img_dim) in zip(self.sideband_position, img_fft_shifted.shape)
        ]

        img_cut_out = img_fft_shifted[
            cut_out_idx[0][0] : cut_out_idx[0][1], cut_out_idx[1][0] : cut_out_idx[1][1]
        ]

        if self.hann_smoothing:
            img_cut_out *= window("hann", img_cut_out.shape)

        padding = np.abs(img_cut_out.shape[0] - self.dimension) // 2
        img_zero_padded = np.pad(
            img_cut_out, ((padding, padding), (padding, padding)), constant_values=0
        )

        return (
            np.abs(sfft.ifft2(img_zero_padded)).swapaxes(0, 1)
            if self.reconstruct_amplitude
            else np.angle(sfft.ifft2(img_zero_padded)).swapaxes(0, 1)
        )

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
