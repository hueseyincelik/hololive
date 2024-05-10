import itertools as it
import os
import sys
from datetime import datetime
from threading import Thread

import numpy as np
import scipy.fft as sfft
from skimage import draw, filters

os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "hide"

import pygame as pg

from . import image, microscope


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
        real_FFT=False,
    ):
        self.dimension = dimension
        self.real_FFT = real_FFT

        self.microscope = microscope.Microscope(ip, port, remote)
        self.microscope.configure_camera(camera, exposure_time, binning)

        self.sideband_position, self.sideband_distance = (0, 0), 0
        self.sideband_area, self.sideband_lock = "upper", False

        self.amplifications, self.phase_amplification = (
            it.islice(it.cycle([1, 2, 3, 4]), 1, None),
            1,
        )
        self.centerband_mask, self.hann_smoothing = 5, True

        self.object_image_wave, self.reference_image_wave = None, None
        self.fringe_contrast = 0

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
                        self.reference_image_wave = None

                    if event.key == pg.K_UP:
                        self.sideband_area = "upper"
                    if event.key == pg.K_DOWN:
                        self.sideband_area = "lower"

                    if event.key == pg.K_PLUS:
                        self.centerband_mask += 1
                    if event.key == pg.K_MINUS and self.centerband_mask > 1:
                        self.centerband_mask -= 1

                    if event.key == pg.K_TAB:
                        self.reconstruct_amplitude = not self.reconstruct_amplitude
                        self.amplifications, self.phase_amplification = (
                            it.islice(it.cycle([1, 2, 3, 4]), 1, None),
                            1,
                        )

                    if event.key == pg.K_a and not self.reconstruct_amplitude:
                        self.phase_amplification = next(self.amplifications)

                    if event.key == pg.K_r:
                        self.reference_image_wave = self.object_image_wave.copy()
                        self.sideband_lock = True

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
                self.current_reconstruction = self.reconstruct()

                if self.phase_amplification != 1 and not self.reconstruct_amplitude:
                    self.current_reconstruction = np.angle(
                        np.exp(
                            1j * self.phase_amplification * self.current_reconstruction
                        )
                    )

                self.current_reconstruction_grayscale = image.grayscale_convert(
                    self.current_reconstruction
                )

            surface_phase_image = pg.transform.smoothscale(
                pg.surfarray.make_surface(self.current_reconstruction_grayscale),
                pg.display.get_surface().get_size(),
            )
            self.screen.blit(surface_phase_image, (0, 0))

            if self.reference_image_wave is not None:
                pg.draw.rect(
                    self.screen,
                    pg.Color("RED"),
                    [0, 0, *pg.display.get_surface().get_size()],
                    4,
                )

            for coordinate, message in zip(
                [(5, 5), (5, 25), (5, 45), (5, 85), (5, 105)],
                [
                    f"Smoothing: {self.hann_smoothing}",
                    f"Amplification: {self.phase_amplification}",
                    f"Mask: {self.centerband_mask}%",
                    f"Sideband: {self.sideband_area}{' (L)' if self.sideband_lock else ''}",
                    f"Contrast: {np.round(100 * self.fringe_contrast, 2)}%",
                ],
            ):
                self.font.render_to(self.screen, coordinate, message, pg.Color("RED"))

            pg.display.flip()

    def reconstruct(self):
        img_CCD = self.microscope.acquire()

        img_fft = (
            image.rfft2_to_fft2(img_CCD.shape, sfft.rfft2(img_CCD))
            if self.real_FFT
            else sfft.fft2(img_CCD)
        )
        img_fft_shifted = sfft.fftshift(img_fft)

        if self.sideband_area == "upper":
            img_fft_shifted[img_fft_shifted.shape[0] // 2 :, :] = 0
        elif self.sideband_area == "lower":
            img_fft_shifted[: img_fft_shifted.shape[0] // 2, :] = 0

        cb_rr, cb_cc = draw.disk(
            np.asarray(img_fft_shifted.shape) / 2,
            min(img_fft_shifted.shape) * (self.centerband_mask / 100),
        )
        img_fft_shifted[cb_rr, cb_cc] = 0

        if not self.sideband_lock:
            self.sideband_position = np.unravel_index(
                img_fft_shifted.argmax(), img_fft_shifted.shape
            )
            self.sideband_distance = np.linalg.norm(
                np.asarray(img_fft_shifted.shape) / 2 - self.sideband_position
            )

        self.fringe_contrast = (
            2
            * np.abs(img_fft_shifted[tuple(self.sideband_position)])
            / np.abs(img_fft[0, 0])
        )

        sb_rr, sb_cc = draw.rectangle(
            np.asarray(self.sideband_position) - self.sideband_distance / 6,
            extent=self.sideband_distance / 3,
        )
        img_cutout = img_fft_shifted[sb_rr, sb_cc]

        if self.hann_smoothing:
            img_cutout *= filters.window("hann", img_cutout.shape)

        img_cutout_padded = image.pad_image(
            img_cutout,
            (self.dimension, self.dimension),
            mode="constant",
            constant_values=0,
        )

        self.object_image_wave = sfft.ifft2(img_cutout_padded)
        reconstructed_image_wave = (
            self.object_image_wave / self.reference_image_wave
            if self.reference_image_wave is not None
            else self.object_image_wave
        )

        return (
            np.abs(reconstructed_image_wave)
            if self.reconstruct_amplitude
            else np.angle(reconstructed_image_wave)
        )

    def save_screenshot(self, format="png"):
        pg.image.save(
            self.screen,
            f"HoloLive_{'PH' if not self.reconstruct_amplitude else 'AMP'}_{format(datetime.now(), '%Y-%m-%d_%H-%M-%S')}.{format}",
        )
