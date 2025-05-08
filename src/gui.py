import contextlib
import datetime
import itertools as it
import multiprocessing
import os
import sys
import threading

import numpy as np
import scipy.fft as sfft
from skimage import draw, restoration

os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "hide"

import pygame as pg

from . import filter, image, microscope


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
        self.dimension = (dimension, dimension)

        self.microscope = microscope.Microscope(ip, port, remote)
        self.microscope.configure_camera(camera, exposure_time, binning)

        self.img_queue = multiprocessing.Queue(maxsize=10)
        self.img_CCD = np.ones(self.microscope.get_image_size())

        self.acquire_process = multiprocessing.Process(target=self.acquire)
        self.acquire_process.start()

        self.sideband_position, self.sideband_distance = (0, 0), 0
        self.sideband_area, self.sideband_lock = "upper", False

        self.amplifications = it.islice(it.cycle([1, 2, 3, 4]), 1, None)
        self.phase_amplification = 1

        self.cutout_filter, self.filter_cutoff = True, 0.2
        self.centerband_mask = 2

        self.object_image_wave, self.reference_image_wave = None, None
        self.fringe_contrast = 0

        self.reconstruct_amplitude = False
        self.unwrap_phase = False

        self.pause = False

        pg.init()

        self.font = pg.freetype.SysFont(None, 18)
        self.screen = pg.display.set_mode(self.dimension, pg.RESIZABLE)

        pg.display.set_caption("HoloLive")

        self.run()

    def run(self):
        while True:
            for event in pg.event.get():
                if event.type == pg.QUIT:
                    self.acquire_process.terminate()
                    self.acquire_process.join()

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
                    if event.key == pg.K_LEFT:
                        self.sideband_area = "left"
                    if event.key == pg.K_RIGHT:
                        self.sideband_area = "right"

                    if event.key == pg.K_GREATER:
                        self.centerband_mask += 1
                    if event.key == pg.K_LESS and self.centerband_mask > 1:
                        self.centerband_mask -= 1

                    if event.key == pg.K_TAB:
                        self.reconstruct_amplitude = not self.reconstruct_amplitude
                        self.unwrap_phase = False

                        self.amplifications = it.islice(it.cycle([1, 2, 3, 4]), 1, None)
                        self.phase_amplification = 1

                    if event.key == pg.K_a and not self.reconstruct_amplitude:
                        self.phase_amplification = next(self.amplifications)

                    if event.key == pg.K_u and not self.reconstruct_amplitude:
                        self.unwrap_phase = not self.unwrap_phase

                    if event.key == pg.K_r:
                        self.reference_image_wave = self.object_image_wave.copy()
                        self.sideband_lock = True

                    if event.key == pg.K_s:
                        self.save_screenshot_thread = threading.Thread(
                            target=self.save_screenshot
                        )
                        self.save_screenshot_thread.start()

                    if event.key == pg.K_p:
                        self.pause = not self.pause

                    if event.key == pg.K_f:
                        self.cutout_filter = not self.cutout_filter

                    if event.key == pg.K_PLUS and self.filter_cutoff <= 0.99:
                        self.filter_cutoff += 0.01
                    if event.key == pg.K_MINUS and self.filter_cutoff >= 0.01:
                        self.filter_cutoff -= 0.01

            if not self.pause:
                self.current_reconstruction = self.reconstruct()

                if self.phase_amplification != 1 and not self.reconstruct_amplitude:
                    amplified_image_wave = 1j * self.current_reconstruction
                    amplified_image_wave *= self.phase_amplification

                    self.current_reconstruction = np.angle(np.exp(amplified_image_wave))

                if self.unwrap_phase and not self.reconstruct_amplitude:
                    self.current_reconstruction = restoration.unwrap_phase(
                        self.current_reconstruction
                    )

                reconstruction_gray = image.convert_gray(self.current_reconstruction)

            pg_surface = pg.surfarray.make_surface(reconstruction_gray)
            pg_display_size = pg.display.get_surface().get_size()

            surface_phase_image = pg.transform.smoothscale(pg_surface, pg_display_size)
            self.screen.blit(surface_phase_image, (0, 0))

            if self.reference_image_wave is not None:
                pg.draw.rect(self.screen, pg.Color("RED"), [0, 0, *pg_display_size], 4)

            # fmt: off
            annotations = [
                ((5, 5), f"Filter: {self.cutout_filter} ({np.round(self.filter_cutoff, 2)})"),
                ((5, 25), f"Phase: {self.phase_amplification}x{' (U)' if self.unwrap_phase else ''}"),
                ((4, 45), f"Mask: {self.centerband_mask}%", pg.Color("RED")),
                ((5, 85), f"Sideband: {self.sideband_area}{' (L)' if self.sideband_lock else ''}"),
                ((5, 105), f"Contrast: {np.round(100 * self.fringe_contrast, 2)}%")
            ]
            # fmt: on

            for annotation in annotations:
                self.font.render_to(self.screen, *annotation, pg.Color("RED"))

            pg.display.flip()

    def acquire(self):
        while True:
            with contextlib.suppress(BaseException):
                self.img_queue.put_nowait(self.microscope.acquire())

    def reconstruct(self):
        with contextlib.suppress(BaseException):
            self.img_CCD = self.img_queue.get_nowait()

        img_fft = sfft.fft2(self.img_CCD)
        img_fft_shifted = sfft.fftshift(img_fft)

        match self.sideband_area:
            case "upper":
                img_fft_shifted[img_fft_shifted.shape[0] // 2 :, :] = 0
            case "lower":
                img_fft_shifted[: img_fft_shifted.shape[0] // 2, :] = 0
            case "left":
                img_fft_shifted[:, img_fft_shifted.shape[1] // 2 :] = 0
            case "right":
                img_fft_shifted[:, : img_fft_shifted.shape[1] // 2] = 0
            case other:
                raise ValueError(f"Error! Unsupported Sideband Area: {other}!")

        cb_center = np.asarray(img_fft_shifted.shape) / 2
        cb_radius = min(img_fft_shifted.shape) * (self.centerband_mask / 100)
        cb_rr, cb_cc = draw.disk(cb_center, cb_radius)

        img_fft_shifted[cb_rr, cb_cc] = 0

        if not self.sideband_lock:
            sb_index = img_fft_shifted.argmax()

            self.sideband_position = np.unravel_index(sb_index, img_fft_shifted.shape)
            self.sideband_distance = np.linalg.norm(cb_center - self.sideband_position)

        sb_intensity = np.abs(img_fft_shifted[tuple(self.sideband_position)])
        cb_intensity = np.abs(img_fft[0, 0])

        self.fringe_contrast = 2 * sb_intensity / cb_intensity

        sb_boundary = np.asarray(self.sideband_position) - self.sideband_distance / 6
        sb_rr, sb_cc = draw.rectangle(sb_boundary, extent=self.sideband_distance / 3)

        img_cutout = img_fft_shifted[sb_rr, sb_cc]

        if self.cutout_filter:
            img_cutout *= filter.window(("tukey", self.filter_cutoff), img_cutout.shape)

        img_cutout_padded = image.pad_image(
            img_cutout, self.dimension, mode="constant", constant_values=0
        )

        self.object_image_wave = sfft.ifft2(sfft.ifftshift(img_cutout_padded))

        reconstructed_image_wave = self.object_image_wave.copy()

        if self.reference_image_wave is not None:
            reconstructed_image_wave /= self.reference_image_wave

        if self.reconstruct_amplitude:
            return np.abs(reconstructed_image_wave)

        return np.angle(reconstructed_image_wave)

    def save_screenshot(self, extension="png"):
        type = "PH" if not self.reconstruct_amplitude else "AMP"
        timestamp = format(datetime.datetime.now(), "%Y-%m-%d_%H-%M-%S")

        pg.image.save(self.screen, f"HoloLive_{type}_{timestamp}.{extension}")
