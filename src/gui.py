import os
import sys

import numpy as np
import scipy.fft as sfft

os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "hide"

import pygame as pg

from . import microscope


class GUI:
    def __init__(self, ip, port, camera="CCD", exposure_time=0.2, dimension=512):
        self.dimension = dimension

        self.microscope = microscope.Microscope(ip, port)
        self.microscope.configure_camera(camera, exposure_time)

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
        img_shift = sfft.fftshift(img_fft)
