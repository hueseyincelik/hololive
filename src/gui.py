import os
import sys

import numpy as np
import scipy.fft as sfft

os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "hide"

import pygame as pg

from . import microscope


class GUI:
    def __init__(self, dimension=512):
        self.dimension = dimension

        pg.init()

        self.screen = pg.display.set_mode((self.dimension, self.dimension))
        pg.display.set_caption("Live Phase")

        self.run()

    def run(self):
        while True:
            for event in pg.event.get():
                if event.type == pg.QUIT:
                    sys.exit()
