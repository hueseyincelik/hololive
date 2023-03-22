import temscript


class Microscope:
    def __init__(self, ip, port, remote):
        self.ip, self.port = ip, port
        self.microscope = (
            temscript.RemoteMicroscope((self.ip, self.port))
            if remote
            else temscript.Microscope()
        )

    def configure_camera(self, camera, exposure_time, binning):
        self.camera, self.exposure_time, self.binning = camera, exposure_time, binning
        self.microscope.set_camera_param(
            self.camera, {"exposure(s)": self.exposure_time, "binning": self.binning}
        )

    def acquire(self):
        image = self.microscope.acquire(self.camera)
        return image[self.camera]
