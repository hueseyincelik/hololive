import temscript


class Microscope:
    def __init__(self, ip, port, remote):
        self.ip, self.port = ip, port
        self.microscope = (
            temscript.RemoteMicroscope((self.ip, self.port))
            if remote
            else temscript.Microscope()
        )

    def configure_camera(self, camera, exposure_time):
        self.camera, self.exposure_time = camera, exposure_time
        self.microscope.set_camera_param(
            self.camera, {"exposure(s)": self.exposure_time}
        )

    def acquire(self):
        image = self.microscope.acquire(self.camera)
        return image[self.camera]
