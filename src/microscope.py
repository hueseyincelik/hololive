from temscript import RemoteMicroscope


class Microscope:
    def __init__(self, ip, port):
        self.ip, self.port = ip, port
        self.microscope = RemoteMicroscope((self.ip, self.port))

    def configure_camera(self, camera, exposure_time):
        self.camera, self.exposure_time = camera, exposure_time
        self.microscope.set_camera_param(
            self.camera, {"exposure(s)": self.exposure_time}
        )

    def acquire(self):
        image = self.microscope.acquire(self.camera)
        return image[self.camera]
