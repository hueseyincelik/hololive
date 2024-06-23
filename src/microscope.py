import temscript


class Microscope:
    def __init__(self, ip, port, remote):
        self.ip, self.port = ip, port
        self.microscope = (
            temscript.RemoteMicroscope((self.ip, self.port))
            if remote
            else temscript.Microscope()
        )

    def __del__(self):
        if self.camera is not None:
            self.microscope.set_camera_param(
                self.camera,
                {
                    "exposure(s)": 4.0,
                    "binning": 1,
                },
            )

    def configure_camera(self, camera, exposure_time, binning):
        self.camera, self.exposure_time, self.binning = camera, exposure_time, binning
        self.microscope.set_camera_param(
            self.camera,
            {
                "exposure(s)": self.exposure_time,
                "binning": self.binning,
            },
        )

    def get_image_size(self):
        self.camera_properties = self.microscope.get_cameras()[self.camera]
        return (
            self.camera_properties["height"] // self.binning,
            self.camera_properties["width"] // self.binning,
        )

    def acquire(self):
        image = self.microscope.acquire(self.camera)
        return image[self.camera]
