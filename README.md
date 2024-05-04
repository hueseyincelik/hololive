# HoloLive
Live phase and amplitude reconstruction for off-axis electron holography with FEI™ microscopes, written in Python using `Pygame` and `temscript`.

![Off-Axis Electron Holography Phase Reconstruction](/docs/TEM-off-axis-holography-reconstruction.png)

The retrieval of the $2\pi$-wrapped phase image works by taking the detector readout, Fourier transforming it such that the autocorrelation lies in the center, automatically finding the sideband in the specified image area, centering a cut-out around the sideband with a width of $1/3$ of the distance between the autocorrelation and the sideband through zero padding in Fourier space and calculating the angle of the complex argument of the inverse Fourier transform. The reconstructed amplitude in turn is defined as the the absolute value of the inverse Fourier transform.

Distortion-induced phase modulations can optionally be corrected using an empty hologram (i.e. without the specimen in the field of view) as a reference hologram.

Estimation of the fringe contrast in Fourier space is defined as twice the amplitude fraction of the sideband and the centerband (i.e. the autocorrelation).
## Installation
Install all required packages with pip using:
```
pip3 install -r requirements.txt
```

## Usage
The program can be started by simply running [`main.py`](/main.py):
```
python3 main.py
```
In order to remotely connect to the microscope, `temscript` has to be running on the accompanying PC (which in turn requires the TEMScripting software interface from Thermo Fisher Scientific™ and FEI™) and the corresponding IP address and port have to be provided.

Various keyboard shortcuts can be used to adjust some of the parameters used for the live reconstruction:
- `L` can be used to lock the sideband position, preventing it from being automatically recalculated until the lock is released again (note that disabling the sideband lock also resets/disables the reference hologram utilized during the reconstruction process)
- `+/-` can be used to adjust the radius of the circle used to mask the centerband, which is zeroed in order to avoid the influence of the autocorrelation during the automatic sideband detection, in steps of $1$% of the (minimum) image dimension
- `UP,DOWN` can be used to change the image area in which the sideband is automatically detected
- `TAB` can be used to switch between the reconstructed phase and amplitude
- `A` can be used to cycle between the values $1,2,3,4$ as an amplification factor for the reconstructed phase
- `H` can be used to apply Hann smoothing to the sideband cut-out before zero padding
- `R` can be used to utilize the current acquisition as a reference hologram during the reconstruction process (note that this also locks the sideband position)
- `S` can be used to save a screenshot of the currently reconstructed phase/amplitude (including on-screen information) as a timestamped PNG image
- `P` can be used to pause the acquisition of new phase/amplitude images

Although *HoloLive* should in principle work with any detector supported by `temscript`, it has only been tested with a Gatan™ US1000 CCD.

## License
Copyright © 2023 [Hüseyin Çelik](https://www.github.com/hueseyincelik).

This project is licensed under [AGPL v3](/LICENSE).
