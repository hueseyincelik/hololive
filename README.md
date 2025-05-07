# HoloLive
Live phase and amplitude reconstruction for off-axis electron holography with FEI™ microscopes, written in Python using `Pygame` and `temscript`.

![Off-Axis Electron Holography Phase Reconstruction](/docs/TEM-off-axis-holography-reconstruction.png)

The retrieval of the $2\pi$-wrapped phase image works by taking the detector readout, Fourier transforming it such that the autocorrelation lies in the center, automatically finding the sideband in the specified image area, centering a cut-out around the sideband with a width of $1/3$ of the distance between the autocorrelation and the sideband through zero padding in Fourier space and calculating the angle of the complex argument of the inverse Fourier transform. The reconstructed amplitude in turn is defined as the the absolute value of the inverse Fourier transform.

Distortion-induced phase modulations can optionally be corrected using an empty hologram (i.e. without the specimen in the field of view) as a reference hologram. Furthermore, a continuous phase image can be retrieved by applying a phase unwrapping algorithm that attempts to remove any $2\pi$-jumps present in the reconstruction.

Estimation of the fringe contrast in Fourier space is defined as twice the amplitude fraction of the sideband and the centerband (i.e. the autocorrelation).

## Performance
The performance of the live reconstruction (i.e. the number of frames that can be processed in a given time interval) heavily depends on the specific setup. For most combinations of microscope, detector, and image size, the majority of time is spent acquiring and transferring the hologram from the controller to the PC running `HoloLive`.

Excluding the time taken to acquire and transfer the hologram, a significant portion of the computation is related to various FFT operations. For commonly used image sizes of $2^n$, the different FFT implementations scale as follows (tested on an Apple™ M3 CPU, where $t_{rec}$ is the average value for $100$ consecutive reconstructions):

![HoloLive Performance Benchmark](/docs/performance_benchmark.svg)

For a typical $2048\text{px} \times 2048\text{px}$ image, the average reconstruction time $t_{rec}$ ranges from approximately $45\text{ms}$ to $85\text{ms}$, depending on the specific FFT implementation used. When the image is binned down to $512\text{px} \times 512\text{px}$, this time is reduced to around $6\text{ms}$ to $8\text{ms}$.

Since the data type of the input image is real, one can exploit the symmetry in the FFT by calculating only half of the frequency spectrum and inferring the other half (this is enabled by the `real_FFT` option). For `NumPy`'s FFT implementation, this can significantly speed up the FFT operation. `SciPy`, on the other hand, already takes advantage of this symmetry when detecting an input image of data type real, which can actually lead to a slowdown when using `real_FFT`.

Generally speaking, the performance ranking (from fastest to slowest) is:

```
scipy.fft.fft2 > scipy.fft.rfft2 > numpy.fft.rfft2 > numpy.fft.fft2
```
Therefore, it is recommended to always use the general `SciPy` implementation for 2D FFT.

Furthermore, applying `scikit-image`'s phase unwrapping algorithm to the reconstructed phase can result in a significant performance penalty (around $50\text{ms}$ in the above tested setup).

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
- `UP,DOWN` can be used to change the image area in which the sideband is automatically detected
- `LEFT/RIGHT` can be used to adjust the radius of the circle used to mask the centerband, which is zeroed in order to avoid the influence of the autocorrelation during the automatic sideband detection, in steps of $1$% of the (minimum) image dimension
- `TAB` can be used to switch between the reconstructed phase and amplitude
- `A` can be used to cycle between the values $1,2,3,4$ as an amplification factor for the reconstructed phase
- `F` can be used to apply a Tukey filter to the sideband cut-out before zero padding (alternatively, a Butterworth filter is available in [`image.py`](src/image.py))
- `+/-` can be used to adjust the shape parameter $\alpha$ of the Tukey filter, representing the fraction of the window inside the cosine tapered region, in steps of $0.01$
- `U` can be used to apply a phase unwrapping algorithm to the reconstructed phase to (attempt to) remove any $2\pi$-jumps
- `R` can be used to utilize the current acquisition as a reference hologram during the reconstruction process (note that this also locks the sideband position)
- `S` can be used to save a screenshot of the currently reconstructed phase/amplitude (including on-screen information) as a timestamped PNG image
- `P` can be used to pause the acquisition of new phase/amplitude images

Although *HoloLive* should in principle work with any detector supported by `temscript`, it has only been tested with a Gatan™ US1000 CCD.

## License
Copyright © 2023 [Hüseyin Çelik](https://www.github.com/hueseyincelik).

This project is licensed under [AGPL v3](/LICENSE).
