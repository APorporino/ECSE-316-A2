The main entry point of this program is fft.py
It can be called using this syntax.
    python fft.py [-m mode] [-i image]

where the argument is defined as follows:
mode (optional):
    - [1] (Default) for fast mode where the image is converted into its FFT form and displayed
    - [2] for denoising where the image is denoised by applying an FFT, truncating high frequencies and then displayed
    - [3] for compressing and saving the image
    - [4] for plotting the runtime graphs for the report
image (optional): filename of the image we wish to take the DFT of (Default: moonloanding.png)


Python 3.7 was used to run and test this program and requirements.txt file contains all necessary requirements.