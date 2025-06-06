# AI Raw/RGB Image Denoising Tools

## Introduction

This repository contains Python code that implements two command-line tools for image denoising:

1.  **RGB Image Denoising (`rgb.py`)**: Denoises standard RGB images (e.g. PNG).
2.  **Raw Image Denoising and Processing (`dng.py`)**: Processes digital negative (DNG) raw image files captured on a Raspberry Pi using either `rpicam-still` or Picamera2. The tool provides options for denoising, defective pixel correction, lens shading correction, and more before converting to RGB or saving as a new DNG.

Both tools use TFLite neural network models (via the `ai_edge_litert` library) to perform the denoising.

## Denoising Example

Here's an example of the capabilities of these tools. The original noisy image is on the left, and the denoised version on the right:

![Denoising Example](example.jpg)

## Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/raspberrypi/AI_denoise
    ```

2.  **Install dependencies:**

    These instructions are for use on Raspberry Pi OS. Note that the tools will in principle run
    on any platform, but there may be differences in the installation procedure. There are therefore no
    dependencies on Raspberry Pi camera software, such as libcamera or Picamera2.

    We assume that Python 3 is already installed, along with numpy.

    First execute:
    ```bash
    sudo apt install -y python3-opencv python3-rawpy libexiv2-dev libboost-python-dev
    ```
    Next, install the following Python packages using pip:
    ```bash
    pip install py3exiv2 tqdm ai-edge-litert
    ```

## Quick Start Tutorial

Once you've completed the installation, let's quickly try the tools out to see what they do.

First of all, check you're in the directory where you cloned the `AI_denoise` repository, and then capture
a noisy RGB image and a noisy DNG file as follows.
```bash
rpicam-still -r -e png -o noisy.png --denoise off --sharpness 0 --gain 12
```
Obsserve that:
* We're capturing both a DNG and a PNG file. The PNG codec is recommended over JPG for use with the RGB denoise model.
* Denoise and sharpening are both turned off, as this is recommended for the RGB denoise model too.
* The gain has been set high so that we have some noise to get rid of!

To denoise the RGB image we can use:
```bash
AI_denoise/rgb.py --input noisy.png --output denoised.jpg
```
Be sure to compare the two images!

Next, we're going to denoise the DNG file. We're going apply _both_ denoise _and_ lens shading correction as calibrated in the camera tuning file, and write out a new DNG. Then we're also going to convert this into
a finished RGB image for us to inspect. Both the output files - the new DNG and the RGB output - are optional.
```bash
AI_denoise/dng.py --input noisy.dng --denoise on --output-dng denoised_small.dng --output-rgb denoised_small.jpg
```
Again, note that:
* We have turned denoise on, but LSC (lens shaing correction) is enabled by default.
* Both output files have been named as "small", because by default the smaller of the two Bayer denoise models is chosen.

Finally, we're going to do the same thing but using the large denoise model. We simply repeat what we did before, changing the output filenames, and specifying the alternative neural network that we wish to use.
```bash
AI_denoise/dng.py --input noisy.dng --denoise on --output-dng denoised_large.dng --output-rgb denoised_large.jpg --network AI_denoise/networks/nafnet_bayer_large.tflite
```
Spend a few moments comparing the three denoised JPG files to each other and to the original noisy image. Also don't forget to look at the two new DNG files (comparing them to the original again) with your favourite raw converter!

## More on RGB Image Denoising (`rgb.py`)

The `rgb.py` script takes an RGB image (e.g. PNG, JPG) as input, applies a denoising neural network, and saves the output.

**Usage:**

```bash
python3 rgb.py --input <input_image_file> --output <output_image_file> [options]
```

**Key Options:**

*   `--input`: (Required) Path to the input RGB image file.
*   `--output`: (Required) Path to save the denoised RGB image.
*   `--network`: Path to the TFLite model for RGB denoising. Defaults to `networks/nafnet_rgb.tflite`.
*   `--overlap <pixels>`: Number of overlap pixels for patch-based processing (default: `16`).
*   `-y`, `--yes`: Overwrite the output file if it already exists.

Please enter `python3 rgb.py --help` for further information.

We recommend using the PNG image format for the input images. Being lossless, PNG preserves the noise structure that the neural network model was trained on. For the same reason, images should also be captured with no denoise or sharpening
applied to them by the Raspberry Pi's imaging pipeline.

**Example:**

```bash
python3 rgb.py --input noisy_photo.png --output denoised_photo.jpg
```

## More on Raw Image Processing & Denoising (`dng.py`)

The `dng.py` script processes DNG raw files. It can apply various corrections and enhancements to the raw sensor data before either saving it as a new DNG file or converting it to a standard RGB image format.

**Usage:**

```bash
python3 dng.py --input <input_dng_file> [options]
```

**Key Options:**

*   `--input`: (Required) Path to the input DNG file.
*   `--output-dng`: Path to save the processed raw data as a new DNG file.
*   `--output-rgb`: Path to save the processed and converted image as an RGB file (e.g. JPG, PNG).
*   `--network`: Path to the TFLite model for raw (Bayer) denoising. Defaults to `networks/nafnet_bayer_small.tflite`.
*   `--denoise <on|off>`: Enable or disable neural network denoising (default: `off`).
*   `--dpc <on|off>`: Enable or disable Defective Pixel Correction (default: `off`).
*   `--lsc <on|off>`: Enable or disable Lens Shading Correction (default: `on`).
*   `--digital-gain <value>`: Apply extra digital gain (default: `1.0`).
*   `--gamma <r_gamma,b_gamma>`: Comma-separated gamma values (e.g. "2.2,4.5") which are interpreted in the same way that [`rawpy.postprocess`](https://letmaik.github.io/rawpy/api/rawpy.Params.html) does. Defaults to tuning file gamma.
*   `--colour-gains <r_gain,b_gain>`: Comma-separated red and blue channel gains (e.g. "1.7,2.3"). Defaults to gains from DNG metadata.
*   `--tuning <tuning_file.json>`: Path to a custom camera tuning file.
*   `--sensor <sensor_name>`: Specify sensor model (e.g. `imx477`), sometimes needed if not in DNG metadata.
*   `--overlap <pixels>`: Number of overlap pixels for patch-based processing (default: `16`).
*   `-y`, `--yes`: Overwrite output file(s) if they already exist.

Please enter `python3 dng.py --help` for further information.

Note that the `--sensor` (or just `-s`) option may be needed with DNG files from Picamera2. Picamera2 has historically not included
the sensor name in the DNG metadata, so it must supplied externally (for example, `-s imx477`). Going forward,
this problem will be fixed.

To use a more specialised camera tuning file, such as `imx477_scientific.json`, you can specify it with the `--tuning` parameter.

**Examples:**

1.  **Denoise a DNG and save as RGB:**
    ```bash
    python3 dng.py --input image.dng --denoise on --output-rgb denoised_image.jpg
    ```

2.  **Apply DPC, LSC, denoise with a specific model, and save as a new DNG:**
    ```bash
    python3 dng.py --input image.dng --dpc on --lsc on --denoise on --network networks/nafnet_bayer_large.tflite --output-dng processed_image.dng
    ```
    We note that DPC is normally only required with the imx219 sensor (v2 camera) as the other sensors generally include it as a built-in feature.

3.  **Convert DNG to JPG with custom gamma and color gains:**
    ```bash
    python3 dng.py --input image.dng --output-rgb custom_render.jpg --gamma "2.2,4.5" --colour-gains "1.8,2.1"
    ```

## Networks and Performance

There are currently 3 networks provided.

1. `nafnet_rgb.tflite` - this is an approximately 7M (million) parameter model which performs high quality denoising of regular RGB images. On a Pi 5 it runs at approximately 12 seconds per megapixel (s/MP).

2. `nafnet_bayer_large.tflite` - this is the larger of the Bayer (raw image) denoise models, with about 17M parameters. But because there are fewer Bayer sample values than in an equivalent resolution RGB image, it actually performs high quality denoising faster than the RGB model, at about 6 s/MP on a Pi 5.

3. `nafnet_bayer_small.tflite` - this is a much smaller model for Bayer image denoising, having only about 1M parameters. It's performance is generally very good, but can be slightly inferior to the other models in some circumstances. It runs at about 3 s/MP (again on a Pi 5).

Some other general points to note:

* The models are trained on Raspberry Pi camera images taken at full resolution. In all other cases, results may vary.
* After denoising, the models tend to be a little soft and will benefit from some sympathetic sharpening.
* All models are based on [NAFNet](https://arxiv.org/pdf/2204.04676v4).

## License

This project is licensed under the BSD 2-Clause License. See the [LICENSE](LICENSE) file for details.