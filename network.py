from ai_edge_litert.interpreter import Interpreter
import numpy as np
from tqdm import tqdm

class Network:
    """
    A class to represent a neural network model implemented in TFLite.

    The underlying TFLite model is expected to take square patches of pixels as input, and return a
    square patch of the same size as output. The Network wrapper takes care of splitting an image
    into patches, padding them as necessary at the edges, running them through the neural network,
    and finally reassembling all the patches for the output image.
    """

    def __init__(self, model_path: str, num_threads: int = 4) -> None:
        """
        Initialize the Network object by loading the TFLite model.

        Args:
            model_path (str): Path to the TFLite model file
            num_threads (int): Number of threads to use for inference
        """
        if not model_path.endswith(".tflite"):
            raise ValueError("Model path must end with .tflite")
        self.interpreter = Interpreter(model_path, num_threads=num_threads)
        self.interpreter.allocate_tensors()
        input_shape = self.interpreter.get_input_details()[0]['shape']
        if input_shape[1] != input_shape[2]:
            raise ValueError("Patches must be square")
        self.patch_size = input_shape[1]

    def _calculate_patch_info(
            self,
            image_shape: tuple[int, int, int],
            overlap_pixels: int
            ) -> tuple[int, int, int, int, int]:
        """
        Calculate the number of patches and padding required to split an image into overlapping patches.

        Args:
            shape: Tuple of (height, width, channels) of the input image
            overlap_pixels: Number of pixels to overlap between patches

        Returns:
            Tuple of (stride, num_patches_w, num_patches_h, width_padded, height_padded) where:
            - stride: The stride between patches
            - num_patches_w: Number of patches in the width dimension
            - num_patches_h: Number of patches in the height dimension
            - width_padded: Width of the padded image needed to make complete patches
            - height_padded: Height of the padded image needed to make complete patches

        Note:
            The function ensures that the image can be split into complete patch_size x patch_size patches
            by calculating the necessary padding. The stride between patches is patch_size - overlap_pixels.
        """
        # Get dimensions
        height, width, _ = image_shape

        # Patch-to-patch stride (patch_size - overlap)
        stride = self.patch_size - overlap_pixels

        # Calculate the number of patches in each dimension, allowing an extra possibly imcomplete patch
        # at the end of each dimension.
        num_patches_w = (width - self.patch_size + stride - 1) // stride + 1
        num_patches_h = (height - self.patch_size + stride - 1) // stride + 1

        # Calculate the padded dimensions of the image to allow for those incomplete patches.
        width_padded = (num_patches_w - 1) * stride + self.patch_size
        height_padded = (num_patches_h - 1) * stride + self.patch_size

        return stride, num_patches_w, num_patches_h, width_padded, height_padded

    def _split_into_patches(self, image: np.ndarray,
                            patch_info: tuple[int, int, int, int, int]) -> np.ndarray:
        """
        Split an image into overlapping patch_size x patch_size patches with reflection padding.

        Args:
            image: Input image as a numpy array of shape (height, width, channels)
            patch_info: Tuple from calculate_patch_info with all the info about how to make the patches

        Returns:
            Array of patches with shape (num_patches, patch_size, patch_size, channels) where:
            - num_patches = num_patches_w * num_patches_h
            - Each patch is patch_size x patch_size pixels
            - channels is preserved from input image

        Note:
            The function pads the input image using reflection padding to ensure
            complete patches at the edges.
        """
        height, width, _ = image.shape

        # Retrieve all the info about how many patches we need, how much padding etc.
        stride, num_patches_w, num_patches_h, width_padded, height_padded = patch_info

        # Create padded image with reflection padding
        img_padded = np.pad(
            image,
            ((0, height_padded - height), (0, width_padded - width), (0, 0)),
            mode='reflect'
        )

        # Extract patches.
        patches = []
        for i in range(num_patches_h):
            for j in range(num_patches_w):
                h_start = i * stride
                w_start = j * stride
                patch = img_padded[h_start:h_start + self.patch_size, w_start:w_start + self.patch_size]
                patches.append(patch)

        return np.array(patches)

    def _reassemble_patches(
            self,
            image_shape: tuple[int, int, int],
            overlap_pixels: int,
            patch_info: tuple[int, int, int, int, int],
            processed_patches: np.ndarray
        ) -> np.ndarray:
        """
        Reassemble processed patches into a single image using linear blending in overlap regions.

        Args:
            image_shape: Tuple of (height, width, channels) of the input image
            overlap_pixels: Number of pixels to overlap between patches
            patch_info: Tuple from calculate_patch_info with all the info about the patches
            processed_patches: Array of processed patches with shape (num_patches, patch_size, patch_size, channels)

        Returns:
            Reassembled image as a numpy array with shape (height, width, channels)

        Note:
            The function uses linear blending in the overlap regions to create smooth transitions
            between patches. The blending weights are generated using a linear ramp from 0 to 1
            across the overlap region. The final image is cropped to the original dimensions
            by removing the padding added during patch extraction.
        """
        height, width, channels = image_shape

        # Get all the patch related info.
        stride, num_patches_w, num_patches_h, width_padded, height_padded = patch_info

        # Create output image accumulator
        output_img = np.zeros((height_padded, width_padded, channels), dtype=np.float32)

        # Generate weights for linear blending in the overlap regions.
        weights = np.array([1.0] * self.patch_size)
        weights[:overlap_pixels] = np.linspace(0.0, 1.0, overlap_pixels)
        weights_left_overlap, weights_top_overlap = np.meshgrid(weights, weights)
        weights_right_overlap, weights_bottom_overlap = np.meshgrid(weights[::-1], weights[::-1])
        weights_left_overlap = weights_left_overlap[..., np.newaxis]
        weights_top_overlap = weights_top_overlap[..., np.newaxis]
        weights_right_overlap = weights_right_overlap[..., np.newaxis]
        weights_bottom_overlap = weights_bottom_overlap[..., np.newaxis]

        for i in range(num_patches_h):
            for j in range(num_patches_w):
                patch_idx = i * num_patches_w + j
                h_start = i * stride
                w_start = j * stride

                patch = processed_patches[patch_idx]
                if i != 0:
                    patch = patch * weights_top_overlap
                if i != num_patches_h - 1:
                    patch = patch * weights_bottom_overlap
                if j != 0:
                    patch = patch * weights_left_overlap
                if j != num_patches_w - 1:
                    patch = patch * weights_right_overlap

                output_img[h_start:h_start + self.patch_size, w_start:w_start + self.patch_size] += patch

        # Remove padding.
        output_img = output_img[:height, :width]

        return output_img

    def _process_patches(self, patches: np.ndarray, show_progress: bool = False) -> np.ndarray:
        """
        Process patches through the TFLite interpreter.

        Args:
            patches: Array of patches to process, shape (n_patches, PATCH_SIZE, PATCH_SIZE, 3)
            show_progress: Whether to show a progress bar (default: False)

        Returns:
            Processed patches array with same shape as input
        """
        # Get input and output tensors
        input_details = self.interpreter.get_input_details()
        output_details = self.interpreter.get_output_details()

        # Check if model is quantized (INT8)
        is_quantized = input_details[0]['dtype'] == np.int8

        # Process patches one by one for TFLite
        processed_patches = []
        patch_iter = tqdm(patches, desc="Denoising patches") if show_progress else patches
        for patch in patch_iter:
            # Scale input for INT8 models
            if is_quantized:
                input_scale = input_details[0]['quantization'][0]
                input_zero_point = input_details[0]['quantization'][1]
                patch = patch / input_scale + input_zero_point
                patch = np.array(patch).astype(np.int8)

            # Set input tensor
            self.interpreter.set_tensor(input_details[0]['index'], np.expand_dims(patch, 0))

            # Run inference
            self.interpreter.invoke()

            # Get output tensor
            output = self.interpreter.get_tensor(output_details[0]['index'])

            # De-scale output for INT8 models
            if is_quantized:
                output_scale = output_details[0]['quantization'][0]
                output_zero_point = output_details[0]['quantization'][1]
                output = (output.astype(np.float32) - output_zero_point) * output_scale

            processed_patches.append(output[0])

        return np.array(processed_patches)

    def run_inference(self, image: np.ndarray, overlap_pixels: int = 16, show_progress: bool = False) -> np.ndarray:
        """
        Break the image up into patches, run them all through the neural network model, and
        reassemble them to make the output image.

        Args:
            image: The image to run inference on, shape (height, width, channels)
            overlap_pixels: The number of pixels to overlap between patches
            show_progress: Whether to show a progress bar (default: False)

        Returns:
            The output image.
        """
        # This tells us how many patches we will need, how much to pad the image etc.
        patch_info = self._calculate_patch_info(image.shape, overlap_pixels)
        # Break the image up into patches.
        patches = self._split_into_patches(image, patch_info)
        # Run the patches through the neural network model.
        outputs = self._process_patches(patches, show_progress) # This is the slow part.
        # Reassemble the patches to make the output image.
        output_image = self._reassemble_patches(image.shape, overlap_pixels, patch_info, outputs)
        return output_image
