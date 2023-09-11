import tensorflow as tf
from skimage import transform
import numpy as np


def preprocess_image(image_path, img_width=224, img_height=224):
    """
    Preprocesses an image for skin type prediction.

    Args:
        image_path (str): Path to the input image.
        img_width (int, optional): Target width of the image (default is 224).
        img_height (int, optional): Target height of the image (default is 224).

    Returns:
        np.ndarray: Preprocessed image as a NumPy array with normalized values.
    """
    # Load the image from the specified path
    image = tf.keras.utils.load_img(image_path)

    # Convert the image to a NumPy array
    input_arr = tf.keras.utils.img_to_array(image)

    # Resize the image to the specified dimensions
    transformed_arr = transform.resize(input_arr, (img_width, img_height, 3))

    # Normalize the pixel values to a range of [-1, 1]
    normalized_arr = (transformed_arr.astype(float) - 128) / 128

    # Create a batch-like array with a single element
    output_arr = np.array([normalized_arr])

    return output_arr
