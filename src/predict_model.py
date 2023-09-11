import tensorflow as tf
import numpy as np
from skimage import transform


def preprocess_image(image_path, img_width=224, img_height=224):
    """
    Preprocesses an image for skin type prediction.

    Args:
        image_path (str): Path to the input image.
        img_width (int, optional): Target width of the image (default is 224).
        img_height (int, optional): Target height of the image (default is 224).

    Returns:
        np.ndarray: Preprocessed image as a NumPy array.
    """
    image = tf.keras.utils.load_img(image_path)
    input_arr = tf.keras.utils.img_to_array(image)

    # Resize the image
    transformed_arr = transform.resize(input_arr, (img_width, img_height, 3))

    # Normalize the image
    normalized_arr = (transformed_arr.astype(float) - 128) / 128
    output_arr = np.array([normalized_arr])
    return output_arr


def predict_skin_type(image_path, pigmentation_model_path, oily_model_path, sensation_model_path, wrinkles_model_path):
    """
    Predicts skin type based on an input image and multiple skin feature models.

    Args:
        image_path (str): Path to the input image.
        pigmentation_model_path (str): Path to the pigmentation model file.
        oily_model_path (str): Path to the oily skin model file.
        sensation_model_path (str): Path to the skin sensitivity model file.
        wrinkles_model_path (str): Path to the wrinkles model file.

    Returns:
        str: Predicted skin type represented as a string, composed of the following letters:
             - 'O' for Oily
             - 'D' for Dry
             - 'S' for Sensitive
             - 'R' for Resilient (Not Sensitive)
             - 'P' for Pigmented
             - 'N' for Non-Pigmented
             - 'W' for Wrinkled
             - 'T' for Tight (Not Wrinkled)
    """
    # Preprocess the input image
    image = preprocess_image(image_path)

    # Load skin feature models
    pigmentation_model = tf.keras.models.load_model(pigmentation_model_path)
    oily_model = tf.keras.models.load_model(oily_model_path)
    sensation_model = tf.keras.models.load_model(sensation_model_path)
    wrinkles_model = tf.keras.models.load_model(wrinkles_model_path)

    # Predict skin features
    is_pigmented = pigmentation_model.predict(image)
    is_sensitive = sensation_model.predict(image)
    is_oily = oily_model.predict(image)
    is_wrinkled = wrinkles_model.predict(image)

    # Determine the skin type based on predictions
    skin_type = 'O' if is_oily[0][0] > 0.5 else 'D'  # Oily or Dry
    skin_type += 'S' if is_sensitive[0][0] > 0.5 else 'R'  # Sensitive or Resilient
    skin_type += 'P' if is_pigmented[0][0] > 0.5 else 'N'  # Pigmented or Non-Pigmented
    skin_type += 'W' if is_wrinkled[0][0] > 0.5 else 'T'  # Wrinkled or Tight
    return skin_type
