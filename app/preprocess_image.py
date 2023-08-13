import tensorflow as tf
from skimage import transform
import numpy as np


def preprocess_image(image_path):
    img_width = 224
    img_height = 224
    image = tf.keras.utils.load_img(image_path)
    input_arr = tf.keras.utils.img_to_array(image)
    transformed_arr = transform.resize(input_arr, (img_width, img_height, 3))
    normalized_arr = (transformed_arr.astype(float) - 128) / 128
    output_arr = np.array([normalized_arr])
    return output_arr
