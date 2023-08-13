import tensorflow as tf
import numpy as np
from skimage import transform
import click


def preprocess_image(image_path):
    img_width = 224
    img_height = 224
    image = tf.keras.utils.load_img(image_path)
    input_arr = tf.keras.utils.img_to_array(image)
    transformed_arr = transform.resize(input_arr, (img_width, img_height, 3))
    normalized_arr = (transformed_arr.astype(float) - 128) / 128
    output_arr = np.array([normalized_arr])
    return output_arr


@click.command()
@click.argument('image_path', type=click.Path())
@click.argument('model_path', type=click.Path())
def predict_pigmentation(image_path, model_path):
    model = tf.keras.saving.load_model(model_path)
    image = preprocess_image(image_path)
    is_pigmented = model.predict(image)
    return 'P' if np.argmax(is_pigmented) else 'N'


@click.command()
@click.argument('image_path', type=click.Path())
@click.argument('model_path', type=click.Path())
def predict_sensation(image_path, model_path):
    model = tf.keras.saving.load_model(model_path)
    image = preprocess_image(image_path)
    is_sensitive = model.predict(image)
    return 'S' if np.argmax(is_sensitive) else 'R'


@click.command()
@click.argument('image_path', type=click.Path())
@click.argument('model_path', type=click.Path())
def predict_oily(image_path, model_path):
    model = tf.keras.saving.load_model(model_path)
    image = preprocess_image(image_path)
    is_oily = model.predict(image)
    return 'O' if np.argmax(is_oily) else 'D'


@click.command()
@click.argument('image_path', type=click.Path())
@click.argument('model_path', type=click.Path())
def predict_wrinkles(image_path, model_path):
    model = tf.keras.saving.load_model(model_path)
    image = preprocess_image(image_path)
    is_wrinkled = model.predict(image)
    return 'W' if np.argmax(is_wrinkled) else 'T'


def predict(image_path,
            pigmentation_model_path,
            oily_model_path,
            sensation_model_path,
            wrinkles_model_path):
    image = preprocess_image(image_path)
    pigmentation_model = tf.keras.models.load_model(pigmentation_model_path)
    oily_model = tf.keras.models.load_model(oily_model_path)
    sensation_model = tf.keras.models.load_model(sensation_model_path)
    wrinkles_model = tf.keras.models.load_model(wrinkles_model_path)
    is_pigmented = pigmentation_model.predict(image)
    is_sensitive = sensation_model.predict(image)
    is_oily = oily_model.predict(image)
    is_wrinkled = wrinkles_model.predict(image)
    skin_type = 'O' if np.argmax(is_oily) else 'D'
    skin_type += 'S' if np.argmax(is_sensitive) else 'R'
    skin_type += 'P' if np.argmax(is_pigmented) else 'N'
    skin_type += 'W' if np.argmax(is_wrinkled) else 'T'
    return skin_type
