import tensorflow as tf


def create_image_dataset(data_dir, subset, img_height=224, img_width=224, batch_size=16):
    """
    Create a TensorFlow image dataset from a directory.

    Args:
        data_dir (str): The directory containing the image data.
        subset (str): Either "training" or "validation" to specify the dataset subset.
        img_height (int, optional): The height of the images in the dataset (default is 224).
        img_width (int, optional): The width of the images in the dataset (default is 224).
        batch_size (int, optional): The batch size for the dataset (default is 16).

    Returns:
        tf.data.Dataset: A TensorFlow dataset containing the image data.
    """
    return tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset=subset,
        seed=123,
        color_mode='rgb',
        image_size=(img_height, img_width),
        batch_size=batch_size
    )
