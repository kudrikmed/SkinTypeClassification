import tensorflow as tf


def preprocess_and_augment(dataset):
    """
    Preprocess and augment a TensorFlow image dataset.

    This function applies three common data preprocessing and augmentation techniques:
    1. Rescaling: Scales pixel values to the range [-1, 1].
    2. Random Horizontal and Vertical Flip: Randomly flips images horizontally and/or vertically.
    3. Random Rotation: Randomly rotates images by a factor of 0.2 radians.

    Args:
        dataset (tf.data.Dataset): The TensorFlow image dataset to preprocess and augment.

    Returns:
        tf.data.Dataset: The preprocessed and augmented TensorFlow image dataset.
    """
    preprocessing_layer = tf.keras.layers.Rescaling(scale=1. / 127.5, offset=-1.)
    random_flip_layer = tf.keras.layers.RandomFlip(mode="horizontal_and_vertical")
    random_rotation_layer = tf.keras.layers.RandomRotation(factor=0.2)

    dataset = dataset.map(lambda a, b: (preprocessing_layer(a), b))
    dataset = dataset.map(lambda a, b: (random_flip_layer(a), b))
    dataset = dataset.map(lambda a, b: (random_rotation_layer(a), b))

    return dataset
