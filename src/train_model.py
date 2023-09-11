# import tensorflow as tf
# from os import path
# import click
# import logging
# import mlflow
#
#
# mlflow.set_experiment('Pretrained mobileNet')
# mlflow.tensorflow.autolog()
#
#
# @click.command()
# @click.argument('data_dir', type=click.types.Path(exists=True))
# @click.argument('model_name', type=click.types.STRING)
# @click.argument('optimizer', type=click.types.STRING)
# @click.argument('learning_rate', type=click.types.FLOAT)
# def train_model(data_dir: str, model_name: str, optimizer: str, learning_rate: float):
#     batch_size = 16
#     img_height = 224
#     img_width = 224
#     learning_rate = float(learning_rate)
#     logger = logging.getLogger(__name__)
#
#     logger.info('loading training dataset...')
#     train_ds = tf.keras.utils.image_dataset_from_directory(
#         data_dir,
#         validation_split=0.2,
#         subset="training",
#         seed=123,
#         color_mode='rgb',
#         image_size=(img_height, img_width),
#         batch_size=batch_size)
#
#     logger.info('loading validation dataset...')
#     val_ds = tf.keras.utils.image_dataset_from_directory(
#         data_dir,
#         validation_split=0.2,
#         subset="validation",
#         seed=123,
#         color_mode='rgb',
#         image_size=(img_height, img_width),
#         batch_size=batch_size)
#
#     # data normalization according to pretrained net docs
#     preprocessing_layer = tf.keras.layers.Rescaling(scale=1. / 127.5, offset=-1.)
#     train_ds = train_ds.map(lambda a, b: (preprocessing_layer(a), b))
#     val_ds = val_ds.map(lambda a, b: (preprocessing_layer(a), b))
#
#     # data augmentation
#     random_flip_layer = tf.keras.layers.RandomFlip(mode="horizontal_and_vertical")
#     train_ds = train_ds.map(lambda a, b: (random_flip_layer(a), b))
#     val_ds = val_ds.map(lambda a, b: (random_flip_layer(a), b))
#
#     random_rotation_layer = tf.keras.layers.RandomRotation(factor=0.2)
#     train_ds = train_ds.map(lambda a, b: (random_rotation_layer(a), b))
#     val_ds = val_ds.map(lambda a, b: (random_rotation_layer(a), b))
#
#     logger.info('loading pretrained model...')
#     mobile = tf.keras.applications.mobilenet.MobileNet()
#     x = mobile.layers[-5].output
#     x = tf.keras.layers.Reshape(target_shape=(1024,))(x)
#     output = tf.keras.layers.Dense(units=1, activation='sigmoid')(x)
#     model = tf.keras.models.Model(inputs=mobile.input, outputs=output)
#     for layer in model.layers[:-22]:
#         layer.trainable = False
#
#     if optimizer == 'SGD':
#         opt = tf.keras.optimizers.SGD(learning_rate)
#     elif optimizer == 'Adam':
#         opt = tf.keras.optimizers.Adam(learning_rate)
#
#     logger.info('Training model...')
#     model.compile(optimizer=opt,
#                   loss='binary_crossentropy',
#                   metrics=['accuracy',
#                            tf.keras.metrics.Precision(),
#                            tf.keras.metrics.Recall(),
#                            tf.keras.metrics.AUC()])
#
#     model.fit(x=train_ds,
#               validation_data=val_ds,
#               epochs=20,
#               verbose=2)
#
#     tf.keras.models.save_model(model, path.join('models/', model_name))
#     mlflow.tensorflow.log_model(
#         model=model,
#         artifact_path='models',
#         registered_model_name=model_name
#     )
#     logger.info(f'Model successfully saved as {model_name}')
#
#
# if __name__ == "__main__":
#     train_model()

import tensorflow as tf
from os import path
import click
import logging
import mlflow
from src.data.create_image_dataset import create_image_dataset
from src.data.preprocess_and_augment import preprocess_and_augment

mlflow.set_experiment('Pretrained mobileNet')
mlflow.tensorflow.autolog()


def build_mobilenet_model(learning_rate, optimizer):
    """
    Build a MobileNet-based TensorFlow model for binary classification.

    Args:
        learning_rate (float): The learning rate for the optimizer.
        optimizer (str): The optimizer to use ('SGD' or 'Adam').

    Returns:
        tf.keras.Model: The constructed TensorFlow model.
    """
    mobile = tf.keras.applications.mobilenet.MobileNet()
    x = mobile.layers[-5].output
    x = tf.keras.layers.Reshape(target_shape=(1024,))(x)
    output = tf.keras.layers.Dense(units=1, activation='sigmoid')(x)
    model = tf.keras.models.Model(inputs=mobile.input, outputs=output)

    # Freeze layers except the last 22
    for layer in model.layers[:-22]:
        layer.trainable = False

    # Choose optimizer
    if optimizer == 'SGD':
        opt = tf.keras.optimizers.SGD(learning_rate)
    elif optimizer == 'Adam':
        opt = tf.keras.optimizers.Adam(learning_rate)

    model.compile(optimizer=opt,
                  loss='binary_crossentropy',
                  metrics=['accuracy',
                           tf.keras.metrics.Precision(),
                           tf.keras.metrics.Recall(),
                           tf.keras.metrics.AUC()])

    return model


@click.command()
@click.argument('data_dir', type=click.types.Path(exists=True))
@click.argument('model_name', type=click.types.STRING)
@click.argument('optimizer', type=click.types.STRING)
@click.argument('learning_rate', type=click.types.FLOAT)
def train_model(data_dir: str, model_name: str, optimizer: str, learning_rate: float):
    batch_size = 16

    logger = logging.getLogger(__name__)
    logger.info('Loading datasets...')
    train_ds = create_image_dataset(data_dir, "training", batch_size=batch_size)
    val_ds = create_image_dataset(data_dir, "validation", batch_size=batch_size)

    logger.info('Preprocessing datasets...')
    train_ds = preprocess_and_augment(train_ds)
    val_ds = preprocess_and_augment(val_ds)

    logger.info('Loading pretrained model...')
    model = build_mobilenet_model(learning_rate, optimizer)

    logger.info('Training model...')
    model.fit(x=train_ds,
              validation_data=val_ds,
              epochs=20,
              verbose=2)

    model_dir = path.join('models/', model_name)
    tf.keras.models.save_model(model, model_dir)

    mlflow.tensorflow.log_model(
        model=model,
        artifact_path='models',
        registered_model_name=model_name
    )
    logger.info(f'Model successfully saved as {model_dir}')


if __name__ == "__main__":
    train_model()
