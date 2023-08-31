import tensorflow as tf
from tensorflow import keras
from keras.layers import Dense, Rescaling, RandomFlip, RandomRotation
from keras.models import Model
from os import path
import click
import logging
import mlflow


mlflow.set_experiment('Pretrained mobileNet')
mlflow.tensorflow.autolog()


@click.command()
@click.argument('data_dir', type=click.types.Path(exists=True))
@click.argument('model_name', type=click.types.STRING)
@click.argument('optimizer', type=click.types.STRING)
@click.argument('learning_rate', type=click.types.FLOAT)
def train_model(data_dir: str, model_name: str, optimizer: str, learning_rate: float):
    batch_size = 16
    img_height = 224
    img_width = 224
    learning_rate = float(learning_rate)
    logger = logging.getLogger(__name__)

    logger.info('loading training dataset...')
    train_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="training",
        seed=123,
        color_mode='rgb',
        image_size=(img_height, img_width),
        batch_size=batch_size)

    logger.info('loading validation dataset...')
    val_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="validation",
        seed=123,
        color_mode='rgb',
        image_size=(img_height, img_width),
        batch_size=batch_size)

    # data normalization according to pretrained net docs
    preprocessing_layer = Rescaling(scale=1. / 127.5, offset=-1.)
    train_ds = train_ds.map(lambda a, b: (preprocessing_layer(a), b))
    val_ds = val_ds.map(lambda a, b: (preprocessing_layer(a), b))

    # data augmentation
    random_flip_layer = RandomFlip(mode="horizontal_and_vertical")
    train_ds = train_ds.map(lambda a, b: (random_flip_layer(a), b))
    val_ds = val_ds.map(lambda a, b: (random_flip_layer(a), b))

    random_rotation_layer = RandomRotation(factor=0.2)
    train_ds = train_ds.map(lambda a, b: (random_rotation_layer(a), b))
    val_ds = val_ds.map(lambda a, b: (random_rotation_layer(a), b))

    logger.info('loading pretrained model...')
    mobile = tf.keras.applications.mobilenet.MobileNet()
    x = mobile.layers[-5].output
    x = tf.keras.layers.Reshape(target_shape=(1024,))(x)
    output = Dense(units=1, activation='sigmoid')(x)
    model = Model(inputs=mobile.input, outputs=output)
    for layer in model.layers[:-22]:
        layer.trainable = False

    if optimizer == 'SGD':
        opt = keras.optimizers.SGD(learning_rate)
    elif optimizer == 'Adam':
        opt = keras.optimizers.Adam(learning_rate)

    logger.info('Training model...')
    model.compile(optimizer=opt,
                  loss='binary_crossentropy',
                  metrics=['accuracy',
                           tf.keras.metrics.Precision(),
                           tf.keras.metrics.Recall(),
                           tf.keras.metrics.AUC()])

    model.fit(x=train_ds,
              validation_data=val_ds,
              epochs=20,
              verbose=2)

    tf.keras.saving.save_model(model, path.join('models/', model_name))
    mlflow.tensorflow.log_model(
        model=model,
        artifact_path='models',
        registered_model_name=model_name
    )
    logger.info(f'Model successfully saved as {model_name}')


if __name__ == "__main__":
    train_model()
