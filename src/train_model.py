import tensorflow as tf
from os import path
import click
import logging
import mlflow
from create_image_dataset import create_image_dataset
from preprocess_and_augment import preprocess_and_augment

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
