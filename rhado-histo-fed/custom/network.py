import tensorflow as tf


# from https://stackoverflow.com/a/54648506
def flatten_model(model_nested):
    layers_flat = []
    for layer in model_nested.layers:
        try:
            layers_flat.extend(layer.layers)
        except AttributeError:
            layers_flat.append(layer)
    model_flat = tf.keras.models.Sequential(layers_flat)
    return model_flat


def build_model(tile_size: tuple):
    network = tf.keras.applications.inception_v3.InceptionV3(
        weights="imagenet", include_top=False, input_shape=(tile_size[0], tile_size[1], 3)
    )
    network.trainable = False
    inputs = tf.keras.Input(shape=(tile_size[0], tile_size[1], 3))
    x = network(inputs, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    outputs = tf.keras.layers.Dense(3, activation="softmax")(x)
    model = tf.keras.Model(inputs, outputs)
    loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
    model.compile(optimizer="adam", loss=loss_fn, metrics=["accuracy"])

    return model


class InceptionV3_Path(tf.keras.Model):
    def __init__(self, classes, flipmode):
        super().__init__()
        self.classes = classes
        self.flipmode = flipmode
        if flipmode:
            self.augment = tf.keras.Sequential([tf.keras.layers.RandomFlip(self.flipmode)])
        self.rescale = tf.keras.Sequential(
            [tf.keras.layers.Rescaling(1.0 / 127.5, offset=-1, input_shape=(299, 299, 3))]
        )

    def call(self, x):
        # if self.flipmode:
        #    x = self.augment(x)
        # x = self.rescale(x)
        # x = tf.keras.applications.inception_v3.preprocess_input(x)
        x = tf.keras.applications.inception_v3.InceptionV3(
            weights=None, include_top=True, classes=self.classes
        )(x)
        return x

    """
    TODO: do I need to specify get_config and from_config?
    https://github.com/keras-team/keras/issues/15699
    """
