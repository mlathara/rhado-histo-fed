import tensorflow as tf


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


# TODO include random flip and rescaling layers here?
