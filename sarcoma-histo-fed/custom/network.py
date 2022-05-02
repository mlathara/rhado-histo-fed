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

    # need tfma for confusion matrix if we want it...
    metrics = [
        tf.keras.metrics.CategoricalAccuracy(name="categorical_accuracy"),
        tf.keras.metrics.AUC(name="auc", curve="ROC"),
        tf.keras.metrics.AUC(name="auc_precision_recall", curve="PR"),
        tf.keras.metrics.FalsePositives(name="false_positives"),
        tf.keras.metrics.FalseNegatives(name="false_negatives"),
        tf.keras.metrics.TruePositives(name="true_positives"),
        tf.keras.metrics.TrueNegatives(name="true_negatives"),
    ]

    model.compile(optimizer="adam", loss=loss_fn, metrics=metrics)

    return model


# TODO include random flip and rescaling layers here?
