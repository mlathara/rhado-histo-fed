from pkgutil import get_data
import tensorflow as tf

from nvflare.apis.dxo import DXO, DataKind, from_shareable
from nvflare.apis.fl_constant import ReturnCode
from nvflare.apis.event_type import EventType
from nvflare.apis.executor import Executor
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable, make_reply
from nvflare.apis.signal import Signal

from preprocess import slides_to_tiles
from network import build_model


def load_image(image, height=299, width=299):
    img = tf.io.read_file(image)
    img = tf.io.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [height, width])
    img = tf.expand_dims(img, axis=0)

    return tf.keras.applications.inception_v3.preprocess_input(img)


def get_dataset(files, batch_size, num_classes):
    paths = [f[0] for f in files]
    labels = [[f[1]] for f in files]

    ds = tf.data.Dataset.from_tensor_slices(paths)
    ds = ds.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)

    label_ds = tf.data.Dataset.from_tensor_slices(labels)
    label_ds = label_ds.map(
        lambda x: tf.one_hot(x, num_classes), num_parallel_calls=tf.data.AUTOTUNE
    )

    ds = tf.data.Dataset.zip((ds, label_ds))
    ds = ds.prefetch(buffer_size=batch_size)
    return ds


class SimpleTrainer(Executor):
    def __init__(
        self,
        epochs_per_round: int,
        slidepath: str,
        overlap: int,
        workers: int,
        output_base: str,
        quality: int,
        tile_size: int,
        background: float,
        magnification: float,
        labels_file: str,
        validation_split: float,
        flipmode: str,
    ):
        super().__init__()
        self.epochs_per_round = epochs_per_round
        self.train_ds = None
        self.validation_ds = None
        self.model = None
        self.slidepath = slidepath
        self.overlap = overlap
        self.workers = workers
        self.output_base = output_base
        self.quality = quality
        self.tile_size = tile_size
        self.background = background
        self.magnification = magnification
        self.labels_file = labels_file
        self.validation_split = validation_split
        if flipmode not in ["horizontal", "vertical", "horizontal_and_vertical"]:
            self.flipmode = None
        else:
            self.flipmode = flipmode

    def handle_event(self, event_type: str, fl_ctx: FLContext):
        if event_type == EventType.START_RUN:
            self.setup(fl_ctx)

    def setup(self, fl_ctx: FLContext):
        num_classes, train_files, validation_files = slides_to_tiles(
            self.slidepath,
            self.overlap,
            self.workers,
            self.output_base,
            self.quality,
            self.tile_size,
            self.background,
            self.magnification,
            self.labels_file,
            self.validation_split,
        )

        # args = ((self.tile_size, self.tile_size), 3, 'bilinear', False)
        # args = (False, 'rgb', (self.tile_size, self.tile_size), 'bilinear', False)

        self.train_ds = get_dataset(train_files, 32, num_classes)
        self.validation_ds = get_dataset(validation_files, 32, num_classes)
        """
        self.train_ds = tf.data.Dataset.from_tensor_slices(train_files)
        self.train_ds = self.train_ds.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
        self.train_ds = self.train_ds.prefetch(buffer_size=32)

        self.validation_ds = tf.data.Dataset.from_tensor_slices(validation_files)
        self.validation_ds = self.validation_ds.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
        self.validation_ds = self.validation_ds.prefetch(buffer_size=32)
        Ideally we do tiling of svs into jpegs here, and also normalize stuff
        - tile svs into jpegs (deeppath way or cucim??)
        - normalize (create keras layer for this in future? may be mor efficient?)
        - use tf.keras.preprocessing.image_dataset_from_directory to get training/validation splits
          https://stackoverflow.com/questions/64374691/apply-different-data-augmentation-to-part-of-the-train-set-based-on-the-category
        - then add image augmentation layers to model https://keras.io/guides/preprocessing_layers/#quick-recipes
          though this seems to be doing augmentation on cpu instead of gpu. maybe do this instead https://keras.io/guides/preprocessing_layers/#preprocessing-data-before-the-model-or-inside-the-model
          - if doing this image augmentation layer then define model as a class and define get_config and from_config methods https://github.com/keras-team/keras/issues/15699
          - downside is that this will augment all classes? we may only want to augment some?
        - alternative augmentation is using ImageDataGenerator. Specifically use multiple and stitch together https://stackoverflow.com/questions/67013645/appliyng-data-augmentation-to-all-but-one-class-in-python
          - labels do need to be fixed up here though...set the class_indices attribute/dictionary??
          - also ImageDataGenerator may be slower!
          - may be cleaner to use customized tf.data.Dataset https://stackoverflow.com/questions/64374691/apply-different-data-augmentation-to-part-of-the-train-set-based-on-the-category
          - also tf.data.Dataset route could be faster?
        - https://www.tensorflow.org/tutorials/images/data_augmentation
        - make sure to pass tf.keras.applications.inception_v3.preprocess_input to ImageDataGenerator (way to do it for image_dataset_from_directory)
          - this will normalize input so that it can be fed to model
          - maybe this can be made part of the model instead of calling
          - x /= 255.; x -= 0.5; x *= 2.

        """
        self.model = build_model((self.tile_size, self.tile_size))

    def execute(
        self,
        task_name: str,
        shareable: Shareable,
        fl_ctx: FLContext,
        abort_signal: Signal,
    ) -> Shareable:
        """
        This function is an extended function from the super class.
        As a supervised learning based trainer, the train function will run
        evaluate and train engines based on model weights from `shareable`.
        After finishing training, a new `Shareable` object will be submitted
        to server for aggregation.

        Args:
            task_name: dispatched task
            shareable: the `Shareable` object acheived from server.
            fl_ctx: the `FLContext` object achieved from server.
            abort_signal: if triggered, the training will be aborted.

        Returns:
            a new `Shareable` object to be submitted to server for aggregation.
        """

        # retrieve model weights download from server's shareable
        if abort_signal.triggered:
            return make_reply(ReturnCode.TASK_ABORTED)

        if task_name != "train":
            return make_reply(ReturnCode.TASK_UNKNOWN)

        dxo = from_shareable(shareable)
        model_weights = dxo.data

        # update local model weights with received weights
        for layer_idx in range(4):
            layer = self.model.get_layer(index=layer_idx)
            layer.set_weights(model_weights[layer.name])
        """
        self.model.set_weights(model_weights)
        """

        # adjust LR or other training time info as needed
        # such as callback in the fit function
        self.model.fit(
            self.train_ds, epochs=self.epochs_per_round, validation_data=self.validation_ds
        )

        # report updated weights in shareable
        """
        weights = {
            self.model.get_layer(index=key).name: value
            for key, value in enumerate(self.model.get_weights())
        }
        """
        weights = {
            self.model.get_layer(index=layer_idx)
            .name: self.model.get_layer(index=layer_idx)
            .get_weights()
            for layer_idx in range(4)
        }
        dxo = DXO(data_kind=DataKind.WEIGHTS, data=weights)

        self.log_info(fl_ctx, "Local epochs finished. Returning shareable")
        new_shareable = dxo.to_shareable()
        return new_shareable
